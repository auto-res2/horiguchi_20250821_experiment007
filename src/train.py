import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utilities

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_selection() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def make_grid_coords(G: int = 8, device: str = 'cpu') -> torch.Tensor:
    xs = torch.linspace(-1.0, 1.0, G, device=device)
    ys = torch.linspace(-1.0, 1.0, G, device=device)
    xv, yv = torch.meshgrid(xs, ys, indexing='xy')
    coords = torch.stack([xv.reshape(-1), yv.reshape(-1)], dim=-1)  # (G*G, 2)
    return coords


def extract_patch(images: torch.Tensor, centers: torch.Tensor, p: int = 10) -> torch.Tensor:
    """
    images: (B,1,H,W) in [0,1]
    centers: (B,2) in [-1,1] normalized grid centers
    returns: (B, p*p) flattened patch
    """
    B, _, H, W = images.shape
    device = images.device
    # local coordinates in [-1,1]
    lin = torch.linspace(-1, 1, p, device=device)
    xv, yv = torch.meshgrid(lin, lin, indexing='xy')
    local = torch.stack([xv, yv], dim=-1)  # (p,p,2)
    local = local.view(1, p, p, 2).repeat(B, 1, 1, 1)
    # approximate scaling from normalized coords: scale patch area relative to whole image
    scale_x = p / float(W)
    scale_y = p / float(H)
    centers = centers.view(B, 1, 1, 2)
    grid = torch.empty_like(local)
    grid[..., 0] = centers[..., 0] + local[..., 0] * scale_x
    grid[..., 1] = centers[..., 1] + local[..., 1] * scale_y
    patch = F.grid_sample(images, grid, mode='bilinear', align_corners=False)
    patch = patch.view(B, -1)
    return patch


# Model components: Saccade Policy, Reservoir, ASRIN, Raster baseline

class SaccadePolicy(nn.Module):
    def __init__(self, grid_size: int = 8, patch_dim: int = 100, res_ctx_dim: int = 64, hid: int = 64):
        super().__init__()
        self.G = grid_size
        # Patch embedding
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh()
        )
        # GRU over policy context (prev patch embedding + reduced reservoir state)
        self.gru = nn.GRU(input_size=64 + res_ctx_dim, hidden_size=hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, self.G * self.G))
        self.coords = None  # populated by parent with make_grid_coords

    def forward(self, ctx_step: torch.Tensor, h0: torch.Tensor | None = None):
        # ctx_step: (B,1, 64+res_ctx_dim)
        out, h = self.gru(ctx_step, h0)
        logits = self.head(out)  # (B,1,V)
        return logits, h

    def sample_centers(self, logits: torch.Tensor, tau: float = 0.7, hard: bool = True):
        B, T, V = logits.shape
        gs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # (B,T,V)
        coords = (gs @ self.coords)  # (B,T,2)
        return coords, gs


class Reservoir(nn.Module):
    def __init__(self, N: int = 1000, in_dim: int = 100, sparsity: float = 0.05, leak: float = 0.3, win_scale: float = 0.5, device: str = 'cpu'):
        super().__init__()
        self.N, self.in_dim, self.leak = N, in_dim, leak
        self.W_in = nn.Parameter(torch.randn(N, in_dim, device=device) * win_scale, requires_grad=False)
        # sparse mask with ~sparsity fraction of nonzeros
        mask = (torch.rand(N, N, device=device) < sparsity).float()
        W = torch.randn(N, N, device=device) / math.sqrt(max(1e-6, sparsity) * N)
        W = W * mask
        # scale to spectral radius ~0.9 (rough)
        with torch.no_grad():
            v = torch.randn(N, 1, device=device)
            for _ in range(10):
                v = W @ v
                v = v / (v.norm() + 1e-8)
            eig_est = (v.T @ (W @ v)).abs().item()
            scale = 0.9 / (eig_est + 1e-6)
            W = W * scale
        self.W_rec = nn.Parameter(W, requires_grad=False)
        self.register_buffer('mask', (W != 0).float())
        # EC-SR gain buffer
        self.register_buffer('g', torch.zeros(1, device=device))
        self.g_eta = 1e-3
        self.g_max = 0.5
        self.s_target = 0.5

    @torch.no_grad()
    def reset_gain(self):
        self.g.zero_()

    def step(self, h: torch.Tensor, x: torch.Tensor):
        # x: (B, in_dim), h: (B, N)
        u = F.linear(x, self.W_in) + F.linear(h, (1.0 + self.g) * self.W_rec)
        h_tilde = torch.tanh(u)
        h_new = (1 - self.leak) * h + self.leak * h_tilde
        # Update EC-SR gain using activity proxy s_t = mean(1 - tanh(u)^2)
        with torch.no_grad():
            s_t = (1.0 - h_tilde.pow(2)).mean()
            delta = self.g_eta * (self.s_target - s_t)
            g_prop = torch.clamp(self.g + delta, -self.g_max, self.g_max)
            self.g.copy_(torch.clamp(0.9 * self.g + 0.1 * g_prop, -self.g_max, self.g_max))
        return h_new, s_t

    def nnz(self) -> int:
        return int((self.W_rec != 0).sum().item())


class ASRIN(nn.Module):
    def __init__(self, N: int = 1000, T: int = 30, p: int = 10, num_classes: int = 10, grid_size: int = 8, device: str = 'cpu'):
        super().__init__()
        self.N, self.T, self.p, self.G = N, T, p, grid_size
        self.res = Reservoir(N=N, in_dim=p*p, device=device)
        self.policy = SaccadePolicy(grid_size=grid_size, patch_dim=p*p, res_ctx_dim=64)
        self.z_dim = p*p + N
        self.readout = nn.Linear(self.z_dim, num_classes)
        # random projection for reservoir context (frozen)
        RP = torch.randn(64, N, device=device) / math.sqrt(N)
        self.register_buffer('res_proj', RP)
        coords = make_grid_coords(G=grid_size, device=device)
        self.policy.coords = coords

    def forward(self, images: torch.Tensor, tau: float = 0.7):
        B = images.size(0)
        device = images.device
        h = torch.zeros(B, self.N, device=device)
        # init previous patch
        prev_patch = torch.zeros(B, self.p*self.p, device=device)
        prev_emb = self.policy.patch_proj(prev_patch)
        logits_list, centers_list = [], []
        h_pol = None
        for t in range(self.T):
            res_ctx = F.linear(h.detach(), self.res_proj)  # (B,64)
            ctx = torch.cat([prev_emb, res_ctx], dim=-1).unsqueeze(1)  # (B,1,128)
            logits_t, h_pol = self.policy(ctx, h0=h_pol)
            centers_t, _ = self.policy.sample_centers(logits_t, tau=tau, hard=True)
            centers_t = centers_t.squeeze(1)  # (B,2)
            x_t = extract_patch(images, centers_t, p=self.p)  # (B,p*p)
            h, _ = self.res.step(h, x_t)
            z_t = torch.cat([x_t, h.detach()], dim=-1)
            logits = self.readout(z_t)
            logits_list.append(logits)
            centers_list.append(centers_t)
            prev_emb = self.policy.patch_proj(x_t)
        logits_T = torch.stack(logits_list, dim=1)  # (B,T,C)
        centers = torch.stack(centers_list, dim=1)   # (B,T,2)
        return logits_T, centers


class RasterESN(nn.Module):
    """Baseline: ESN with raster scan; dual-stream readout [x_t; h_t]."""
    def __init__(self, N: int = 1000, num_classes: int = 10, T: int = 784, device: str = 'cpu'):
        super().__init__()
        self.N, self.T = N, T
        self.res = Reservoir(N=N, in_dim=1, device=device)
        self.readout = nn.Linear(1 + N, num_classes)

    def forward(self, images: torch.Tensor):
        B, C, H, W = images.shape
        x = images.view(B, -1)  # raster order
        h = torch.zeros(B, self.N, device=images.device)
        logits_list = []
        for t in range(self.T):
            x_t = x[:, t:t+1]
            h, _ = self.res.step(h, x_t)
            z_t = torch.cat([x_t, h.detach()], dim=-1)
            logits = self.readout(z_t)
            logits_list.append(logits)
        logits_T = torch.stack(logits_list, dim=1)
        return logits_T


# Training loops

def train_one_epoch_asrin(model: ASRIN, loader, opt, device: str = 'cpu', tau: float = 0.7) -> Tuple[float, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        opt.zero_grad(set_to_none=True)
        logits_T, _ = model(images, tau=tau)
        logits = logits_T[:, -1, :]
        loss = ce(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.policy.parameters()) + list(model.readout.parameters()), 1.0)
        opt.step()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)
    return total_loss / total_count, total_correct / total_count


def train_one_epoch_raster(model: RasterESN, loader, opt, device: str = 'cpu') -> Tuple[float, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        opt.zero_grad(set_to_none=True)
        logits_T = model(images)
        logits = logits_T[:, -1, :]
        loss = ce(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.readout.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_loss += loss.item() * images.size(0)
            total_count += images.size(0)
    return total_loss / total_count, total_correct / total_count
