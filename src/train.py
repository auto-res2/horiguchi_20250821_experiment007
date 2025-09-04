import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# Reproducibility
# =============================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================
# Models: Saccadic encoder, Reservoir, Baselines
# =============================

class PatchSampler(nn.Module):
    def __init__(self, img_h=28, img_w=28, patch_size=14):
        super().__init__()
        self.H, self.W = img_h, img_w
        self.P = patch_size

    def forward(self, images, centers):
        # images: (B,1,H,W); centers: (B,2) in [-1,1], (y,x)
        B = images.size(0)
        P = self.P
        lin = torch.linspace(-1, 1, P, device=images.device)
        gy, gx = torch.meshgrid(lin, lin, indexing='ij')
        grid = torch.stack([gy, gx], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        scale_y = self.P / self.H
        scale_x = self.P / self.W
        grid[..., 0] = grid[..., 0] * scale_y / 2 + centers[:, 0].view(B, 1, 1)
        grid[..., 1] = grid[..., 1] * scale_x / 2 + centers[:, 1].view(B, 1, 1)
        patch = F.grid_sample(images, grid, mode='bilinear', align_corners=True)
        return patch  # (B,1,P,P)


class SaccadePolicy(nn.Module):
    def __init__(self, P=14, grid_sz=8, emb_dim=256, hidden=64, n_classes=10):
        super().__init__()
        self.P = P
        self.grid_sz = grid_sz
        self.embed = nn.Sequential(nn.Flatten(), nn.Linear(P * P, emb_dim), nn.ReLU())
        self.gru = nn.GRU(emb_dim + n_classes, hidden, batch_first=True)
        self.head = nn.Linear(hidden, grid_sz * grid_sz)
        coords_1d = torch.linspace(-1, 1, grid_sz)
        yy, xx = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
        centers = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
        self.register_buffer('centers', centers)  # (G,2)
        self.n_classes = n_classes

    def forward(self, patch_t, logits_prev, h_gru=None, tau=0.5, hard=False):
        B = patch_t.size(0)
        z = self.embed(patch_t)  # (B,emb)
        inp = torch.cat([z, logits_prev], dim=1).unsqueeze(1)
        out, h_gru = self.gru(inp, h_gru)
        logits = self.head(out.squeeze(1))  # (B,G)
        g = F.gumbel_softmax(logits, tau=tau, hard=hard)
        centers = torch.matmul(g, self.centers)  # (B,2)
        return centers, logits, h_gru


class FixedRasterPolicy:
    def __init__(self, P=14, T=30):
        self.P = P
        self.T = T
        coords = []
        gy = torch.linspace(-1, 1, 5)
        gx = torch.linspace(-1, 1, 6)
        for y in gy:
            for x in gx:
                coords.append(torch.tensor([y, x]))
        self.centers = torch.stack(coords, dim=0)[:T]

    def forward(self, B, device):
        return self.centers.to(device).unsqueeze(1).repeat(1, B, 1)  # (T,B,2)


class SparseESN(nn.Module):
    def __init__(self, N=1000, input_dim=196, density=0.02, leak=0.3, spectral_radius=1.0, device='cpu'):
        super().__init__()
        self.N = N
        self.input_dim = input_dim
        self.leak = leak
        self.Win = nn.Linear(input_dim, N, bias=False)
        # Sparse W
        idx = torch.nonzero(torch.rand(N, N, device=device) < density)
        vals = torch.randn(idx.size(0), device=device) / math.sqrt(max(1e-6, density * N))
        W = torch.sparse_coo_tensor(idx.t(), vals, (N, N), device=device)
        # Scale spectral radius via power iteration
        with torch.no_grad():
            x = torch.randn(N, device=device)
            for _ in range(50):
                x = torch.sparse.mm(W, x.unsqueeze(1)).squeeze(1)
                x = x / (x.norm() + 1e-6)
            lam = (x @ torch.sparse.mm(W, x.unsqueeze(1)).squeeze(1))
            scale = spectral_radius / (lam.abs() + 1e-6)
            W = torch.sparse_coo_tensor(idx.t(), vals * scale, (N, N), device=device)
        self.register_buffer('W_idx', W._indices())
        self.register_buffer('W_val', W._values())
        self.register_buffer('W_size', torch.tensor([N, N], device=device))
        self.register_buffer('g', torch.tensor(1.0, device=device))  # astrocyte gain
        self.register_buffer('m_slope', torch.tensor(0.5, device=device))
        self.momentum = 0.9
        self.eta = 1e-3

    def sparse_mm(self, h):
        W = torch.sparse_coo_tensor(self.W_idx, self.W_val, tuple(self.W_size.tolist()), device=h.device)
        return torch.sparse.mm(W, h)

    def forward(self, x_seq, h0=None, enable_ecsr=True):
        # x_seq: (T,B,input_dim)
        T, B, _ = x_seq.size()
        device = x_seq.device
        h = torch.zeros(B, self.N, device=device) if h0 is None else h0
        hs = []
        g = self.g
        for _t in range(T):
            u = self.Win(x_seq[_t])  # (B,N)
            pre = g * (self.sparse_mm(h.t()).t()) + u  # (B,N)
            h_tilde = torch.tanh(pre)
            h = (1 - self.leak) * h + self.leak * h_tilde
            hs.append(h)
            if enable_ecsr:
                slope = (1 - h_tilde.pow(2)).mean().detach()
                self.m_slope = self.momentum * self.m_slope + (1 - self.momentum) * slope
                L_t = self.leak * g * 1.0 * self.m_slope + (1 - self.leak)
                g = torch.clamp(g * torch.exp(self.eta * (1 - L_t)), 0.5, 2.0)
        self.g = g.detach()
        return torch.stack(hs, dim=0)  # (T,B,N)


class ASRIN(nn.Module):
    def __init__(self, N=1000, P=14, T=30, n_classes=10, grid_sz=8, device='cpu', enable_ecsr=True):
        super().__init__()
        self.P = P
        self.T = T
        self.n_classes = n_classes
        self.policy = SaccadePolicy(P=P, grid_sz=grid_sz, n_classes=n_classes)
        self.sampler = PatchSampler(28, 28, P)
        self.esn = SparseESN(N=N, input_dim=P * P, device=device)
        self.enable_ecsr = enable_ecsr
        self.readout = nn.Linear(N + P * P, n_classes)

    def forward(self, images, tau=0.5, hard=False, return_traj=False):
        B = images.size(0)
        device = images.device
        logits_prev = torch.zeros(B, self.n_classes, device=device)
        h_gru = None
        x_seq = []
        patches = []
        centers_list = []
        center = torch.zeros(B, 2, device=device)
        patch = self.sampler(images, center)
        for _t in range(self.T):
            p = patch.view(B, -1)
            p = (p - p.mean(dim=1, keepdim=True)) / (p.std(dim=1, keepdim=True) + 1e-6)
            patches.append(p.view(B, 1, self.P, self.P))
            x_seq.append(p)
            center, _, h_gru = self.policy(patches[-1], logits_prev.detach(), h_gru, tau=tau, hard=hard)
            centers_list.append(center)
            patch = self.sampler(images, center)
        x_seq = torch.stack(x_seq, dim=0)
        hs = self.esn(x_seq, enable_ecsr=self.enable_ecsr)
        x_last = x_seq[-1]
        h_last = hs[-1]
        logits = self.readout(torch.cat([x_last, h_last], dim=1))
        if return_traj:
            centers = torch.stack(centers_list, dim=0)
            return logits, centers, hs, x_seq
        return logits


class ASRIN_NoDSE(nn.Module):
    def __init__(self, N=1000, P=14, T=30, n_classes=10, device='cpu', enable_ecsr=True):
        super().__init__()
        self.P = P
        self.T = T
        self.sampler = PatchSampler(28, 28, P)
        self.policy = FixedRasterPolicy(P=P, T=T)
        self.esn = SparseESN(N=N, input_dim=P * P, device=device)
        self.readout = nn.Linear(N + P * P, n_classes)
        self.enable_ecsr = enable_ecsr

    def forward(self, images):
        B = images.size(0)
        centers = self.policy.forward(B, images.device)
        patches = []
        for _t in range(self.T):
            patch = self.sampler(images, centers[_t])
            p = patch.view(B, -1)
            p = (p - p.mean(dim=1, keepdim=True)) / (p.std(dim=1, keepdim=True) + 1e-6)
            patches.append(p)
        x_seq = torch.stack(patches, dim=0)
        hs = self.esn(x_seq, enable_ecsr=self.enable_ecsr)
        logits = self.readout(torch.cat([x_seq[-1], hs[-1]], dim=1))
        return logits


class RasterESN(nn.Module):
    def __init__(self, N=1000, T=784, n_classes=10, spectral_radius=0.9, density=0.02, device='cpu'):
        super().__init__()
        self.T = T
        self.N = N
        self.n_classes = n_classes
        self.esn = SparseESN(N=N, input_dim=1, density=density, leak=0.3, spectral_radius=spectral_radius, device=device)
        self.readout = nn.Linear(N + 1, n_classes)

    def forward(self, images):
        B = images.size(0)
        x = images.view(B, -1).t().unsqueeze(-1)  # (T,B,1)
        hs = self.esn(x, enable_ecsr=False)
        x_last = x[-1].squeeze(-1)
        h_last = hs[-1]
        return self.readout(torch.cat([x_last, h_last], dim=1))


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, n_classes)

    def forward(self, x):
        z = self.net(x)
        return self.fc(z.view(z.size(0), -1))


class RandomFeatureMLP(nn.Module):
    def __init__(self, hidden=512, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden, bias=False)
        for p in self.fc1.parameters():
            p.requires_grad = False
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        z = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(z)


# =============================
# Training helpers
# =============================

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n

    @property
    def avg(self):
        return self.sum / max(1, self.cnt)


def train_one_epoch(model: nn.Module, loader, opt, device, scheduler=None, tau=0.7):
    model.train()
    ce = nn.CrossEntropyLoss()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad()
        if isinstance(model, ASRIN):
            logits = model(images, tau=tau)
        else:
            logits = model(images)
        loss = ce(logits, labels)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        pred = logits.argmax(1)
        acc = (pred == labels).float().mean().item()
        acc_meter.update(acc, labels.size(0))
        loss_meter.update(loss.item(), labels.size(0))
    return acc_meter.avg, loss_meter.avg
