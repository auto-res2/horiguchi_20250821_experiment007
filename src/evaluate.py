import os
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms

try:
    from torchmetrics.classification import MulticlassCalibrationError
    TM_ECE_AVAILABLE = True
except Exception:
    TM_ECE_AVAILABLE = False

from .train import ASRIN, RasterESN, Reservoir, extract_patch


# Plot style for publication-ready PDFs
plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (5, 3),
    'savefig.format': 'pdf',
    'pdf.fonttype': 42,  # TrueType fonts in PDF
    'ps.fonttype': 42,
})


# Metrics helpers

def ece_score(logits: torch.Tensor, targets: torch.Tensor, num_bins: int = 15) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    conf, preds = probs.max(dim=-1)
    acc = preds.eq(targets).float()
    if TM_ECE_AVAILABLE:
        ece = MulticlassCalibrationError(num_classes=logits.shape[-1], n_bins=num_bins, norm='l1').to(logits.device)
        return ece(probs, targets)
    # fallback ECE
    bins = torch.linspace(0, 1, num_bins+1, device=logits.device)
    ece = torch.tensor(0.0, device=logits.device)
    for i in range(num_bins):
        mask = (conf > bins[i]) & (conf <= bins[i+1])
        if mask.sum() > 0:
            e = (conf[mask].mean() - acc[mask].mean()).abs()
            ece += (mask.float().mean() * e)
    return ece


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


# Evaluation loops

def evaluate_asrin(model: ASRIN, loader, device: str = 'cpu', collect_curves: bool = False) -> Tuple[float, float, float, np.ndarray, float | None]:
    model.eval()
    total_correct, total_count, total_ece = 0, 0, 0.0
    ce = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    early_correct_steps: List[int] = []
    all_targets, all_preds = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits_T, _ = model(images, tau=0.5)
            logits = logits_T[:, -1, :]
            loss = ce(logits, targets).item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += images.size(0)
            total_loss += loss
            total_ece += ece_score(logits, targets, num_bins=15).item() * images.size(0)
            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())
            if collect_curves:
                correct_mask = (logits_T.argmax(dim=-1) == targets.unsqueeze(1))
                for i in range(images.size(0)):
                    t90 = model.T
                    for t in range(model.T):
                        if correct_mask[i, t] and correct_mask[i, t:].all():
                            t90 = t + 1
                            break
                    early_correct_steps.append(t90)
    acc = total_correct / total_count
    ece = total_ece / total_count
    cm = confusion_matrix(torch.cat(all_targets).numpy(), torch.cat(all_preds).numpy())
    return total_loss / total_count, acc, ece, cm, (float(np.mean(early_correct_steps)) if collect_curves and len(early_correct_steps) > 0 else None)


def evaluate_raster(model: RasterESN, loader, device: str = 'cpu', collect_curves: bool = False) -> Tuple[float, float, float, np.ndarray, float | None]:
    model.eval()
    total_correct, total_count, total_ece = 0, 0, 0.0
    ce = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    early_correct_steps: List[int] = []
    all_targets, all_preds = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits_T = model(images)
            logits = logits_T[:, -1, :]
            loss = ce(logits, targets).item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += images.size(0)
            total_loss += loss
            total_ece += ece_score(logits, targets, num_bins=15).item() * images.size(0)
            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())
            if collect_curves:
                correct_mask = (logits_T.argmax(dim=-1) == targets.unsqueeze(1))
                for i in range(images.size(0)):
                    t90 = model.T
                    for t in range(model.T):
                        if correct_mask[i, t] and correct_mask[i, t:].all():
                            t90 = t + 1
                            break
                    early_correct_steps.append(t90)
    acc = total_correct / total_count
    ece = total_ece / total_count
    cm = confusion_matrix(torch.cat(all_targets).numpy(), torch.cat(all_preds).numpy())
    return total_loss / total_count, acc, ece, cm, (float(np.mean(early_correct_steps)) if collect_curves and len(early_correct_steps) > 0 else None)


# Curves and compute proxy

def accuracy_vs_t_asrin(model: ASRIN, loader, device: str = 'cpu') -> np.ndarray:
    model.eval()
    correct_counts = None
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits_T, _ = model(images, tau=0.5)
            preds_T = logits_T.argmax(dim=-1)  # (B,T)
            if correct_counts is None:
                correct_counts = torch.zeros(model.T, device=device)
            for t in range(model.T):
                correct_counts[t] += (preds_T[:, t] == targets).sum()
            total += images.size(0)
    return (correct_counts / total).cpu().numpy()


def flops_proxy_reservoir(reservoir: Reservoir, T: int, input_dim: int) -> int:
    nnz = reservoir.nnz()
    N = reservoir.N
    # per step: 2*nnz for matmul + N*input_dim for W_in + N for tanh
    per_step = 2*nnz + N*input_dim + N
    return int(per_step * T)


# Perturbations and adversarial

def add_noise(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    return torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)


def rotate_batch(x: torch.Tensor, deg: float = 15) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    return TF.rotate(x, deg, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)


def occlude(x: torch.Tensor, area: float = 0.2) -> torch.Tensor:
    B, C, H, W = x.shape
    rect_h = max(1, int(math.sqrt(area) * H))
    rect_w = max(1, int(math.sqrt(area) * W))
    out = x.clone()
    for i in range(B):
        top = np.random.randint(0, max(1, H - rect_h + 1))
        left = np.random.randint(0, max(1, W - rect_w + 1))
        out[i, :, top:top+rect_h, left:left+rect_w] = 0.0
    return out


def fgsm_asrin(model: ASRIN, images: torch.Tensor, targets: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    # Use training mode to allow cuDNN RNN backward, then restore original state
    was_training = model.training
    try:
        model.train()
        images_adv = images.clone().detach().requires_grad_(True)
        logits_T, _ = model(images_adv, tau=0.5)
        logits = logits_T[:, -1, :]
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        x_adv = torch.clamp(images_adv + eps * images_adv.grad.sign(), 0.0, 1.0).detach()
    finally:
        model.train(was_training)
    return x_adv


def saccade_stability(model: ASRIN, images: torch.Tensor, images_pert: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        _, centers = model(images, tau=0.5)
        _, centers_p = model(images_pert, tau=0.5)
        drift = (centers - centers_p).pow(2).sum(dim=-1).sqrt().mean().item()
    return float(drift)


# Dynamics and kernel-limit
@torch.no_grad()
def rollout_patches(model: ASRIN, images: torch.Tensor) -> torch.Tensor:
    model.eval()
    B = images.size(0)
    device = images.device
    h = torch.zeros(B, model.N, device=device)
    prev_patch = torch.zeros(B, model.p*model.p, device=device)
    prev_emb = model.policy.patch_proj(prev_patch)
    patches = []
    h_pol = None
    for _t in range(model.T):
        res_ctx = F.linear(h, model.res_proj)
        ctx = torch.cat([prev_emb, res_ctx], dim=-1).unsqueeze(1)
        logits_t, h_pol = model.policy(ctx, h0=h_pol)
        centers_t, _ = model.policy.sample_centers(logits_t, tau=0.5, hard=True)
        centers_t = centers_t.squeeze(1)
        x_t = extract_patch(images, centers_t, p=model.p)
        patches.append(x_t)
        h, _ = model.res.step(h, x_t)
        prev_emb = model.policy.patch_proj(x_t)
    patches = torch.stack(patches, dim=1)  # (B,T,D)
    return patches


@torch.no_grad()
def lyapunov_benettin(reservoir: Reservoir, seq_x: torch.Tensor, T: int = 200, delta: float = 1e-7) -> float:
    # seq_x: (T, in_dim)
    device = seq_x.device
    B = 1
    h = torch.zeros(B, reservoir.N, device=device)
    h_pert = h + delta * F.normalize(torch.randn_like(h), dim=-1)
    sum_log = 0.0
    for t in range(min(T, seq_x.size(0))):
        x_t = seq_x[t:t+1, :].expand(B, -1)
        h, _ = reservoir.step(h, x_t)
        h_pert, _ = reservoir.step(h_pert, x_t)
        d = h_pert - h
        r = d.norm(dim=-1, keepdim=True) + 1e-12
        h_pert = h + (d / r) * delta
        sum_log += torch.log((r / delta)).mean().item()
    return float(sum_log / max(1, min(T, seq_x.size(0))))


@torch.no_grad()
def memory_capacity(reservoir: Reservoir, T: int = 1000, K: int = 30, lamb: float = 1e-6, device: str = 'cpu') -> float:
    # Drive with scalar u_t in [-1,1]
    u = (torch.rand(T, 1, device=device) * 2.0 - 1.0)
    h = torch.zeros(1, reservoir.N, device=device)
    H = []
    # Map scalar to reservoir.in_dim via random fixed matrix if needed
    W_tmp = torch.randn(reservoir.in_dim, 1, device=device) / math.sqrt(reservoir.in_dim)
    for t in range(T):
        u_t = u[t:t+1]  # (1,1)
        x = u_t @ W_tmp.T  # (1,in_dim)
        h, _ = reservoir.step(h, x)
        H.append(h.clone())
    H = torch.cat(H, dim=0)  # (T,N)
    MC = 0.0
    for k in range(1, K+1):
        y = u[k: T]
        X = H[:T-k]
        XtX = X.T @ X
        W = torch.linalg.solve(XtX + lamb * torch.eye(XtX.shape[0], device=device), X.T @ y)
        yhat = X @ W
        R2 = 1.0 - ((y - yhat).pow(2).sum() / (y.pow(2).sum() + 1e-12)).item()
        MC += max(0.0, float(R2))
    return float(MC)


# Simplified recurrent NNGP kernel recursion for sequences (arcsin approximation)

def arcsin_kernel(c12: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    c = torch.clamp(c12 / torch.sqrt((1 + q1) * (1 + q2) + 1e-8), -0.999999, 0.999999)
    return (2.0 / math.pi) * torch.asin(c)


@torch.no_grad()
def kasrin_kernel(seqs: List[torch.Tensor], leak: float = 0.3, sig_in: float = 1.0) -> np.ndarray:
    # seqs: list of T x D tensors (on same device)
    n = len(seqs)
    T = seqs[0].shape[0]
    D = seqs[0].shape[1]
    device = seqs[0].device
    K = torch.zeros(n, n, device=device)
    Q = torch.zeros(n, device=device)
    for _t in range(T):
        X_t = torch.stack([seqs[i][_t] for i in range(n)], dim=0)  # (n,D)
        G_in = (X_t @ X_t.T) * (sig_in**2) / D
        C = (1 - leak)**2 * K + (leak**2) * G_in
        q_new = (1 - leak)**2 * Q + (leak**2) * torch.diag(G_in)
        K = arcsin_kernel(C, q_new.unsqueeze(0), q_new.unsqueeze(1))
        Q = torch.diag(K).clone()
    return K.detach().cpu().numpy()


# Plotting helpers (save to output_dir)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_and_save_line(xs: List[float], ys_dict: Dict[str, np.ndarray | List[float]], xlabel: str, ylabel: str, title: str, output_dir: str, filename: str):
    _ensure_dir(output_dir)
    plt.figure(figsize=(5, 3))
    for k, v in ys_dict.items():
        plt.plot(xs, v, label=k)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', format='pdf')
    plt.close()


def plot_and_save_bar(labels: List[str], values: List[float], ylabel: str, title: str, output_dir: str, filename: str):
    _ensure_dir(output_dir)
    plt.figure(figsize=(5, 3))
    sns.barplot(x=labels, y=values, palette='deep')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', format='pdf')
    plt.close()


def plot_and_save_confusion(cm: np.ndarray, class_names: List[str], title: str, output_dir: str, filename: str):
    _ensure_dir(output_dir)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', format='pdf')
    plt.close()
