import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.kernel_ridge import KernelRidge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set high-quality PDF defaults
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['savefig.dpi'] = 300

from .train import ASRIN

# =============================
# Metrics and core evaluation
# =============================

class ECE(nn.Module):
    def __init__(self, n_bins=15):
        super().__init__()
        self.n_bins = n_bins

    @torch.no_grad()
    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        acc = pred.eq(labels).float()
        ece = torch.zeros(1, device=logits.device)
        bins = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        for i in range(self.n_bins):
            m = (conf > bins[i]) & (conf <= bins[i + 1])
            if m.any():
                ece += (m.float().mean()) * (conf[m].mean() - acc[m].mean()).abs()
        return ece.item()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, return_logits_labels=False):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')
    total, correct, loss_sum = 0, 0, 0.0
    logits_all, labels_all = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if isinstance(model, ASRIN):
            logits = model(images, tau=0.3)
        else:
            logits = model(images)
        loss_sum += ce(logits, labels).item()
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())
    acc = correct / total
    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    ece = ECE(n_bins=15)(logits_all, labels_all)
    if return_logits_labels:
        return acc, loss_sum / total, ece, logits_all.numpy(), labels_all.numpy()
    return acc, loss_sum / total, ece


@torch.no_grad()
def evaluate_early_exit(asrin_model: ASRIN, loader, device):
    asrin_model.eval()
    T = asrin_model.T
    correct_t = np.zeros(T, dtype=np.int64)
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, centers, hs, x_seq = asrin_model(images, tau=0.3, return_traj=True)
        T_, B, _ = hs.size()
        for t in range(T_):
            logits_t = asrin_model.readout(torch.cat([x_seq[t], hs[t]], dim=1))
            pred = logits_t.argmax(1)
            correct_t[t] += (pred == labels).sum().item()
        total += labels.size(0)
    acc_t = correct_t / max(1, total)
    return acc_t


# =============================
# Corruptions and adversarial
# =============================

@torch.no_grad()
def add_gaussian_noise(x, sigma):
    return (x + sigma * torch.randn_like(x)).clamp(0, 1)


@torch.no_grad()
def rotate_images(x, degrees=15):
    B = x.size(0)
    theta = math.radians(degrees)
    Tm = torch.tensor([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0]
    ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(B, 1, 1)
    grid = F.affine_grid(Tm, x.size(), align_corners=True)
    return F.grid_sample(x, grid, align_corners=True)


@torch.no_grad()
def occlude(x, frac=0.2, val=0.5):
    B, _, H, W = x.size()
    side = max(1, int(np.sqrt(frac) * H))
    out = x.clone()
    for i in range(B):
        y = np.random.randint(0, max(1, H - side))
        z = np.random.randint(0, max(1, W - side))
        out[i, :, y:y + side, z:z + side] = val
    return out


def fgsm_attack(model, images, labels, eps=0.1):
    prev_mode = model.training
    try:
        model.train()  # ensure cudnn RNN backward allowed
        images = images.clone().detach().requires_grad_(True)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        if isinstance(model, ASRIN):
            logits = model(images, tau=0.3)
        else:
            logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        adv = images + eps * images.grad.sign()
        return adv.clamp(0, 1).detach()
    finally:
        model.train(prev_mode)


def evaluate_corruptions(model, loader, device, eps=0.1):
    model.eval()

    def eval_on(data_iter):
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in data_iter:
                images, labels = images.to(device), labels.to(device)
                logits = model(images, tau=0.3) if isinstance(model, ASRIN) else model(images)
                pred = logits.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return correct / total

    # Clean
    acc_clean = eval_on(loader)

    # Corruptions
    def loader_map(fn):
        for images, labels in loader:
            yield fn(images.to(device)), labels.to(device)

    acc_noise_01 = eval_on(loader_map(lambda x: add_gaussian_noise(x, 0.1)))
    acc_noise_03 = eval_on(loader_map(lambda x: add_gaussian_noise(x, 0.3)))
    acc_rot = eval_on(loader_map(lambda x: rotate_images(x, 15)))
    acc_occ = eval_on(loader_map(lambda x: occlude(x, 0.2)))

    # Adversarial
    total, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.enable_grad():
            adv = fgsm_attack(model, images, labels, eps=eps)
        with torch.no_grad():
            logits = model(adv, tau=0.3) if isinstance(model, ASRIN) else model(adv)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc_adv = correct / total

    deltas = {
        'clean': acc_clean,
        'noise_0.1': acc_clean - acc_noise_01,
        'noise_0.3': acc_clean - acc_noise_03,
        'rot_15': acc_clean - acc_rot,
        'occ_20': acc_clean - acc_occ,
        'fgsm_0.1': acc_clean - acc_adv,
    }
    return deltas


# =============================
# Dynamics: LLE and Memory Capacity
# =============================

@torch.no_grad()
def estimate_lle(asrin_model: ASRIN, images):
    asrin_model.eval()
    logits, centers, hs, x_seq = asrin_model(images, tau=0.3, return_traj=True)
    T, B, N = hs.size()
    v = F.normalize(torch.randn(B, N, device=images.device), dim=1)
    lam_sum = torch.zeros(B, device=images.device)
    g = asrin_model.esn.g
    W = torch.sparse_coo_tensor(asrin_model.esn.W_idx, asrin_model.esn.W_val, (N, N), device=images.device)
    for t in range(T):
        s = 1 - hs[t].pow(2)  # (B,N)
        v1 = v * s
        v2 = torch.stack([torch.sparse.mm(W, v1[i].unsqueeze(1)).squeeze(1) for i in range(B)], dim=0)
        Jv = (1 - asrin_model.esn.leak) * v + asrin_model.esn.leak * g * v2
        norm = Jv.norm(dim=1) + 1e-12
        lam_sum += torch.log(norm)
        v = Jv / norm.unsqueeze(1)
    lle = (lam_sum / T).mean().item()
    return lle


@torch.no_grad()
def memory_capacity(asrin_model: ASRIN, T=400, K=20, sigma=1.0):
    N = asrin_model.esn.N
    B = 1
    device = next(asrin_model.parameters()).device
    u = torch.randn(T, B, 1, device=device) * sigma
    x_seq = u.repeat(1, 1, asrin_model.esn.input_dim)
    hs = asrin_model.esn(x_seq, enable_ecsr=True)
    H = hs.squeeze(1)  # (T,N)
    H_np = H.cpu().numpy()
    MC = 0.0
    for k in range(1, K + 1):
        y = u[k:, :, 0].cpu().numpy()
        X = H_np[:-k]
        lam = 1e-3
        w = np.linalg.solve(X.T @ X + lam * np.eye(N), X.T @ y)
        yhat = X @ w
        num = ((y - yhat) ** 2).sum()
        den = ((y - y.mean()) ** 2).sum() + 1e-12
        r2 = 1 - num / den
        MC += max(r2, 0)
    return MC


# =============================
# Kernel construction and alignment
# =============================

def arcsin_tanh_kernel(S, Qi, Qj):
    denom = torch.sqrt((1 + 2 * Qi).clamp_min(1e-6) * (1 + 2 * Qj).clamp_min(1e-6))
    X = (S / denom).clamp(-1 + 1e-6, 1 - 1e-6)
    return (2 / np.pi) * torch.asin(X)


@torch.no_grad()
def build_sequences(policy, sampler, loader, T=30, P=14, tau=0.05, device='cpu', n_classes=10):
    policy.eval()
    X_list = []
    Y_list = []
    for images, labels in loader:
        images = images.to(device)
        B = images.size(0)
        logits_prev = torch.zeros(B, n_classes, device=device)
        h_gru = None
        center = torch.zeros(B, 2, device=device)
        patches = []
        patch = sampler(images, center)
        for _t in range(T):
            p = patch.view(B, -1)
            p = (p - p.mean(dim=1, keepdim=True)) / (p.std(dim=1, keepdim=True) + 1e-6)
            patches.append(p)
            center, _, h_gru = policy(p.view(B, 1, P, P), logits_prev, h_gru, tau=tau, hard=False)
            patch = sampler(images, center)
        X_list.append(torch.stack(patches, dim=1).cpu())  # (B,T,d)
        Y_list.append(labels)
    X = torch.cat(X_list, dim=0).numpy()
    Y = torch.cat(Y_list, dim=0).numpy()
    return X, Y


@torch.no_grad()
def build_kernel(X, alpha=0.3, g=1.0, rho=1.0, sigma_in=1.0, lam_dual=1.0):
    X_t = torch.from_numpy(X).float()
    n, T, d = X_t.shape
    Cx = []
    for t in range(T):
        Xt = X_t[:, t, :]
        Cx_t = (Xt @ Xt.t()) / d
        Cx.append(Cx_t)
    Cx_sum = torch.stack(Cx, dim=0).sum(0)
    Ch = torch.zeros(n, n)
    qi = torch.zeros(n)
    for t in range(T):
        S = (g ** 2) * (rho ** 2) * Ch + (sigma_in ** 2) * Cx[t]
        qi = torch.diag(S)
        Qi = qi.view(-1, 1).expand_as(S)
        Qj = qi.view(1, -1).expand_as(S)
        Phi = arcsin_tanh_kernel(S, Qi, Qj)
        Ch = (1 - alpha) ** 2 * Ch + (alpha ** 2) * Phi
    K = Ch + lam_dual * Cx_sum
    return K.numpy()


def cka(K1, K2):
    def hsic(K):
        return np.sum(K * K)
    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1c = H @ K1 @ H
    K2c = H @ K2 @ H
    return (np.sum(K1c * K2c) / np.sqrt(hsic(K1c) * hsic(K2c) + 1e-12))


# =============================
# Plotting utilities (save PDFs)
# =============================

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_training_losses(history_dict, images_dir, fname_prefix='training_loss'):
    _ensure_dir(images_dir)
    plt.figure(figsize=(4, 3))
    for name, vals in history_dict.items():
        plt.plot(vals, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{fname_prefix}.pdf'), bbox_inches='tight')
    plt.close()


def plot_accuracy_bars(results_dict, images_dir, fname='accuracy_baselines'):
    _ensure_dir(images_dir)
    names = list(results_dict.keys())
    vals = [results_dict[k] for k in names]
    plt.figure(figsize=(4, 3))
    sns.barplot(x=names, y=vals, color='steelblue')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_pdf(y_true, y_pred, images_dir, fname='confusion_matrix_asrin'):
    _ensure_dir(images_dir)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, cmap='Blues', square=True, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()


def plot_robustness_bars(deltas, images_dir, fname='robustness_drop_asrin'):
    _ensure_dir(images_dir)
    keys = ['noise_0.1', 'noise_0.3', 'rot_15', 'occ_20', 'fgsm_0.1']
    vals = [deltas[k] for k in keys]
    plt.figure(figsize=(4, 3))
    sns.barplot(x=keys, y=vals, color='salmon')
    plt.ylabel('Δ Accuracy (clean - corrupted)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()


def plot_early_exit(acc_t, images_dir, fname='early_exit_accuracy_asrin'):
    _ensure_dir(images_dir)
    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(1, len(acc_t) + 1), acc_t, marker='o')
    plt.xlabel('Time step t')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()


def plot_lle_hist(values, images_dir, fname='lle_asrin'):
    _ensure_dir(images_dir)
    plt.figure(figsize=(4, 3))
    sns.histplot(values, bins=20, color='purple')
    plt.xlabel('Estimated LLE')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()


def plot_kernel_accuracy_vs_N(N_list, acc_list, acc_krr, images_dir, fname='kernel_accuracy_pair1'):
    _ensure_dir(images_dir)
    plt.figure(figsize=(4, 3))
    plt.plot(N_list, acc_list, marker='o', label='Finite reservoir')
    plt.axhline(acc_krr, color='k', linestyle='--', label='K_ASRIN (KRR)')
    plt.xlabel('Reservoir size N')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()


def plot_kernel_alignment_vs_N(N_list, cka_list, images_dir, fname='kernel_alignment_pair2'):
    _ensure_dir(images_dir)
    plt.figure(figsize=(4, 3))
    plt.plot(N_list, cka_list, marker='o', color='green')
    plt.xlabel('Reservoir size N')
    plt.ylabel('CKA alignment')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()
