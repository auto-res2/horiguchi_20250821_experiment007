import os
import time
import argparse
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .train import ASRIN, RasterESN, train_one_epoch_asrin, train_one_epoch_raster, set_seeds, device_selection
from .evaluate import (
    evaluate_asrin,
    evaluate_raster,
    accuracy_vs_t_asrin,
    plot_and_save_line,
    plot_and_save_bar,
    plot_and_save_confusion,
    flops_proxy_reservoir,
    add_noise,
    rotate_batch,
    occlude,
    fgsm_asrin,
    saccade_stability,
    rollout_patches,
    lyapunov_benettin,
    memory_capacity,
    kasrin_kernel,
)
from .preprocess import get_mnist_loaders, get_small_mnist_loaders


DEFAULT_CONFIG = {
    'seed': 7,
    'output_dir': '.research/iteration3/images',
    'epochs': 2,
    'asrin': {'N': 128, 'T': 8, 'p': 6, 'grid_size': 8, 'lr': 1e-3},
    'raster': {'N': 128, 'T': 784, 'lr': 1e-3},
    'subsets': {'train': 1024, 'val': 256, 'test': 512},
}


def load_config(path: str | None) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if path is not None and os.path.isfile(path):
        with open(path, 'r') as f:
            user_cfg = yaml.safe_load(f)
        # shallow merge
        for k, v in user_cfg.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def experiment_1_mnist(cfg: Dict[str, Any], device: str) -> Dict[str, Any]:
    print('=== Experiment 1 (MNIST, clean) ===')
    seed = cfg['seed']
    set_seeds(seed)
    out_dir = cfg['output_dir']
    ensure_output_dir(out_dir)

    # Data
    train_loader, val_loader, test_loader = get_mnist_loaders(
        train_subset=cfg['subsets']['train'], val_subset=cfg['subsets']['val'], test_subset=cfg['subsets']['test'],
        batch_size_train=128, batch_size_eval=256)

    # ASRIN model
    asrin = ASRIN(N=cfg['asrin']['N'], T=cfg['asrin']['T'], p=cfg['asrin']['p'], num_classes=10, grid_size=cfg['asrin']['grid_size'], device=device).to(device)
    opt = torch.optim.Adam(list(asrin.policy.parameters()) + list(asrin.readout.parameters()), lr=cfg['asrin']['lr'])

    tr_losses, val_accs = [], []
    best_val, best_state = 0.0, None
    for e in range(cfg['epochs']):
        tau = max(0.5, 1.0 - 0.5 * (e / max(1, cfg['epochs']-1)))
        tl, ta = train_one_epoch_asrin(asrin, train_loader, opt, device=device, tau=tau)
        vl, va, ve, vcm, t90 = evaluate_asrin(asrin, val_loader, device=device, collect_curves=True)
        tr_losses.append(tl); val_accs.append(va)
        print(f"[ASRIN] Epoch {e+1}/{cfg['epochs']} - train_loss={tl:.4f} train_acc={ta:.4f} val_acc={va:.4f} val_ece={ve:.4f} t90={(t90 if t90 is not None else float('nan')):.2f}")
        if va > best_val:
            best_val = va
            best_state = asrin.state_dict()
    if best_state is not None:
        asrin.load_state_dict(best_state)

    # Test ASRIN
    tl, acc_asrin, ece_asrin, cm_asrin, t90_asrin = evaluate_asrin(asrin, test_loader, device=device, collect_curves=True)
    print(f"[ASRIN] Test acc={acc_asrin:.4f} ECE={ece_asrin:.4f} t90={(t90_asrin if t90_asrin is not None else float('nan')):.2f}")

    # Raster baseline
    raster_T = cfg['raster']['T']
    raster = RasterESN(N=cfg['raster']['N'], num_classes=10, T=raster_T, device=device).to(device)
    opt_r = torch.optim.Adam(raster.readout.parameters(), lr=cfg['raster']['lr'])
    tr_losses_r, val_accs_r = [], []
    for e in range(max(1, cfg['epochs']-1)):
        tlr, tar = train_one_epoch_raster(raster, train_loader, opt_r, device=device)
        vlr, var, ver, vcmr, t90r = evaluate_raster(raster, val_loader, device=device, collect_curves=True)
        tr_losses_r.append(tlr); val_accs_r.append(var)
        print(f"[Raster] Epoch {e+1}/{max(1, cfg['epochs']-1)} - train_loss={tlr:.4f} train_acc={tar:.4f} val_acc={var:.4f} val_ece={ver:.4f} t90={(t90r if t90r is not None else float('nan')):.2f}")
    tlr, acc_raster, ece_raster, cm_raster, t90_raster = evaluate_raster(raster, test_loader, device=device, collect_curves=True)
    print(f"[Raster] Test acc={acc_raster:.4f} ECE={ece_raster:.4f} t90={(t90_raster if t90_raster is not None else float('nan')):.2f}")

    # Plots: training loss and val accuracy curves
    plot_and_save_line(list(range(1, len(tr_losses)+1)), {'asrin': tr_losses}, 'epoch', 'train loss', 'Training Loss (ASRIN)', out_dir, 'training_loss_asrin.pdf')
    plot_and_save_line(list(range(1, len(val_accs)+1)), {'asrin': val_accs}, 'epoch', 'val accuracy', 'Validation Accuracy (ASRIN)', out_dir, 'accuracy_asrin.pdf')

    # Early decision accuracy curve on test
    acc_vs_t = accuracy_vs_t_asrin(asrin, test_loader, device=device)
    plot_and_save_line(list(range(1, len(acc_vs_t)+1)), {'asrin': acc_vs_t.tolist()}, 'time step', 'accuracy', 'Early-decision Accuracy vs Time (ASRIN)', out_dir, 'early_decision_asrin.pdf')

    # Confusion matrix
    plot_and_save_confusion(cm_asrin, [str(i) for i in range(10)], 'ASRIN Confusion Matrix (MNIST)', out_dir, 'confusion_matrix_asrin.pdf')

    # FLOPs proxy comparison
    flops_asrin = flops_proxy_reservoir(asrin.res, T=cfg['asrin']['T'], input_dim=cfg['asrin']['p']*cfg['asrin']['p'])
    flops_raster = flops_proxy_reservoir(raster.res, T=raster_T, input_dim=1)
    print(f"[Compute] FLOPs proxy - ASRIN ~ {flops_asrin/1e6:.2f}M vs Raster ~ {flops_raster/1e6:.2f}M; reduction ~ {flops_raster/max(1, flops_asrin):.1f}x")
    plot_and_save_bar(['ASRIN','Raster'], [acc_asrin, acc_raster], 'test accuracy', 'Accuracy Comparison', out_dir, 'accuracy_asrin_vs_raster.pdf')

    # Save ASRIN model
    os.makedirs('models', exist_ok=True)
    torch.save(asrin.state_dict(), os.path.join('models', 'asrin_mnist_quick.pth'))

    return {
        'asrin': {'acc': acc_asrin, 'ece': ece_asrin, 't90': t90_asrin, 'cm': cm_asrin},
        'raster': {'acc': acc_raster, 'ece': ece_raster, 't90': t90_raster, 'cm': cm_raster},
        'flops': {'asrin': flops_asrin, 'raster': flops_raster},
        'model': asrin,
    }


def experiment_2_robustness(asrin_model: ASRIN, cfg: Dict[str, Any], device: str, test_loader) -> Dict[str, Any]:
    print('=== Experiment 2 (Robustness) ===')
    model = asrin_model
    model.eval()
    out_dir = cfg['output_dir']

    def run(eval_transform=None, adv: bool = False) -> float:
        total_correct, total_count = 0, 0
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            if adv:
                images = fgsm_asrin(model, images, targets, eps=0.1)
            elif eval_transform is not None:
                images = eval_transform(images)
            with torch.no_grad():
                logits_T, _ = model(images, tau=0.5)
                preds = logits_T[:, -1, :].argmax(dim=-1)
                total_correct += (preds == targets).sum().item()
                total_count += images.size(0)
        return total_correct / total_count

    acc_clean = run()
    acc_noise01 = run(eval_transform=lambda x: add_noise(x, 0.1))
    acc_noise03 = run(eval_transform=lambda x: add_noise(x, 0.3))
    acc_rot = run(eval_transform=lambda x: rotate_batch(x, 15))
    acc_occ = run(eval_transform=lambda x: occlude(x, 0.2))
    acc_fgsm = run(adv=True)

    deltas = {
        'noise_0.1': acc_clean - acc_noise01,
        'noise_0.3': acc_clean - acc_noise03,
        'rot15': acc_clean - acc_rot,
        'occ20': acc_clean - acc_occ,
        'fgsm': acc_clean - acc_fgsm,
    }

    print(f"Clean acc={acc_clean:.4f}; Δacc noise0.1={deltas['noise_0.1']:.4f}, noise0.3={deltas['noise_0.3']:.4f}, rot15={deltas['rot15']:.4f}, occ20={deltas['occ20']:.4f}, fgsm={deltas['fgsm']:.4f}")

    # Saccade stability example
    images, targets = next(iter(test_loader))
    images = images.to(device)[:128]
    noisy = add_noise(images, 0.1)
    drift = saccade_stability(model, images, noisy)
    print(f'Saccade drift (mean L2) under noise σ=0.1: {drift:.4f}')

    # Plot Δacc bar
    labels = list(deltas.keys())
    values = [deltas[k] for k in labels]
    plot_and_save_bar(labels, values, 'Δ accuracy (clean − perturbed)', 'ASRIN Robustness Drops', out_dir, 'delta_acc_robustness_asrin.pdf')
    # Plot saccade drift
    plot_and_save_bar(['noise_0.1'], [drift], 'mean L2 drift', 'Saccade Stability', out_dir, 'saccade_drift_noise0.1.pdf')

    return {'acc_clean': acc_clean, 'deltas': deltas, 'drift_noise0.1': drift}


def experiment_3_dynamics_kernel(asrin_model: ASRIN, cfg: Dict[str, Any], device: str, train_loader_small, test_loader_small) -> Dict[str, Any]:
    print('=== Experiment 3 (Dynamics & Kernel-limit) ===')
    model = asrin_model
    model.eval()
    out_dir = cfg['output_dir']

    # Lyapunov estimate on a small batch
    images, _ = next(iter(test_loader_small))
    images = images.to(device)[:16]
    patches = rollout_patches(model, images)  # (B,T,D)

    lams_ec = []
    for i in range(patches.size(0)):
        lams_ec.append(lyapunov_benettin(model.res, patches[i], T=100))
    lam_mean_ec = float(np.mean(lams_ec))

    # Compare with EC-SR off (g_t frozen at 0). Clone reservoir and disable gain updates.
    res_copy = Reservoir(N=model.res.N, in_dim=model.res.in_dim, device=device)
    with torch.no_grad():
        res_copy.W_in.copy_(model.res.W_in)
        res_copy.W_rec.copy_(model.res.W_rec)
        res_copy.g.zero_()

    def step_no_ec(h, x):
        u = torch.nn.functional.linear(x, res_copy.W_in) + torch.nn.functional.linear(h, res_copy.W_rec)
        h_tilde = torch.tanh(u)
        h_new = (1 - model.res.leak) * h + model.res.leak * h_tilde
        s_t = (1.0 - h_tilde.pow(2)).mean()
        return h_new, s_t
    res_copy.step = step_no_ec  # type: ignore

    lams_fixed = []
    for i in range(patches.size(0)):
        seq = patches[i]
        lams_fixed.append(lyapunov_benettin(res_copy, seq, T=100))
    lam_mean_fixed = float(np.mean(lams_fixed))

    print(f'Lyapunov (mean) with EC-SR: {lam_mean_ec:.4f}; without EC-SR: {lam_mean_fixed:.4f}')
    plot_and_save_bar(['EC-SR','No EC-SR'], [lam_mean_ec, lam_mean_fixed], 'λ_max (Benettin)', 'Reservoir Dynamics', out_dir, 'lyapunov_vs_variant.pdf')

    # Memory capacity
    mc_ec = memory_capacity(model.res, T=800, K=20, device=device)
    mc_fixed = memory_capacity(res_copy, T=800, K=20, device=device)
    print(f'Memory Capacity (approx): EC-SR={mc_ec:.2f}, No EC-SR={mc_fixed:.2f}')
    plot_and_save_bar(['EC-SR','No EC-SR'], [mc_ec, mc_fixed], 'MC (sum R^2)', 'Memory Capacity', out_dir, 'memory_capacity_vs_variant.pdf')

    # Kernel-limit (small subset)
    train_imgs, train_lbls = next(iter(train_loader_small))
    test_imgs, test_lbls = next(iter(test_loader_small))
    train_imgs = train_imgs.to(device)[:128]
    test_imgs = test_imgs.to(device)[:64]
    y_tr = train_lbls.numpy()[:train_imgs.size(0)]
    y_te = test_lbls.numpy()[:test_imgs.size(0)]
    with torch.no_grad():
        tr_seq = rollout_patches(model, train_imgs)  # (n,T,D)
        te_seq = rollout_patches(model, test_imgs)
    tr_list = [tr_seq[i] for i in range(tr_seq.size(0))]
    te_list = [te_seq[i] for i in range(te_seq.size(0))]
    print('Computing K_ASRIN Gram (train)...')
    K_tr = kasrin_kernel(tr_list)
    print('Computing K_ASRIN Gram (cross train-test)...')
    all_list = tr_list + te_list
    K_all = kasrin_kernel(all_list)
    K_te = K_all[:len(tr_list), len(tr_list):]

    C = 10  # MNIST classes
    Y = np.eye(C)[y_tr]
    lam = 1e-3
    A = K_tr + lam * np.eye(K_tr.shape[0])
    Alu = np.linalg.cholesky(A + 1e-8*np.eye(A.shape[0]))

    def chol_solve(L, B):
        y = np.linalg.solve(L, B)
        x = np.linalg.solve(L.T, y)
        return x

    W = chol_solve(Alu, Y)  # (n, C)
    scores = (K_te.T @ W)  # (m, C)
    pred = np.argmax(scores, axis=1)
    acc_kernel = (pred == y_te[:scores.shape[0]]).mean()
    print(f'K_ASRIN (subset) accuracy: {acc_kernel:.4f}')

    # Compare to finite ASRIN on same subset
    model.eval()
    with torch.no_grad():
        logits_T, _ = model(test_imgs, tau=0.5)
        logits = logits_T[:, -1, :].cpu().numpy()
    pred_finite = logits.argmax(axis=1)
    acc_finite = (pred_finite[:scores.shape[0]] == y_te[:scores.shape[0]]).mean()
    print(f'Finite ASRIN (subset) accuracy: {acc_finite:.4f}; gap={acc_finite - acc_kernel:.4f}')
    plot_and_save_bar(['Finite','Kernel'], [float(acc_finite), float(acc_kernel)], 'accuracy', 'Finite vs Kernel (subset)', out_dir, 'kernel_vs_finite_accuracy.pdf')

    return {
        'lambda_ec': lam_mean_ec,
        'lambda_fixed': lam_mean_fixed,
        'mc_ec': mc_ec,
        'mc_fixed': mc_fixed,
        'acc_kernel_subset': float(acc_kernel),
        'acc_finite_subset': float(acc_finite)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to YAML config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg['output_dir']
    ensure_output_dir(out_dir)

    device = device_selection()
    print(f'Using device: {device}')

    start_time = time.time()

    # Experiment 1
    exp1 = experiment_1_mnist(cfg, device)

    # Prepare loaders for robustness and dynamics
    from .preprocess import get_small_mnist_loaders
    train_loader_small, test_loader_small = get_small_mnist_loaders(subset=512, batch_size=128)

    # If we want to ensure an ASRIN is available (it is from exp1):
    asrin = exp1['model']

    # Optionally fine-tune briefly to ensure model is trained enough (skip to keep quick)

    # Experiment 2
    _, _, test_loader = get_mnist_loaders(
        train_subset=cfg['subsets']['train'], val_subset=cfg['subsets']['val'], test_subset=cfg['subsets']['test'],
        batch_size_train=128, batch_size_eval=256)
    exp2 = experiment_2_robustness(asrin, cfg, device, test_loader)

    # Experiment 3
    exp3 = experiment_3_dynamics_kernel(asrin, cfg, device, train_loader_small, test_loader_small)

    elapsed = time.time() - start_time
    print('=== Run Completed ===')
    print(f"Summary:\n - Exp1 ASRIN acc={exp1['asrin']['acc']:.4f} vs Raster acc={exp1['raster']['acc']:.4f}\n - Exp2 Δacc avg={np.mean(list(exp2['deltas'].values())):.4f}, drift_noise0.1={exp2['drift_noise0.1']:.4f}\n - Exp3 λ_ec={exp3['lambda_ec']:.4f}, λ_noec={exp3['lambda_fixed']:.4f}, MC_ec={exp3['mc_ec']:.2f}, MC_noec={exp3['mc_fixed']:.2f}, kernel_acc={exp3['acc_kernel_subset']:.4f}, finite_acc={exp3['acc_finite_subset']:.4f}")
    print(f'Total runtime (this config): {elapsed:.1f}s')


if __name__ == '__main__':
    main()
