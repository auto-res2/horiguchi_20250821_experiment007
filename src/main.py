import os
import argparse
import yaml
import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge

from .preprocess import get_loaders, ensure_dir
from .train import set_seed, ASRIN, ASRIN_NoDSE, RasterESN, SmallCNN, RandomFeatureMLP, train_one_epoch
from .evaluate import (
    evaluate,
    evaluate_early_exit,
    evaluate_corruptions,
    estimate_lle,
    memory_capacity,
    build_sequences,
    build_kernel,
    cka,
    plot_training_losses,
    plot_accuracy_bars,
    plot_confusion_matrix_pdf,
    plot_robustness_bars,
    plot_early_exit,
    plot_lle_hist,
    plot_kernel_accuracy_vs_N,
    plot_kernel_alignment_vs_N,
)


def get_device(pref='auto'):
    if pref == 'cpu':
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def run_experiment1(cfg, images_dir):
    print('--- Experiment 1: Accuracy–Efficiency on clean datasets ---')
    device = cfg['device']
    dataset = cfg['dataset']
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    N = cfg['N']
    P = cfg['P']
    T = cfg['T']
    subset_train = cfg.get('subset_train', None)
    subset_test = cfg.get('subset_test', None)

    print(f'Dataset={dataset}, device={device}, epochs={epochs}, N={N}, P={P}, T={T}')
    loader_train, loader_test, n_classes = get_loaders(dataset, batch_size=batch_size, aug=True, subset_train=subset_train, subset_test=subset_test)

    results = {}
    loss_hist = {}

    set_seed(cfg['seeds'][0])

    # Models
    models = {}
    models['ASRIN'] = ASRIN(N=N, P=P, T=T, n_classes=n_classes, device=device, enable_ecsr=True).to(device)
    models['ASRIN-NoDSE'] = ASRIN_NoDSE(N=N, P=P, T=T, n_classes=n_classes, device=device, enable_ecsr=True).to(device)
    models['ASRIN-NoECSR'] = ASRIN(N=N, P=P, T=T, n_classes=n_classes, device=device, enable_ecsr=False).to(device)
    models['RasterESN'] = RasterESN(N=N, T=28 * 28, n_classes=n_classes, device=device).to(device)
    models['CNN'] = SmallCNN(n_classes=n_classes).to(device)
    models['RF-MLP'] = RandomFeatureMLP(hidden=512, n_classes=n_classes).to(device)

    for name, model in models.items():
        print(f'\nTraining model: {name}')
        if name == 'ASRIN' or name == 'ASRIN-NoECSR':
            params = [
                {'params': list(model.policy.parameters()), 'lr': 2e-3},
                {'params': list(model.readout.parameters()), 'lr': 1e-2},
            ]
        elif name == 'ASRIN-NoDSE' or name == 'RasterESN':
            params = [{'params': list(model.readout.parameters()), 'lr': 1e-2}]
        elif name == 'CNN':
            params = [{'params': model.parameters(), 'lr': 1e-3}]
        else:  # RF-MLP
            params = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 1e-2}]

        opt = torch.optim.Adam(params)
        total_steps = epochs * len(loader_train)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        losses = []
        for ep in range(epochs):
            tau = 1.0 - 0.7 * (ep / (max(1, epochs - 1))) if isinstance(model, ASRIN) else 0.3
            acc_tr, loss_tr = train_one_epoch(model, loader_train, opt, device, scheduler, tau=tau)
            losses.append(loss_tr)
            if ep == epochs - 1:
                print(f'  Epoch {ep + 1}/{epochs} | train acc={acc_tr:.3f} loss={loss_tr:.3f}')
        loss_hist[name] = losses
        acc_te, loss_te, ece = evaluate(model, loader_test, device)
        print(f'  Test acc={acc_te:.4f} | ECE={ece:.4f}')
        results[name] = {'acc': acc_te, 'ece': ece, 'loss_te': loss_te}

    # Plots for training loss (ASRIN and RasterESN)
    if 'ASRIN' in loss_hist:
        plot_training_losses({'asrin': loss_hist['ASRIN']}, images_dir, fname_prefix='training_loss_asrin')
    if 'RasterESN' in loss_hist:
        plot_training_losses({'raster': loss_hist['RasterESN']}, images_dir, fname_prefix='training_loss_baseline')

    # Accuracy bars
    accs = {k: v['acc'] for k, v in results.items()}
    plot_accuracy_bars(accs, images_dir, fname='accuracy_baselines')

    # Confusion matrix for ASRIN
    asrin = models['ASRIN']
    acc, _, _, logits_np, labels_np = evaluate(asrin, loader_test, device, return_logits_labels=True)
    y_pred = logits_np.argmax(axis=1)
    plot_confusion_matrix_pdf(labels_np, y_pred, images_dir, fname='confusion_matrix_asrin')

    # Early-exit curve for ASRIN
    acc_t = evaluate_early_exit(asrin, loader_test, device)
    plot_early_exit(acc_t, images_dir, fname='early_exit_accuracy_asrin')

    # Print FLOPs (rough estimate) and sequence lengths
    class FlopCounter:
        def __init__(self, nnz_W, N, input_dim):
            self.nnz_W = nnz_W
            self.N = N
            self.input_dim = input_dim
        @property
        def per_step(self):
            return self.nnz_W + self.N * self.input_dim
        def per_sample(self, TT):
            return TT * self.per_step

    def estimate_flops(model_):
        if isinstance(model_, ASRIN) or isinstance(model_, ASRIN_NoDSE):
            N_ = model_.esn.N
            P_ = model_.P
            T_ = model_.T
            nnz = model_.esn.W_val.numel()
            return FlopCounter(nnz, N_, P_ * P_).per_sample(T_)
        if isinstance(model_, RasterESN):
            N_ = model_.esn.N
            T_ = model_.T
            nnz = model_.esn.W_val.numel()
            return FlopCounter(nnz, N_, 1).per_sample(T_)
        return None

    print('\nCompute profile (approx mults/sample):')
    print(f"  ASRIN:   T={T:>3}  FLOPs≈{estimate_flops(models['ASRIN']):,}")
    print(f"  Raster:  T={28 * 28}  FLOPs≈{estimate_flops(models['RasterESN']):,}")

    return results, models, loader_test


def run_experiment2(asrin_model: ASRIN, baseline_model, loader_test, device, images_dir):
    print('\n--- Experiment 2: Robustness and dynamical self-tuning ---')
    # Robustness deltas
    deltas_asrin = evaluate_corruptions(asrin_model, loader_test, device)
    deltas_base = evaluate_corruptions(baseline_model, loader_test, device)
    print('ASRIN robustness (Δacc):', deltas_asrin)
    print('Baseline (Raster ESN) robustness (Δacc):', deltas_base)
    plot_robustness_bars(deltas_asrin, images_dir, fname='robustness_drop_asrin')

    # Dynamics: LLE and MC (sample few batches for speed)
    lle_vals = []
    for i, (images, labels) in enumerate(loader_test):
        images = images.to(device)
        lle = estimate_lle(asrin_model, images)
        lle_vals.append(lle)
        if i >= 2:
            break  # quick
    print(f'Estimated LLE (mean over batches) = {float(np.mean(lle_vals)):.4f}')
    plot_lle_hist(lle_vals, images_dir, fname='lle_asrin')

    mc = memory_capacity(asrin_model, T=400, K=20)
    print(f'Estimated Memory Capacity (MC) ≈ {mc:.2f}')

    return deltas_asrin, deltas_base, lle_vals, mc


def run_experiment3(asrin_model: ASRIN, loader_train_small, loader_test_small, device, images_dir, N_list):
    print('\n--- Experiment 3: Kernel correspondence and finite-width scaling ---')
    # Freeze policy, build sequences
    policy = asrin_model.policy
    sampler = asrin_model.sampler.to(device)
    n_classes = asrin_model.n_classes
    Xtr, Ytr = build_sequences(policy, sampler, loader_train_small, T=asrin_model.T, P=asrin_model.P, tau=0.05, device=device, n_classes=n_classes)
    Xte, Yte = build_sequences(policy, sampler, loader_test_small, T=asrin_model.T, P=asrin_model.P, tau=0.05, device=device, n_classes=n_classes)

    print('Building K_ASRIN kernel (train/test) ...')
    Ktr = build_kernel(Xtr, alpha=0.3, g=1.0, rho=1.0, sigma_in=1.0, lam_dual=1.0)
    Kte = build_kernel(np.concatenate([Xte, Xtr[:1]], axis=0), alpha=0.3, g=1.0, rho=1.0, sigma_in=1.0, lam_dual=1.0)[:len(Xte), :len(Xtr)]

    # Simple KRR (one-vs-rest)
    alpha = 1e-2
    scores = []
    for c in range(n_classes):
        y_bin = np.where(Ytr == c, 1.0, -1.0)
        clf = KernelRidge(alpha=alpha, kernel='precomputed')
        clf.fit(Ktr, y_bin)
        scores.append(clf.predict(Kte))
    S = np.stack(scores, axis=1)
    y_pred_krr = S.argmax(axis=1)
    acc_krr = (y_pred_krr == Yte).mean()
    print(f'K_ASRIN (KRR) test accuracy: {acc_krr:.4f}')

    # Finite reservoirs with same sequences (train readout only)
    from .train import SparseESN
    import torch.nn as nn

    accs_finite = []
    ckas = []
    for N in N_list:
        model = SparseESN(N=N, input_dim=Xtr.shape[2], device=device)
        readout = nn.Linear(N + Xtr.shape[2], n_classes).to(device)
        # Prepare tensors
        Xtr_t = torch.from_numpy(Xtr).float().to(device)  # (n,T,d)
        Xte_t = torch.from_numpy(Xte).float().to(device)
        Ytr_t = torch.from_numpy(Ytr).long().to(device)
        Yte_t = torch.from_numpy(Yte).long().to(device)
        # Compute last states
        model.eval()
        with torch.no_grad():
            Htr = []
            for i in range(len(Xtr_t)):
                hs = model(Xtr_t[i].unsqueeze(1), enable_ecsr=True)  # (T,1,N)
                Htr.append(hs[-1, 0])
            Htr = torch.stack(Htr, dim=0)  # (n,N)
            Hte = []
            for i in range(len(Xte_t)):
                hs = model(Xte_t[i].unsqueeze(1), enable_ecsr=True)
                Hte.append(hs[-1, 0])
            Hte = torch.stack(Hte, dim=0)
        # Train readout (ridge-like with Adam)
        opt = torch.optim.Adam(readout.parameters(), lr=1e-2)
        ce = nn.CrossEntropyLoss()
        for _ep in range(20):
            opt.zero_grad()
            logits = readout(torch.cat([Xtr_t[:, -1, :], Htr], dim=1))
            loss = ce(logits, Ytr_t)
            loss.backward(); opt.step()
        with torch.no_grad():
            logits = readout(torch.cat([Xte_t[:, -1, :], Hte], dim=1))
            y_pred = logits.argmax(1).cpu().numpy()
            acc = (y_pred == Yte).mean()
            accs_finite.append(acc)
        # CKA between reservoir Gram and kernel (use train set)
        K_res = (Htr.cpu().numpy() @ Htr.cpu().numpy().T)
        ckas.append(cka(K_res, Ktr))
        print(f'  N={N} | finite acc={acc:.4f} | CKA={ckas[-1]:.3f}')

    plot_kernel_accuracy_vs_N(N_list, accs_finite, acc_krr, images_dir, fname='kernel_accuracy_pair1')
    plot_kernel_alignment_vs_N(N_list, ckas, images_dir, fname='kernel_alignment_pair2')

    return acc_krr, accs_finite, ckas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/experiment.yaml')
    parser.add_argument('--run_mode', type=str, default='quick', choices=['quick', 'full'])
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        # Fallback minimal config
        cfg = {
            'dataset': 'MNIST',
            'seeds': [0],
            'epochs': 2,
            'batch_size': 64,
            'N': 128,
            'P': 8,
            'T': 6,
            'subset_train': 512,
            'subset_test': 256,
        }

    device = get_device(cfg.get('device', 'auto'))
    cfg['device'] = device

    # Prepare image output dir (iteration5 as requested)
    images_dir = os.path.join('.research', 'iteration5', 'images')
    ensure_dir(images_dir)

    # Run Experiment 1
    results, models, loader_test = run_experiment1(cfg, images_dir)

    # Experiment 2: Robustness and dynamics
    asrin = models['ASRIN']
    raster = models['RasterESN']
    deltas_asrin, deltas_base, lle_vals, mc = run_experiment2(asrin, raster, loader_test, device, images_dir)

    # Experiment 3: Kernel correspondence on small split
    loader_train_small, loader_test_small, _ = get_loaders(cfg['dataset'], batch_size=64, aug=False, subset_train=min(256, cfg.get('subset_train', 512)), subset_test=min(128, cfg.get('subset_test', 256)))
    acc_krr, accs_finite, ckas = run_experiment3(asrin, loader_train_small, loader_test_small, device, images_dir, N_list=[max(32, cfg['N']//4), cfg['N']//2])

    print('Saved figures to:', images_dir)
    print(' - training_loss_asrin.pdf')
    print(' - training_loss_baseline.pdf')
    print(' - accuracy_baselines.pdf')
    print(' - confusion_matrix_asrin.pdf')
    print(' - early_exit_accuracy_asrin.pdf')
    print(' - robustness_drop_asrin.pdf')
    print(' - lle_asrin.pdf')
    print(' - kernel_accuracy_pair1.pdf')
    print(' - kernel_alignment_pair2.pdf')


if __name__ == '__main__':
    main()
