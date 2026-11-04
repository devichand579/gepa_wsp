import numpy as np

def edit(w, f):
    # ---------- load data ----------
    X = f["x"].reshape(2000, 784).astype(np.float64)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    classes = np.arange(10)

    # ---------- validation split ----------
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    tr = perm[:1600]
    va = perm[1600:]
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]

    # ---------- class means / global mean ----------
    Mtr = np.array([Xtr[ytr == c].mean(0) for c in classes])   # [10,784]
    mu_tr = Xtr.mean(0)

    # ---------- within-class scatter ----------
    Rtr = Xtr - Mtr[ytr]
    Sw = (Rtr.T @ Rtr) / len(tr)

    # ---------- LDA directions via regularized Sw ----------
    # We will try a few regularization strengths.
    lda_regs = [1e-3, 1e-2, 1e-1, 1.0]
    lda_dirs = {}
    for reg in lda_regs:
        Sw_reg = Sw + reg * np.eye(784)
        L = np.linalg.cholesky(Sw_reg)
        # B = (Mtr - mu_tr).T * sqrt(n_c)  -> [784,10]
        n_c = np.array([(ytr == c).sum() for c in classes], dtype=np.float64)
        B = (Mtr - mu_tr).T * np.sqrt(n_c)
        Z = np.linalg.solve(L, B)          # [784,10]
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        # back-transform: v = L^{-T} U
        V = np.linalg.solve(L.T, U)        # [784,10]
        # take first 9 (10th is redundant)
        lda_dirs[reg] = V[:, :9].T         # [9,784]

    # ---------- PCA directions on centered train data ----------
    Xc = Xtr - mu_tr
    # SVD of Xc (1600x784): right singular vectors are PCA directions
    # We only need top 64.
    U_pca, S_pca, Vt_pca = np.linalg.svd(Xc, full_matrices=False)
    pca_dirs = Vt_pca[:64]                 # [64,784]

    # ---------- helper: build features from a set of directions ----------
    def build_feats(dirs, Xdata):
        # dirs: [D,784]
        proj = Xdata @ dirs.T              # [N,D]
        # center each projection using train mean
        mean_proj = (Xtr @ dirs.T).mean(0)
        proj = proj - mean_proj
        # duplicate: positive and negative ReLU
        pos = np.maximum(proj, 0.0)
        neg = np.maximum(-proj, 0.0)
        # interleave: [pos0, neg0, pos1, neg1, ...]
        feats = np.empty((Xdata.shape[0], 2 * dirs.shape[0]), dtype=np.float64)
        feats[:, 0::2] = pos
        feats[:, 1::2] = neg
        return feats

    # ---------- search over LDA/PCA split and fc2 ridge alpha ----------
    best = None
    best_acc = -1.0
    # We need exactly 64 directions total (128 units after duplication).
    # Try n_lda from 0 to 9, n_pca = 64 - n_lda.
    for reg in lda_regs:
        LDA = lda_dirs[reg]                # [9,784]
        for n_lda in range(10):
            n_pca = 64 - n_lda
            if n_pca < 0 or n_pca > 64:
                continue
            dirs = np.vstack([LDA[:n_lda], pca_dirs[:n_pca]])  # [64,784]
            Ftr = build_feats(dirs, Xtr)
            Fva = build_feats(dirs, Xva)
            # one-hot targets
            Ytr = np.eye(10)[ytr]
            # ridge regression: W = (F^T F + alpha I)^{-1} F^T Y
            # We'll try a few alphas.
            for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                A = Ftr.T @ Ftr + alpha * np.eye(Ftr.shape[1])
                Bmat = Ftr.T @ Ytr
                W = np.linalg.solve(A, Bmat)   # [128,10]
                bias = Ytr.mean(0) - Ftr.mean(0) @ W
                logits = Fva @ W + bias
                pred = logits.argmax(1)
                acc = (pred == yva).mean()
                if acc > best_acc:
                    best_acc = acc
                    best = (reg, n_lda, n_pca, alpha, W, bias, dirs)

    # ---------- refit on full data with best hyperparameters ----------
    reg, n_lda, n_pca, alpha, _, _, _ = best

    # Recompute LDA and PCA on full data
    M = np.array([X[y == c].mean(0) for c in classes])
    mu = X.mean(0)
    R = X - M[y]
    Sw_full = (R.T @ R) / N
    Sw_reg = Sw_full + reg * np.eye(784)
    L = np.linalg.cholesky(Sw_reg)
    n_c = np.array([(y == c).sum() for c in classes], dtype=np.float64)
    B = (M - mu).T * np.sqrt(n_c)
    Z = np.linalg.solve(L, B)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    V = np.linalg.solve(L.T, U)
    LDA_full = V[:, :9].T                  # [9,784]

    Xc_full = X - mu
    U_pca, S_pca, Vt_pca = np.linalg.svd(Xc_full, full_matrices=False)
    pca_full = Vt_pca[:64]                 # [64,784]

    dirs_full = np.vstack([LDA_full[:n_lda], pca_full[:n_pca]])  # [64,784]

    # Build final fc1 weights and biases
    # For each direction v, bias = -mean(v^T x) on full data.
    proj_full = X @ dirs_full.T            # [2000,64]
    mean_proj = proj_full.mean(0)
    fc1_w = np.zeros((128, 784), dtype=np.float32)
    fc1_b = np.zeros(128, dtype=np.float32)
    for i in range(64):
        v = dirs_full[i].astype(np.float32)
        b = -mean_proj[i].astype(np.float32)
        fc1_w[2*i] = v
        fc1_b[2*i] = b
        fc1_w[2*i+1] = -v
        fc1_b[2*i+1] = -b

    # Compute final hidden features
    H = np.maximum(X @ fc1_w.T + fc1_b, 0.0)   # [2000,128]

    # Fit fc2 by ridge on full data
    Y = np.eye(10)[y]
    A = H.T @ H + alpha * np.eye(128)
    Bmat = H.T @ Y
    W2 = np.linalg.solve(A, Bmat)              # [128,10]
    b2 = Y.mean(0) - H.mean(0) @ W2

    # Write back
    w["fc1.weight"] = fc1_w.astype(np.float32)
    w["fc1.bias"] = fc1_b.astype(np.float32)
    w["fc2.weight"] = W2.T.astype(np.float32)  # [10,128]
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    logits_full = H @ W2 + b2
    pred_full = logits_full.argmax(1)
    acc_full = (pred_full == y).mean()
    ce_full = -np.log(np.exp(logits_full - logits_full.max(1, keepdims=True)).sum(1) / np.exp(logits_full - logits_full.max(1, keepdims=True)).sum(1)[:, None] + 1e-12)
    # simpler CE
    log_probs = logits_full - logits_full.max(1, keepdims=True)
    log_probs -= np.log(np.exp(log_probs).sum(1, keepdims=True))
    ce = -log_probs[np.arange(N), y].mean()

    diag = {
        "val_acc": float(best_acc),
        "full_acc": float(acc_full),
        "full_ce": float(ce),
        "best_reg": float(reg),
        "best_n_lda": int(n_lda),
        "best_n_pca": int(n_pca),
        "best_alpha": float(alpha),
        "fc1_weight_std": float(fc1_w.std()),
        "fc2_weight_std": float(W2.std()),
    }
    return {"weights": w, "diagnostics": diag}
