import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # Current weights (for reference / candidate inclusion)
    W1_cur = w["fc1.weight"].astype(np.float64)  # (128, 784)

    # Class statistics
    mu = X.mean(axis=0)
    mu_c = np.zeros((C, D))
    n_c = np.zeros(C, dtype=int)
    for c in range(C):
        idx = (y == c)
        n_c[c] = np.sum(idx)
        if n_c[c] > 0:
            mu_c[c] = X[idx].mean(axis=0)

    candidates = []

    # 1. Current fc1 rows
    for i in range(W1_cur.shape[0]):
        d = W1_cur[i]
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            candidates.append(d / nrm)

    # 2. Class means (centered)
    for c in range(C):
        d = mu_c[c] - mu
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            candidates.append(d / nrm)

    # 3. Pairwise class mean differences
    for i in range(C):
        for j in range(i + 1, C):
            d = mu_c[i] - mu_c[j]
            nrm = np.linalg.norm(d)
            if nrm > 1e-8:
                candidates.append(d / nrm)

    # 4. LDA directions
    Sw = np.zeros((D, D))
    for c in range(C):
        idx = (y == c)
        if n_c[c] > 1:
            Xc = X[idx]
            mu_c_c = mu_c[c]
            Sw += Xc.T @ Xc - n_c[c] * np.outer(mu_c_c, mu_c_c)
    Sw_reg = Sw + 1e-3 * np.eye(D)
    Sw_inv = np.linalg.pinv(Sw_reg)
    for c in range(C):
        d = Sw_inv @ (mu_c[c] - mu)
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            candidates.append(d / nrm)

    # Top LDA eigenvectors
    Sb = np.zeros((D, D))
    for c in range(C):
        diff = mu_c[c] - mu
        Sb += n_c[c] * np.outer(diff, diff)
    M = Sw_inv @ Sb
    eigvals, eigvecs = np.linalg.eig(M)
    idx_sort = np.argsort(-np.real(eigvals))
    for i in idx_sort[:9]:
        d = np.real(eigvecs[:, i])
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            candidates.append(d / nrm)

    # 5. PCA directions (top 128)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    for i in range(min(128, Vt.shape[0])):
        d = Vt[i]
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            candidates.append(d / nrm)

    # 6. Ridge regression directions (one-vs-rest)
    XTX = X.T @ X
    for lam in [1.0, 0.1, 0.01]:
        A = XTX + lam * np.eye(D)
        for c in range(C):
            y_c = (y == c).astype(np.float64)
            XTy = X.T @ y_c
            try:
                d = np.linalg.solve(A, XTy)
            except np.linalg.LinAlgError:
                d = np.linalg.pinv(A) @ XTy
            nrm = np.linalg.norm(d)
            if nrm > 1e-8:
                candidates.append(d / nrm)

    # 7. Random projections
    rng = np.random.RandomState(42)
    for _ in range(100):
        d = rng.randn(D)
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            candidates.append(d / nrm)

    # Compute ANOVA F-values for all candidates
    D_mat = np.array(candidates)  # (num_cand, D)
    Z = X @ D_mat.T               # (N, num_cand)
    z_mean = Z.mean(axis=0)
    total_var = np.sum((Z - z_mean) ** 2, axis=0)
    between = np.zeros(D_mat.shape[0])
    for c in range(C):
        idx = (y == c)
        if n_c[c] > 0:
            mean_c = Z[idx].mean(axis=0)
            between += n_c[c] * (mean_c - z_mean) ** 2
    within = total_var - between
    F_vals = (between / (C - 1)) / (within / (N - C) + 1e-12)

    # Select top 128 directions
    top_idx = np.argsort(-F_vals)[:128]
    selected_dirs = D_mat[top_idx]  # (128, D)

    # Build new fc1
    W1_new = selected_dirs.astype(np.float32)
    Z_sel = X @ selected_dirs.T
    b1_new = (-np.median(Z_sel, axis=0)).astype(np.float32)

    # Hidden features
    H = np.maximum(0, Z_sel + b1_new)  # (N, 128)

    # Fit fc2 via ridge regression with intercept
    H_aug = np.hstack([H, np.ones((N, 1))])
    Y = np.zeros((N, C))
    for c in range(C):
        Y[y == c, c] = 1.0

    lam = 0.1
    A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1])
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # (129, 10)

    W2_new = W_aug[:-1].T.astype(np.float32)  # (10, 128)
    b2_new = W_aug[-1].astype(np.float32)     # (10,)

    # Update weights
    w["fc1.weight"] = W1_new
    w["fc1.bias"] = b1_new
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new

    # Diagnostics
    logits = H @ W2_new.T + b2_new
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "num_candidates": int(D_mat.shape[0]),
        "selected_indices": top_idx[:10].tolist(),
        "fc1_weight_mean": float(W1_new.mean()),
        "fc1_weight_std": float(W1_new.std()),
        "fc2_weight_mean": float(W2_new.mean()),
        "fc2_weight_std": float(W2_new.std()),
        "fc2_bias_mean": float(b2_new.mean()),
        "fc2_bias_std": float(b2_new.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
