import numpy as np

def edit(w, f):
    # Extract current weights
    w1 = w["fc1.weight"].copy()          # [128, 784]
    b1 = w["fc1.bias"].copy()            # [128]
    w2 = w["fc2.weight"].copy()          # [10, 128]
    b2 = w["fc2.bias"].copy()            # [10]

    # Training data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # [2000, 784]
    y = f["y"].astype(np.int64)                                 # [2000]
    N = X.shape[0]

    # Identify the 9 most negative-bias units (likely dead)
    dead_idx = np.argsort(b1)[:9]

    # Compute LDA directions on raw pixels
    classes = np.unique(y)
    mu = X.mean(axis=0)  # [784]
    Sw = np.zeros((784, 784), dtype=np.float64)
    Sb = np.zeros((784, 784), dtype=np.float64)
    for c in classes:
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        n_c = Xc.shape[0]
        diff = Xc - mu_c
        Sw += diff.T @ diff
        d = mu_c - mu
        Sb += n_c * np.outer(d, d)

    # Regularize Sw slightly for numerical stability
    Sw_reg = Sw + 1e-2 * np.eye(784)
    # Solve Sw_reg * V = Sb * V  =>  V = Sw_reg^{-1} Sb V
    M = np.linalg.solve(Sw_reg, Sb)
    eigvals, eigvecs = np.linalg.eig(M)
    # Keep real parts and sort by eigenvalue magnitude
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    order = np.argsort(eigvals)[::-1]
    lda_dirs = eigvecs[:, order[:9]].T  # [9, 784]

    # Normalize and center each LDA direction
    for i in range(9):
        v = lda_dirs[i]
        v = v / (np.linalg.norm(v) + 1e-12)
        proj = X @ v
        std = proj.std() + 1e-12
        v = v / std
        lda_dirs[i] = v

    # Replace dead units with LDA directions
    for i, idx in enumerate(dead_idx):
        w1[idx] = lda_dirs[i].astype(np.float32)
        proj = X @ lda_dirs[i]
        b1[idx] = -proj.mean()

    # Compute new hidden features
    Z = X @ w1.T + b1  # [2000, 128]
    H = np.maximum(Z, 0.0)

    # Refit fc2 via ridge regression on one-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    H_aug = np.hstack([H, np.ones((N, 1))])  # [2000, 129]
    alpha = 1e-2
    A = H_aug.T @ H_aug + alpha * np.eye(129)
    W_aug = np.linalg.solve(A, H_aug.T @ Y)  # [129, 10]
    W_aug = W_aug.T  # [10, 129]
    w2_new = W_aug[:, :128]
    b2_new = W_aug[:, 128]

    # Clip to safe range
    w1 = np.clip(w1, -1000, 1000).astype(np.float32)
    b1 = np.clip(b1, -1000, 1000).astype(np.float32)
    w2_new = np.clip(w2_new, -1000, 1000).astype(np.float32)
    b2_new = np.clip(b2_new, -1000, 1000).astype(np.float32)

    # Update weights
    w["fc1.weight"] = w1
    w["fc1.bias"] = b1
    w["fc2.weight"] = w2_new
    w["fc2.bias"] = b2_new

    # Diagnostics
    logits = H @ w2_new.T + b2_new
    preds = np.argmax(logits, axis=1)
    acc = float((preds == y).mean())
    diagnostics = {
        "replaced_units": dead_idx.tolist(),
        "lda_eigenvalues": eigvals[order[:9]].tolist(),
        "full_pool_accuracy": acc,
        "ridge_alpha": alpha,
        "fc1_weight_norm": float(np.linalg.norm(w1)),
        "fc2_weight_norm": float(np.linalg.norm(w2_new)),
    }
    return {"weights": w, "diagnostics": diagnostics}
