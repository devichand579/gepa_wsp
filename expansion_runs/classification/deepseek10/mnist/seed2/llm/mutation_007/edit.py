import numpy as np

def edit(w, f):
    # Read feedback pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # [2000, 784]
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # ---- 1. Closed-form linear classifier on raw pixels via ridge ----
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])  # [2000, 785]
    lam_lin = 1e-3
    A = X_aug.T @ X_aug + lam_lin * np.eye(785, dtype=np.float32)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B).T.astype(np.float32)  # [10, 785]
    W_lin = W_aug[:, :784]      # [10, 784]
    b_lin = W_aug[:, 784]       # [10]

    # Ensure ReLU is identity for these 10 units on the feedback pool
    lin_vals = X @ W_lin.T + b_lin
    min_val = float(np.min(lin_vals))
    shift = -min_val + 1.0
    shift = min(shift, 1000.0)

    # ---- 2. Build new fc1 ----
    fc1_w = np.zeros((128, 784), dtype=np.float32)
    fc1_b = np.zeros(128, dtype=np.float32)

    # First 10 rows: exact linear classifier (shifted to stay positive)
    fc1_w[:10] = W_lin
    fc1_b[:10] = b_lin + shift

    # Remaining 118 rows: random ReLU features (He-like scale)
    rng = np.random.RandomState(0)
    scale = np.sqrt(2.0 / 784.0)
    fc1_w[10:] = rng.randn(118, 784).astype(np.float32) * scale
    fc1_b[10:] = 0.0

    # ---- 3. Compute hidden features for the whole feedback pool ----
    H = np.maximum(X @ fc1_w.T + fc1_b, 0.0).astype(np.float32)  # [2000, 128]

    # ---- 4. Fit fc2 by ridge regression on hidden features ----
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])  # [2000, 129]
    lam_fc2 = 1e-3
    A2 = H_aug.T @ H_aug + lam_fc2 * np.eye(129, dtype=np.float32)
    B2 = H_aug.T @ Y
    W2_aug = np.linalg.solve(A2, B2).T.astype(np.float32)  # [10, 129]
    fc2_w = W2_aug[:, :128]
    fc2_b = W2_aug[:, 128]

    # ---- 5. Write back ----
    w["fc1.weight"] = fc1_w
    w["fc1.bias"] = fc1_b
    w["fc2.weight"] = fc2_w
    w["fc2.bias"] = fc2_b

    # Diagnostics
    diagnostics = {
        "shift": float(shift),
        "fc1_weight_norm": float(np.linalg.norm(fc1_w)),
        "fc1_bias_norm": float(np.linalg.norm(fc1_b)),
        "fc2_weight_norm": float(np.linalg.norm(fc2_w)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_b)),
        "hidden_mean": float(np.mean(H)),
        "hidden_std": float(np.std(H)),
        "hidden_zero_fraction": float(np.mean(H == 0.0)),
    }
    return {"weights": w, "diagnostics": diagnostics}
