import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (N, 784)
    y = f["y"].astype(np.int64)                                 # (N,)
    N = X.shape[0]
    D = X.shape[1]
    H = 128
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current weights
    W1 = w["fc1.weight"].astype(np.float64)   # (128, 784)
    b1 = w["fc1.bias"].astype(np.float64)     # (128,)
    W2 = w["fc2.weight"].astype(np.float64)   # (10, 128)
    b2 = w["fc2.bias"].astype(np.float64)     # (10,)

    # Current hidden features (after ReLU) and logits
    H_cur = f["features"].astype(np.float64)  # (N, 128)
    L_cur = f["logits"].astype(np.float64)    # (N, 10)

    # ------------------------------------------------------------------
    # 1. Compute a target hidden representation H_new that would yield
    #    the one-hot labels through the current fc2 (minimum-norm change).
    # ------------------------------------------------------------------
    R = Y - L_cur                             # (N, 10)
    # SVD of W2 (10 x 128)
    U, S, Vt = np.linalg.svd(W2, full_matrices=False)
    # S has length 10 (rank of W2)
    # delta = R @ U @ diag(1/S) @ Vt
    delta = (R @ U) / S                       # (N, 10) * (10,) -> (N, 10)
    delta = delta @ Vt                        # (N, 10) @ (10, 128) -> (N, 128)
    H_new = H_cur + delta                     # (N, 128)

    # ------------------------------------------------------------------
    # 2. Fit new fc1 (W1_new, b1_new) to approximate H_new from X.
    #    Ridge regression per hidden unit.
    # ------------------------------------------------------------------
    # Augment X with a bias column
    X_aug = np.concatenate([X, np.ones((N, 1))], axis=1)  # (N, 785)
    # Ridge parameter (small to avoid overfitting)
    lam1 = 1e-2
    # Precompute X_aug^T X_aug + lam1 * I
    XtX = X_aug.T @ X_aug
    reg = lam1 * np.eye(D + 1)
    # Solve for all units at once: (XtX + reg) W_aug = X_aug^T H_new
    # W_aug is (785, 128)
    W_aug = np.linalg.solve(XtX + reg, X_aug.T @ H_new)
    W1_new = W_aug[:D, :].T                   # (128, 784)
    b1_new = W_aug[D, :]                      # (128,)

    # ------------------------------------------------------------------
    # 3. Compute new hidden features with actual ReLU.
    # ------------------------------------------------------------------
    pre_act = X @ W1_new.T + b1_new           # (N, 128)
    H_new_act = np.maximum(pre_act, 0.0)      # (N, 128)

    # ------------------------------------------------------------------
    # 4. Refit fc2 (W2_new, b2_new) on the new features.
    #    Ridge regression for 10 classes.
    # ------------------------------------------------------------------
    # Augment H_new_act with bias
    H_aug = np.concatenate([H_new_act, np.ones((N, 1))], axis=1)  # (N, 129)
    lam2 = 1e-2
    HtH = H_aug.T @ H_aug
    reg2 = lam2 * np.eye(H + 1)
    # Solve for all classes: (HtH + reg2) W2_aug = H_aug^T Y
    W2_aug = np.linalg.solve(HtH + reg2, H_aug.T @ Y)  # (129, 10)
    W2_new = W2_aug[:H, :].T                  # (10, 128)
    b2_new = W2_aug[H, :]                     # (10,)

    # ------------------------------------------------------------------
    # 5. Optional: scale logits to match original confidence? Not needed
    #    for accuracy, but can help cross-entropy. We'll leave as is.
    # ------------------------------------------------------------------

    # Write back to w (preserve float32 and shapes)
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"]   = b1_new.astype(np.float32)
    w["fc2.weight"] = W2_new.astype(np.float32)
    w["fc2.bias"]   = b2_new.astype(np.float32)

    # Diagnostics
    diag = {
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc1_bias_norm": float(np.linalg.norm(w["fc1.bias"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "dead_units_before": int(np.sum(np.all(H_cur == 0, axis=0))),
        "dead_units_after": int(np.sum(np.all(H_new_act == 0, axis=0))),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
    }

    return {"weights": w, "diagnostics": diag}
