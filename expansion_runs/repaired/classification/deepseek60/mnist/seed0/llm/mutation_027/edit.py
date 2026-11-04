import numpy as np

def edit(w, f):
    # Extract current fc2 weights and biases
    W0 = w["fc2.weight"].astype(np.float64)  # (10, 128)
    b0 = w["fc2.bias"].astype(np.float64)    # (10,)

    # Feedback features and labels
    X = f["features"].astype(np.float64)     # (N, 128)
    y = f["y"].astype(np.int64)              # (N,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Add bias column to features
    X_aug = np.hstack([X, np.ones((N, 1))])  # (N, D+1)
    W0_aug = np.hstack([W0, b0[:, None]])    # (C, D+1)

    # Hold-out split for lambda selection
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_val = max(200, N // 5)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    X_tr = X_aug[tr_idx]
    Y_tr = Y[tr_idx]
    X_val = X_aug[val_idx]
    Y_val = Y[val_idx]

    # Candidate lambdas (prior strength)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    best_lambda = None
    best_acc = -1.0
    best_W = None

    for lam in lambdas:
        # Ridge regression with prior towards W0_aug:
        # minimize ||X W^T - Y||^2 + lam * ||W - W0||^2
        # Solution: W = (X^T X + lam I)^{-1} (X^T Y + lam W0)
        A = X_tr.T @ X_tr + lam * np.eye(D + 1)
        B = X_tr.T @ Y_tr + lam * W0_aug.T
        W_sol = np.linalg.solve(A, B).T  # (C, D+1)

        # Evaluate on validation
        logits_val = X_val @ W_sol.T
        pred_val = np.argmax(logits_val, axis=1)
        acc = np.mean(pred_val == y[val_idx])
        if acc > best_acc:
            best_acc = acc
            best_lambda = lam
            best_W = W_sol

    # Refit on full data with best lambda
    lam = best_lambda if best_lambda is not None else 1.0
    A_full = X_aug.T @ X_aug + lam * np.eye(D + 1)
    B_full = X_aug.T @ Y + lam * W0_aug.T
    W_final = np.linalg.solve(A_full, B_full).T  # (C, D+1)

    # Extract weights and biases
    W_new = W_final[:, :D]
    b_new = W_final[:, D]

    # Clip to safe range
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["fc2.weight"] = W_new
    w["fc2.bias"] = b_new

    # Diagnostics
    diag = {
        "best_lambda": float(lam),
        "val_accuracy": float(best_acc),
        "fc2_weight_norm": float(np.linalg.norm(W_new)),
        "fc2_bias_norm": float(np.linalg.norm(b_new)),
        "fc2_weight_mean": float(np.mean(W_new)),
        "fc2_weight_std": float(np.std(W_new)),
    }
    return {"weights": w, "diagnostics": diag}
