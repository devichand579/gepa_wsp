import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # [2000, 128]
    y = f["y"].astype(np.int64)           # [2000]
    n, d = X.shape
    num_classes = 10

    # One-hot targets
    Y = np.zeros((n, num_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Split for ridge lambda selection: use last 20% as validation
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_size = int(0.2 * n)
    val_idx = perm[:val_size]
    tr_idx = perm[val_size:]

    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    # Center features and targets for better conditioning
    X_mean = X_tr.mean(axis=0, keepdims=True)
    Y_mean = Y_tr.mean(axis=0, keepdims=True)
    X_tr_c = X_tr - X_mean
    Y_tr_c = Y_tr - Y_mean
    X_val_c = X_val - X_mean

    # Candidate ridge lambdas
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lambda = None
    best_val_acc = -1.0
    best_W = None
    best_b = None

    for lam in lambdas:
        # Closed-form ridge: W = (X^T X + lam I)^-1 X^T Y
        # Use SVD for stability
        U, S, Vt = np.linalg.svd(X_tr_c, full_matrices=False)
        S_inv = S / (S**2 + lam)
        W = Vt.T @ (S_inv[:, None] * (U.T @ Y_tr_c))
        b = Y_mean.flatten() - X_mean.flatten() @ W

        # Evaluate on validation
        logits_val = X_val_c @ W + b
        pred_val = np.argmax(logits_val, axis=1)
        acc = np.mean(pred_val == y[val_idx])
        if acc > best_val_acc:
            best_val_acc = acc
            best_lambda = lam
            best_W = W
            best_b = b

    # Refit on the entire feedback pool with the chosen lambda
    X_mean_full = X.mean(axis=0, keepdims=True)
    Y_mean_full = Y.mean(axis=0, keepdims=True)
    X_c = X - X_mean_full
    Y_c = Y - Y_mean_full

    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    S_inv = S / (S**2 + best_lambda)
    W_full = Vt.T @ (S_inv[:, None] * (U.T @ Y_c))
    b_full = Y_mean_full.flatten() - X_mean_full.flatten() @ W_full

    # Update fc2 weights and bias
    w["fc2.weight"] = W_full.T.astype(np.float32)  # [10, 128]
    w["fc2.bias"] = b_full.astype(np.float32)      # [10]

    # Diagnostics
    logits_full = X_c @ W_full + b_full
    pred_full = np.argmax(logits_full, axis=1)
    acc_full = np.mean(pred_full == y)
    # Cross-entropy
    logits_full -= logits_full.max(axis=1, keepdims=True)
    probs_full = np.exp(logits_full)
    probs_full /= probs_full.sum(axis=1, keepdims=True)
    ce_full = -np.mean(np.log(probs_full[np.arange(n), y] + 1e-12))

    diagnostics = {
        "best_lambda": float(best_lambda),
        "val_accuracy": float(best_val_acc),
        "full_accuracy": float(acc_full),
        "full_cross_entropy": float(ce_full),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
