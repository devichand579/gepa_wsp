import numpy as np

def edit(w, f):
    # Read current fc2 parameters
    W_old = w["fc2.weight"].astype(np.float64)  # [10, 128]
    b_old = w["fc2.bias"].astype(np.float64)    # [10]

    # Feedback arrays
    H = f["features"].astype(np.float64)        # [N, 128]
    L = f["logits"].astype(np.float64)          # [N, 10]
    y = f["y"].astype(np.int64)                 # [N]
    N = H.shape[0]
    C = L.shape[1]

    # Current predictions
    pred = np.argmax(L, axis=1)

    # Build target residual R: desired change in logits.
    # For correct examples, R = 0.
    # For errors, we want to increase true-class logit and decrease predicted-class logit.
    # Use a small target margin change, e.g., delta = 1.0.
    delta = 1.0
    R = np.zeros_like(L)
    err_mask = pred != y
    if np.any(err_mask):
        idx = np.where(err_mask)[0]
        R[idx, y[idx]] += delta
        R[idx, pred[idx]] -= delta

    # Augment features with bias column
    X = np.concatenate([H, np.ones((N, 1))], axis=1)  # [N, 129]

    # Ridge regression for correction dW_aug: minimize ||X dW_aug^T - R||^2 + lambda ||dW_aug||^2
    # We choose a small lambda to regularize. Since N=2000, features=129, lambda=1.0 is mild.
    lam = 1.0
    # Solve (X^T X + lam I) dW_aug^T = X^T R
    XtX = X.T @ X
    XtR = X.T @ R
    dW_aug = np.linalg.solve(XtX + lam * np.eye(X.shape[1]), XtR).T  # [10, 129]

    dW = dW_aug[:, :128]
    db = dW_aug[:, 128]

    # Apply correction
    W_new = W_old + dW
    b_new = b_old + db

    # Clip to safe range
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    w["fc2.weight"] = W_new
    w["fc2.bias"] = b_new

    # Diagnostics
    # Compute new logits and metrics on the full feedback pool
    L_new = H @ W_new.T + b_new
    pred_new = np.argmax(L_new, axis=1)
    acc_new = np.mean(pred_new == y)
    # Cross-entropy
    log_probs = L_new - np.max(L_new, axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs /= np.sum(probs, axis=1, keepdims=True)
    ce_new = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "delta": delta,
        "lambda": lam,
        "num_errors_before": int(np.sum(err_mask)),
        "num_errors_after": int(np.sum(pred_new != y)),
        "accuracy_before": float(np.mean(pred == y)),
        "accuracy_after": float(acc_new),
        "ce_before": float(-np.mean(np.log(f["probs"][np.arange(N), y] + 1e-12))),
        "ce_after": float(ce_new),
        "dW_norm": float(np.linalg.norm(dW)),
        "db_norm": float(np.linalg.norm(db)),
    }

    return {"weights": w, "diagnostics": diagnostics}
