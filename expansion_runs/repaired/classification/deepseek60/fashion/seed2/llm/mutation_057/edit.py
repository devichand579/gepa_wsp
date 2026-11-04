import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"]          # (2000, 1568)
    y = f["y"]                 # (2000,)
    N, D = X.shape
    C = 10

    # Current head weights and bias
    W = w["head.weight"].copy()   # (10, 1568)
    b = w["head.bias"].copy()     # (10,)

    # Compute current logits (use provided logits if available, but recompute to be safe)
    Z = X @ W.T + b               # (2000, 10)
    pred = Z.argmax(axis=1)
    acc = (pred == y).mean()
    if acc < 1.0:
        # If current model is not perfect on the feedback pool, return unchanged
        return w

    # Compute class means of features
    mu = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        mask = (y == c)
        if mask.sum() > 0:
            mu[c] = X[mask].mean(axis=0)

    # For each example, find the most confused incorrect class (highest logit among wrong classes)
    # We will build a target direction that increases correct class logit and decreases the confused class logit.
    # This is a closed-form least-squares fit to a margin target.
    # Target: for each example i, we want delta_logit[y_i] = +1, delta_logit[j*] = -1, others 0.
    # We solve for delta_W, delta_b that minimize ||X @ delta_W.T + delta_b - T||^2 + lambda * ||delta_W||^2
    # where T is the target matrix (N, C).
    T = np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        c = y[i]
        # find incorrect class with max logit
        wrong_logits = Z[i].copy()
        wrong_logits[c] = -np.inf
        j = wrong_logits.argmax()
        T[i, c] = 1.0
        T[i, j] = -1.0

    # Solve ridge regression: [delta_W, delta_b] = argmin ||X_aug @ V - T||^2 + lambda * ||V||^2
    # Augment X with a column of ones for bias
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])  # (N, D+1)
    lambda_reg = 1e-4
    # V shape: (D+1, C)
    # Normal equations: (X_aug^T X_aug + lambda I) V = X_aug^T T
    A = X_aug.T @ X_aug + lambda_reg * np.eye(D+1, dtype=np.float32)
    B = X_aug.T @ T
    V = np.linalg.solve(A, B)  # (D+1, C)

    delta_W = V[:D, :].T       # (C, D)
    delta_b = V[D, :]          # (C,)

    # Scale the correction so that the new head weights stay within [-1000, 1000]
    # We can apply the correction with a scaling factor alpha. Choose alpha to maximize margin
    # while keeping max abs weight <= 1000. Since current max abs is already 1000, we first scale down
    # the original weights slightly to make room, then add the correction.
    # Alternatively, we can just add a scaled correction and clip.
    # To be safe, we will blend: W_new = beta * W + alpha * delta_W, b_new = beta * b + alpha * delta_b
    # Choose beta = 0.5 to make room, then choose alpha such that max abs <= 1000.
    beta = 0.5
    W_scaled = beta * W
    b_scaled = beta * b

    # Compute max abs of delta_W
    max_delta = np.max(np.abs(delta_W))
    if max_delta > 0:
        # We want max abs of W_scaled + alpha * delta_W <= 1000
        # So alpha <= (1000 - max_abs_W_scaled) / max_delta
        max_abs_W_scaled = np.max(np.abs(W_scaled))
        alpha_max = (1000.0 - max_abs_W_scaled) / max_delta
        alpha = min(alpha_max, 100.0)  # cap at 100 to avoid extreme values
    else:
        alpha = 0.0

    W_new = W_scaled + alpha * delta_W
    b_new = b_scaled + alpha * delta_b

    # Clip to [-1000, 1000] just in case
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    # Verify that the new head still achieves 100% accuracy on the feedback pool
    Z_new = X @ W_new.T + b_new
    pred_new = Z_new.argmax(axis=1)
    acc_new = (pred_new == y).mean()

    if acc_new == 1.0:
        # Compute new cross-entropy to ensure improvement
        # Softmax
        Z_new_max = Z_new.max(axis=1, keepdims=True)
        exp_Z = np.exp(Z_new - Z_new_max)
        probs_new = exp_Z / exp_Z.sum(axis=1, keepdims=True)
        ce_new = -np.log(probs_new[np.arange(N), y] + 1e-12).mean()

        # Current CE
        Z_max = Z.max(axis=1, keepdims=True)
        exp_Z_old = np.exp(Z - Z_max)
        probs_old = exp_Z_old / exp_Z_old.sum(axis=1, keepdims=True)
        ce_old = -np.log(probs_old[np.arange(N), y] + 1e-12).mean()

        if ce_new < ce_old:
            w["head.weight"] = W_new
            w["head.bias"] = b_new
            diagnostics = {
                "old_ce": float(ce_old),
                "new_ce": float(ce_new),
                "old_acc": float(acc),
                "new_acc": float(acc_new),
                "alpha": float(alpha),
                "beta": float(beta),
                "max_abs_weight": float(np.max(np.abs(W_new))),
                "max_abs_bias": float(np.max(np.abs(b_new)))
            }
            return {"weights": w, "diagnostics": diagnostics}

    # Fallback: return original weights unchanged
    return w
