import numpy as np

def edit(w, f):
    # Extract arrays
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    logits = f["logits"].astype(np.float64)       # (2000, 10)
    N, D = X.shape
    C = logits.shape[1]

    # Current predictions
    pred = np.argmax(logits, axis=1)
    mis = np.where(pred != y)[0]

    # If no misclassification, just return current weights
    if len(mis) == 0:
        return w

    # We will correct the first misclassified sample
    m = mis[0]
    x_m = X[m]                                    # (D,)
    y_m = y[m]
    p_m = pred[m]

    # Build S = X_other^T X_other + lambda I
    # To save memory/time, use all samples except m
    mask = np.ones(N, dtype=bool)
    mask[m] = False
    X_other = X[mask]                             # (N-1, D)
    # Regularized covariance (ridge)
    lam = 1e-3
    S = X_other.T @ X_other + lam * np.eye(D)

    # Solve S v = x_m
    v = np.linalg.solve(S, x_m)                   # (D,)
    denom = v @ x_m
    if denom <= 0:
        return w
    # c_i = (v @ x_i) / denom
    c = (X @ v) / denom                           # (N,)

    # Current logits for the misclassified sample
    L_m = logits[m]
    # Required alpha to make y_m the argmax
    # delta vector: +1 at y_m, -1 at p_m, 0 elsewhere
    delta = np.zeros(C)
    delta[y_m] = 1.0
    delta[p_m] = -1.0

    # Compute required alpha
    # Condition: L_m[y_m] + alpha > L_m[k] for all k != y_m
    # For k != p_m: alpha > L_m[k] - L_m[y_m]
    # For k = p_m: L_m[y_m] + alpha > L_m[p_m] - alpha  => 2 alpha > L_m[p_m] - L_m[y_m]
    diffs = L_m - L_m[y_m]
    diffs[y_m] = -np.inf
    # For k != p_m, need alpha > diffs[k]
    # For p_m, need alpha > diffs[p_m] / 2
    alpha_req = max(np.max(diffs[np.arange(C) != p_m]), diffs[p_m] / 2.0)
    if alpha_req < 0:
        alpha_req = 0.0

    # Try a few alpha values around alpha_req + margin
    best_acc = -1.0
    best_ce = np.inf
    best_alpha = 0.0
    best_W = None
    best_b = None

    # Current head weights and bias
    W = w["head.weight"].astype(np.float64).copy()   # (10, 1568)
    b = w["head.bias"].astype(np.float64).copy()     # (10,)

    # We will update W and b. The update to logits is:
    # L_new = L + alpha * outer(c, delta)
    # This corresponds to W_new = W + alpha * outer(delta, v) / denom
    # and b_new = b (since c_m = 1, but for other samples c_i varies; bias update would be constant, so we only update W)
    # Actually, L_new[i] = W_new @ x_i + b_new = W @ x_i + b + alpha * c_i * delta
    # We want W_new @ x_i = W @ x_i + alpha * c_i * delta
    # Since c_i = (v @ x_i)/denom, we need W_new = W + alpha * outer(delta, v) / denom
    # Then W_new @ x_i = W @ x_i + alpha * (delta * (v @ x_i)) / denom = W @ x_i + alpha * c_i * delta. Correct.
    # Bias unchanged.

    # Candidate alphas
    for margin in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        alpha = alpha_req + margin
        if alpha <= 0:
            continue
        W_new = W + alpha * np.outer(delta, v) / denom
        # Compute new logits
        L_new = X @ W_new.T + b
        # Evaluate
        pred_new = np.argmax(L_new, axis=1)
        acc = np.mean(pred_new == y)
        # Cross-entropy
        # Softmax
        L_new_shifted = L_new - np.max(L_new, axis=1, keepdims=True)
        exp_L = np.exp(L_new_shifted)
        probs = exp_L / np.sum(exp_L, axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc = acc
            best_ce = ce
            best_alpha = alpha
            best_W = W_new

    if best_W is None:
        return w

    # Check max abs weight constraint
    max_abs = np.max(np.abs(best_W))
    if max_abs > 1000.0:
        best_W = best_W * (1000.0 / max_abs)

    # Update w
    w["head.weight"] = best_W.astype(np.float32)
    # bias unchanged
    # Ensure finite and within bounds
    w["head.weight"] = np.clip(w["head.weight"], -1000.0, 1000.0).astype(np.float32)
    w["head.bias"] = np.clip(w["head.bias"], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "misclassified_index": int(m),
        "alpha_req": float(alpha_req),
        "chosen_alpha": float(best_alpha),
        "full_pool_accuracy": float(best_acc),
        "full_pool_cross_entropy": float(best_ce),
        "max_abs_head_weight": float(np.max(np.abs(w["head.weight"])))
    }
    return {"weights": w, "diagnostics": diagnostics}
