import numpy as np

def edit(w, f):
    # Extract current head parameters and feedback arrays
    W = w["head.weight"].copy()  # (10, 1568)
    b = w["head.bias"].copy()    # (10,)
    F = f["features"]            # (2000, 1568)
    y = f["y"]                   # (2000,)
    logits = f["logits"]         # (2000, 10)
    N, D = F.shape
    C = 10

    # Find misclassified samples
    preds = np.argmax(logits, axis=1)
    mis = np.where(preds != y)[0]
    if len(mis) == 0:
        return w  # already perfect

    # We will attempt to fix the first misclassified sample
    i = mis[0]
    y_i = y[i]
    p_i = preds[i]
    margin = logits[i, p_i] - logits[i, y_i]
    if margin <= 0:
        return w  # should not happen

    # Compute direction d that minimizes disturbance on other samples
    # Solve: minimize ||F_other d||^2 subject to F_i d = 1
    # A = F^T F - F_i^T F_i
    A = F.T @ F - np.outer(F[i], F[i])
    # Add small ridge for numerical stability
    lam = 1e-6
    A_reg = A + lam * np.eye(D, dtype=np.float32)
    try:
        d = np.linalg.solve(A_reg, F[i].astype(np.float64))
    except np.linalg.LinAlgError:
        return w
    d = d.astype(np.float32)
    denom = F[i] @ d
    if abs(denom) < 1e-12:
        return w
    d /= denom  # now F_i @ d = 1

    # Compute disturbance s_j = F_j @ d for all j
    s = F @ d  # (2000,)

    # Required alpha to fix sample i: margin - 2*alpha < 0 => alpha > margin/2
    alpha_req = margin / 2.0 + 1e-3

    # Compute maximum alpha that keeps all other samples correct
    # For each j != i, for each c != y_j:
    #   logits[j,y_j] + alpha*delta_y > logits[j,c] + alpha*delta_c
    #   alpha * (delta_y - delta_c) > logits[j,c] - logits[j,y_j]
    # where delta_y = s_j if y_j==y_i else (-s_j if y_j==p_i else 0)
    #       delta_c = s_j if c==y_i else (-s_j if c==p_i else 0)
    delta = np.zeros((N, C), dtype=np.float32)
    delta[:, y_i] = s
    delta[:, p_i] = -s

    # For j != i
    mask_j = np.ones(N, dtype=bool)
    mask_j[i] = False
    j_idx = np.where(mask_j)[0]
    y_j = y[j_idx]  # (N-1,)
    delta_y = delta[j_idx, y_j]  # (N-1,)
    # denom = delta_y[:, None] - delta[j_idx]  # (N-1, C)
    denom = delta_y[:, None] - delta[j_idx]
    # logit_diff = logits[j_idx, c] - logits[j_idx, y_j]
    logit_diff = logits[j_idx] - logits[j_idx, y_j][:, None]  # (N-1, C)

    # We only care about c != y_j and denom < 0
    c_mask = np.arange(C)[None, :] != y_j[:, None]  # (N-1, C)
    # For denom < 0, bound = logit_diff / denom (both negative? logit_diff is usually negative, denom negative -> positive)
    # Actually we need alpha < logit_diff / denom. Since denom < 0, and logit_diff < 0 (for c != y_j, logit_diff is negative),
    # the ratio is positive. We want the minimum positive ratio.
    valid = c_mask & (denom < -1e-12)
    if np.any(valid):
        bounds = logit_diff[valid] / denom[valid]
        # bounds should be positive. Take min of positive bounds.
        pos_bounds = bounds[bounds > 0]
        if len(pos_bounds) > 0:
            alpha_max = np.min(pos_bounds)
        else:
            alpha_max = np.inf
    else:
        alpha_max = np.inf

    # Choose alpha
    if alpha_max == np.inf:
        alpha = alpha_req * 1.1
    else:
        alpha = min(alpha_req * 1.1, alpha_max * 0.99)
        if alpha < alpha_req:
            # Cannot fix without breaking others; try a smaller step to at least improve CE? 
            # But we want accuracy, so return original.
            return w

    # Apply update: add alpha*d to row y_i, subtract alpha*d from row p_i
    W_new = W.copy()
    W_new[y_i] += alpha * d
    W_new[p_i] -= alpha * d
    # Bias unchanged (or we could adjust bias, but not needed)

    # Clip to allowed range
    W_new = np.clip(W_new, -1000.0, 1000.0)
    b_new = b.copy()
    b_new = np.clip(b_new, -1000.0, 1000.0)

    # Evaluate new full-pool accuracy and cross-entropy
    new_logits = F @ W_new.T + b_new
    new_preds = np.argmax(new_logits, axis=1)
    new_acc = np.mean(new_preds == y)
    # Cross-entropy
    max_logits = np.max(new_logits, axis=1, keepdims=True)
    exp_logits = np.exp(new_logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    # Clip for numerical stability
    probs = np.clip(probs, 1e-15, 1.0)
    new_ce = -np.mean(np.log(probs[np.arange(N), y]))

    # Current metrics
    cur_preds = np.argmax(logits, axis=1)
    cur_acc = np.mean(cur_preds == y)
    max_logits_cur = np.max(logits, axis=1, keepdims=True)
    exp_logits_cur = np.exp(logits - max_logits_cur)
    probs_cur = exp_logits_cur / np.sum(exp_logits_cur, axis=1, keepdims=True)
    probs_cur = np.clip(probs_cur, 1e-15, 1.0)
    cur_ce = -np.mean(np.log(probs_cur[np.arange(N), y]))

    # Only accept if accuracy improves or (accuracy same and CE improves)
    if new_acc > cur_acc or (new_acc == cur_acc and new_ce < cur_ce):
        w["head.weight"] = W_new.astype(np.float32)
        w["head.bias"] = b_new.astype(np.float32)
        diagnostics = {
            "misclassified_index": int(i),
            "margin": float(margin),
            "alpha_required": float(alpha_req),
            "alpha_max": float(alpha_max) if alpha_max != np.inf else None,
            "chosen_alpha": float(alpha),
            "new_accuracy": float(new_acc),
            "new_cross_entropy": float(new_ce),
            "old_accuracy": float(cur_acc),
            "old_cross_entropy": float(cur_ce),
            "max_abs_head_weight": float(np.max(np.abs(W_new))),
            "max_abs_head_bias": float(np.max(np.abs(b_new)))
        }
        return {"weights": w, "diagnostics": diagnostics}
    else:
        return w
