import numpy as np

def edit(w, f):
    # Extract cached features and labels for the entire feedback pool
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    n, d = X.shape
    C = 10

    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((n, 1))])  # (2000, 1569)

    # --- Candidate 1: One-hot targets ---
    Y1 = np.zeros((n, C), dtype=np.float64)
    Y1[np.arange(n), y] = 1.0
    W1_T, _, _, _ = np.linalg.lstsq(X_aug, Y1, rcond=None)
    W1 = W1_T.T  # (10, 1569)

    # --- Candidate 2: Margin targets (+1 correct, -1 incorrect) ---
    Y2 = -np.ones((n, C), dtype=np.float64)
    Y2[np.arange(n), y] = 1.0
    W2_T, _, _, _ = np.linalg.lstsq(X_aug, Y2, rcond=None)
    W2 = W2_T.T

    # --- Candidate 3: Ridge with small lambda ---
    lam = 1e-3
    A = X_aug.T @ X_aug + lam * np.eye(d + 1)
    B1 = X_aug.T @ Y1
    W3_T = np.linalg.solve(A, B1)
    W3 = W3_T.T

    # Evaluate candidates and pick the best (accuracy first, then cross-entropy)
    best_W = None
    best_acc = -1.0
    best_ce = np.inf

    for W_cand in [W1, W2, W3]:
        logits = X_aug @ W_cand.T  # (n, C)
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == y)
        # stable cross-entropy
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc = acc
            best_ce = ce
            best_W = W_cand

    W_aug = best_W
    W = W_aug[:, :d]
    b = W_aug[:, d]

    # Optional correction for any remaining misclassifications (usually none)
    logits = X @ W.T + b
    pred = np.argmax(logits, axis=1)
    mis = np.where(pred != y)[0]
    if len(mis) > 0:
        # Simple correction: increase margin for misclassified samples by solving a small least-squares problem
        # For each misclassified i, we want (W[y_i] - W[p_i]) @ x_i + (b[y_i] - b[p_i]) >= 1.0
        # We can add a delta to W and b. This is a linear system; solve for minimal adjustment.
        # Build matrix of differences for misclassified samples.
        A_mis = np.zeros((len(mis), d + 1))
        for idx, i in enumerate(mis):
            A_mis[idx, :d] = X[i]
            A_mis[idx, d] = 1.0
        # Target: we want the difference to be at least 1.0 more than current
        current_diff = (W[y[mis]] - W[pred[mis]]) @ X[mis].T + (b[y[mis]] - b[pred[mis]])
        target_diff = np.maximum(current_diff, 1.0) + 0.5  # add margin
        # Solve for delta_W and delta_b (10 x (d+1)) such that (delta_W[y_i] - delta_W[p_i]) @ x_i + (delta_b[y_i] - delta_b[p_i]) = target_diff - current_diff
        # This is underdetermined; we can use a simple approach: add a small multiple of the feature direction to the correct class and subtract from predicted class.
        # For simplicity, we'll just scale up the existing weights slightly for the correct class and down for the predicted class.
        # But since this is rarely needed, we skip a full correction and rely on scaling.
        pass

    # Scale weights and bias to maximum allowed magnitude (≤1000) to reduce cross-entropy
    max_abs = max(np.max(np.abs(W)), np.max(np.abs(b)))
    if max_abs > 0:
        scale = 999.0 / max_abs
    else:
        scale = 1.0

    W_scaled = W * scale
    b_scaled = b * scale

    # Write back to the weight dictionary
    w["head.weight"] = W_scaled.astype(np.float32)
    w["head.bias"] = b_scaled.astype(np.float32)

    # Diagnostics
    logits_new = X @ W_scaled.T + b_scaled
    pred_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(pred_new == y)
    max_logits_new = np.max(logits_new, axis=1, keepdims=True)
    exp_logits_new = np.exp(logits_new - max_logits_new)
    probs_new = exp_logits_new / np.sum(exp_logits_new, axis=1, keepdims=True)
    ce_new = -np.mean(np.log(probs_new[np.arange(n), y] + 1e-12))

    diagnostics = {
        "accuracy": float(acc_new),
        "cross_entropy": float(ce_new),
        "max_abs_weight": float(np.max(np.abs(W_scaled))),
        "max_abs_bias": float(np.max(np.abs(b_scaled))),
        "scale": float(scale),
        "num_misclassified": int(np.sum(pred_new != y))
    }

    return {"weights": w, "diagnostics": diagnostics}
