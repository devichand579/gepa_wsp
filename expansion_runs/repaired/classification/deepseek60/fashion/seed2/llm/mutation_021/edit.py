import numpy as np

def edit(w, f):
    # Extract feedback arrays
    features = f["features"]          # (2000, 1568)
    y = f["y"]                        # (2000,)
    logits = f["logits"]              # (2000, 10)
    probs = f["probs"]                # (2000, 10)

    # Current predictions and errors
    pred = np.argmax(logits, axis=1)
    errors = np.where(pred != y)[0]
    old_acc = np.mean(pred == y)
    old_ce = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))

    if len(errors) == 0:
        return w

    # We will fix all errors using a closed-form minimal-norm update to head weights.
    # For each error i, we want to change logits by d_i = alpha_i * (e_{y_i} - e_{pred_i})
    # where alpha_i is chosen to flip the margin.
    n_errors = len(errors)
    F_err = features[errors]          # (n_errors, 1568)
    margins = logits[errors, pred[errors]] - logits[errors, y[errors]]  # wrong - correct
    # We want new margin < 0. Set alpha = margin/2 + epsilon. Use a small epsilon.
    alphas = margins / 2.0 + 0.01

    # Build D matrix (10 x n_errors)
    D = np.zeros((10, n_errors), dtype=np.float32)
    for k, idx in enumerate(errors):
        c = y[idx]
        w_class = pred[idx]
        D[c, k] = alphas[k]
        D[w_class, k] = -alphas[k]

    # Solve for delta_W such that delta_W @ F_err.T = D
    # Minimum norm solution: delta_W = D @ F_err @ inv(F_err @ F_err.T)
    G = F_err @ F_err.T
    G += np.eye(n_errors) * 1e-6      # numerical stability
    G_inv = np.linalg.inv(G)
    delta_W = (D @ F_err) @ G_inv     # (10, 1568)

    # Apply update to head weights
    w["head.weight"] = w["head.weight"] + delta_W.astype(np.float32)

    # Evaluate new logits and accuracy
    new_logits = features @ w["head.weight"].T + w["head.bias"]
    new_pred = np.argmax(new_logits, axis=1)
    new_acc = np.mean(new_pred == y)

    # Compute new cross-entropy
    max_logits = np.max(new_logits, axis=1, keepdims=True)
    exp_logits = np.exp(new_logits - max_logits)
    new_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    new_ce = -np.mean(np.log(new_probs[np.arange(len(y)), y] + 1e-12))

    # If new accuracy is worse, revert
    if new_acc < old_acc:
        w["head.weight"] = w["head.weight"] - delta_W.astype(np.float32)
        return w

    # Clamp weights to [-1000, 1000] as required
    w["head.weight"] = np.clip(w["head.weight"], -1000, 1000).astype(np.float32)
    w["head.bias"] = np.clip(w["head.bias"], -1000, 1000).astype(np.float32)

    diagnostics = {
        "old_accuracy": float(old_acc),
        "new_accuracy": float(new_acc),
        "old_ce": float(old_ce),
        "new_ce": float(new_ce),
        "n_errors_fixed": int(len(errors)),
        "max_abs_head_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_head_bias": float(np.max(np.abs(w["head.bias"])))
    }
    return {"weights": w, "diagnostics": diagnostics}
