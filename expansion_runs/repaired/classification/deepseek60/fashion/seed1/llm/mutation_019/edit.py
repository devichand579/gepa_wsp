import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)  # [2000, 1568]
    y = f["y"].astype(np.int64)           # [2000]
    n, d = X.shape
    K = 10

    # Build one-hot targets
    Y = np.zeros((n, K), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Center features and targets for a proper affine fit
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Ridge regression in the dual (n < d) for efficiency
    # W = Xc^T (Xc Xc^T + lambda I)^{-1} Yc
    # b = Y_mean - X_mean @ W^T
    # Use a small lambda; features are already informative.
    lam = 1e-4
    G = Xc @ Xc.T                       # [2000, 2000]
    G[np.diag_indices_from(G)] += lam
    alpha = np.linalg.solve(G, Yc)      # [2000, 10]
    W = Xc.T @ alpha                    # [1568, 10]
    W = W.T                             # [10, 1568]
    b = (Y_mean - X_mean @ W.T).ravel() # [10]

    # Optional: scale the logits to reduce cross-entropy on the full pool.
    # Since argmax is invariant to positive scaling, we can choose a scale
    # that minimizes cross-entropy on the full feedback pool.
    logits = X @ W.T + b                # [2000, 10]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    # Cross-entropy as a function of scale s: CE(s) = -mean(log softmax(s * logits))
    # We can find the optimal s by a simple 1D search, but to avoid any
    # iterative search we use a closed-form approximation or just a safe scale.
    # A simple robust choice: scale so that the mean top-1 logit margin is ~log(9).
    # Compute current margins.
    sorted_logits = np.sort(logits, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    mean_margin = margins.mean()
    target_margin = np.log(9.0)
    if mean_margin > 1e-8:
        scale = target_margin / mean_margin
    else:
        scale = 1.0
    # Clamp scale to a reasonable range to avoid extreme values.
    scale = float(np.clip(scale, 0.1, 100.0))
    W *= scale
    b *= scale

    # Write back to the weight dictionary
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    logits_new = X @ W.T + b
    preds = np.argmax(logits_new, axis=1)
    acc = float((preds == y).mean())
    # Cross-entropy
    logits_new -= logits_new.max(axis=1, keepdims=True)
    probs_new = np.exp(logits_new)
    probs_new /= probs_new.sum(axis=1, keepdims=True)
    ce = float(-np.log(probs_new[np.arange(n), y] + 1e-12).mean())

    diagnostics = {
        "lambda": lam,
        "scale": scale,
        "full_accuracy": acc,
        "full_cross_entropy": ce,
        "head_weight_std": float(W.std()),
        "head_bias_mean": float(b.mean()),
        "head_weight_min": float(W.min()),
        "head_weight_max": float(W.max()),
    }
    return {"weights": w, "diagnostics": diagnostics}
