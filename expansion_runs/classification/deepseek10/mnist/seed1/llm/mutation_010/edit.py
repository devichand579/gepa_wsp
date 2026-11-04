import numpy as np

def edit(w, f):
    # Extract full-pool features and labels
    F = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = F.shape
    C = 10

    # Class means and global mean
    mu_c = np.zeros((C, D), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        counts[c] = np.sum(mask)
        if counts[c] > 0:
            mu_c[c] = np.mean(F[mask], axis=0)
    mu = np.mean(F, axis=0)

    # Within-class scatter matrix
    S_w = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        if counts[c] > 0:
            Xc = F[mask] - mu_c[c]
            S_w += Xc.T @ Xc

    # Regularize: add small ridge proportional to mean variance
    mean_var = np.trace(S_w) / D
    lam = 1e-3 * mean_var
    S_w_reg = S_w + lam * np.eye(D, dtype=np.float64)

    # Solve for LDA weights: W = S_w_reg^{-1} @ (mu_c - mu).T
    # Using solve for stability
    W = np.linalg.solve(S_w_reg, (mu_c - mu).T)   # (D, C)

    # Bias: -0.5 * (mu_c + mu)^T @ W + log(prior)
    prior = counts / N
    b = -0.5 * np.sum((mu_c + mu) * W.T, axis=1) + np.log(prior + 1e-12)

    new_fc2_weight = W.T          # (C, D)
    new_fc2_bias = b              # (C,)

    # Scale to match current logits standard deviation
    std_cur = np.std(f["logits"])
    Z = F @ new_fc2_weight.T + new_fc2_bias
    std_new = np.std(Z)
    if std_new > 1e-12:
        alpha = std_cur / std_new
        new_fc2_weight *= alpha
        new_fc2_bias *= alpha
    else:
        alpha = 1.0

    # Clip to safe bounds
    new_fc2_weight = np.clip(new_fc2_weight, -1000, 1000)
    new_fc2_bias = np.clip(new_fc2_bias, -1000, 1000)

    # Update weights
    w["fc2.weight"] = new_fc2_weight.astype(np.float32)
    w["fc2.bias"] = new_fc2_bias.astype(np.float32)

    # Diagnostics: training accuracy and cross-entropy on the full pool
    logits_new = F @ new_fc2_weight.T + new_fc2_bias
    logits_new -= np.max(logits_new, axis=1, keepdims=True)
    exp_logits = np.exp(logits_new)
    probs_new = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    preds = np.argmax(probs_new, axis=1)
    acc = np.mean(preds == y)
    ce = -np.mean(np.log(probs_new[np.arange(N), y] + 1e-12))

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "fc2_weight_norm": float(np.linalg.norm(new_fc2_weight)),
        "fc2_bias_norm": float(np.linalg.norm(new_fc2_bias)),
        "std_cur": float(std_cur),
        "std_new": float(std_new),
        "alpha": float(alpha),
        "S_w_trace": float(np.trace(S_w)),
        "lambda": float(lam),
    }

    return {"weights": w, "diagnostics": diagnostics}
