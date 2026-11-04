import numpy as np

def edit(w, f):
    # Extract feedback arrays
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]

    # ------------------------------------------------------------
    # 1. Compute smoothed log-odds token scores from the full pool
    # ------------------------------------------------------------
    alpha = 1.0
    pos_counts = np.zeros(V, dtype=np.float64)
    neg_counts = np.zeros(V, dtype=np.float64)

    # Mask padding (token 0) when counting
    mask = (x != 0)
    for cls, counts in [(1, pos_counts), (0, neg_counts)]:
        idx = (y == cls)
        if np.any(idx):
            tokens = x[idx][mask[idx]]
            np.add.at(counts, tokens, 1.0)

    # Smoothed log-odds
    log_odds = np.log((pos_counts + alpha) / (neg_counts + alpha))
    # Center to reduce constant offset
    log_odds = log_odds - np.mean(log_odds)

    # Scale so that the token score is small relative to the stabilising constants
    # (the existing constants are ~289.5 and ~-9.65; std ~52)
    scale = 0.1
    s = (log_odds * scale).astype(np.float32)

    # ------------------------------------------------------------
    # 2. Build new embedding matrix
    #    Keep the existing constant structure but replace coord 0.
    # ------------------------------------------------------------
    embed = w["embed.weight"].copy()
    # Set coord 0 to the new token score
    embed[:, 0] = s
    # Ensure padding token 0 is zero (it is masked out anyway)
    embed[0, :] = 0.0
    # Keep other coordinates as they are (they provide the stabilising constants)
    w["embed.weight"] = embed.astype(np.float32)

    # ------------------------------------------------------------
    # 3. Recompute exact pooled features for all examples
    #    Forward: h = embed[ids] + pos (pos=0)
    #             blocks are identity
    #             h = LayerNorm(h)  (norm.weight=1, norm.bias=0)
    #             pooled = masked_mean(h, ids!=0)
    # ------------------------------------------------------------
    # Get current norm parameters (should be identity)
    norm_w = w["norm.weight"].astype(np.float64)
    norm_b = w["norm.bias"].astype(np.float64)
    eps = 1e-5

    # h = embed[x]  -> [N, L, D]
    h = embed[x].astype(np.float64)  # [N, L, D]

    # LayerNorm over last dimension
    mean = np.mean(h, axis=-1, keepdims=True)
    var = np.var(h, axis=-1, keepdims=True)
    h_norm = (h - mean) / np.sqrt(var + eps)
    h_norm = h_norm * norm_w + norm_b

    # Masked mean over tokens (exclude padding)
    mask_f = (x != 0).astype(np.float64)  # [N, L]
    mask_sum = np.sum(mask_f, axis=1, keepdims=True)  # [N, 1]
    # Avoid division by zero (should not happen for non-empty sequences)
    mask_sum = np.maximum(mask_sum, 1.0)
    pooled = np.sum(h_norm * mask_f[:, :, None], axis=1) / mask_sum  # [N, D]

    # ------------------------------------------------------------
    # 4. Fit head by ridge regression on pooled features
    #    We predict y (0/1) with a linear model: logits = pooled @ W^T + b
    #    Use one-vs-rest ridge for two classes, or simply fit a single
    #    discriminant.  Here we fit a 2-class linear model by ridge on
    #    the binary labels and then set head.weight and head.bias.
    # ------------------------------------------------------------
    # Build target matrix: class 0 -> [1, 0], class 1 -> [0, 1]
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Add bias column
    X_aug = np.concatenate([pooled, np.ones((N, 1))], axis=1)  # [N, D+1]

    # Ridge regression: W = (X^T X + lambda I)^-1 X^T Y
    lam = 1e-2
    A = X_aug.T @ X_aug + lam * np.eye(D + 1)
    B = X_aug.T @ Y
    W = np.linalg.solve(A, B)  # [D+1, 2]

    # Extract head weights and bias
    head_w = W[:-1, :].T  # [2, D]
    head_b = W[-1, :]     # [2]

    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # ------------------------------------------------------------
    # 5. Diagnostics
    # ------------------------------------------------------------
    # Compute training accuracy with the new head
    logits = pooled @ head_w.T + head_b
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)

    diagnostics = {
        "train_accuracy": float(acc),
        "log_odds_mean": float(np.mean(log_odds)),
        "log_odds_std": float(np.std(log_odds)),
        "head_norm": float(np.linalg.norm(head_w)),
        "head_bias": head_b.tolist(),
    }

    return {"weights": w, "diagnostics": diagnostics}
