import numpy as np

def edit(w, f):
    # ---------------------------------------------------------------
    # 1. Collapse the transformer blocks to identity.
    #    Zero the residual projections and the second FF projections
    #    so that h = h + 0 + 0 = h through both blocks.
    # ---------------------------------------------------------------
    for b in ("block0", "block1"):
        w[f"{b}.proj.weight"][:] = 0.0
        w[f"{b}.proj.bias"][:]   = 0.0
        w[f"{b}.ff2.weight"][:]  = 0.0
        w[f"{b}.ff2.bias"][:]    = 0.0
        # Also zero qkv and ff1 to avoid any accidental contribution
        w[f"{b}.qkv.weight"][:]  = 0.0
        w[f"{b}.qkv.bias"][:]    = 0.0
        w[f"{b}.ff1.weight"][:]  = 0.0
        w[f"{b}.ff1.bias"][:]    = 0.0

    # Zero positional embedding so the representation is position independent
    w["pos.weight"][:] = 0.0

    # Make the final LayerNorm an identity (weight=1, bias=0)
    w["norm.weight"][:] = 1.0
    w["norm.bias"][:]   = 0.0

    # ---------------------------------------------------------------
    # 2. Build a bag-of-words feature matrix from the feedback pool.
    #    f["x"] is [N,64] int64 token ids; f["y"] is [N] labels.
    # ---------------------------------------------------------------
    X_ids = f["x"]                     # [N,64]
    y     = f["y"].astype(np.float32)  # [N]
    N     = X_ids.shape[0]
    V     = w["embed.weight"].shape[0] # 2000

    # One-hot bag-of-words counts (excluding padding id 0)
    # Use a dense matrix for simplicity; 4000 x 2000 is manageable.
    counts = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        ids = X_ids[i]
        # ignore padding (0)
        valid = ids[ids != 0]
        if valid.size:
            np.add.at(counts[i], valid, 1.0)

    # Normalize by sequence length to mimic masked mean
    lengths = np.maximum((X_ids != 0).sum(axis=1), 1).astype(np.float32)
    counts /= lengths[:, None]

    # ---------------------------------------------------------------
    # 3. Fit a linear classifier on the bag-of-words features.
    #    Ridge regression to targets t = +1 for class 1, -1 for class 0.
    # ---------------------------------------------------------------
    t = 2.0 * y - 1.0                  # [N]
    # Center features
    mu = counts.mean(axis=0)           # [V]
    Xc = counts - mu                   # [N,V]
    # Standardize per dimension (with small epsilon)
    sd = Xc.std(axis=0) + 1e-6
    Xs = Xc / sd                       # [N,V]

    # Ridge normal equations: w = (Xs^T Xs + λI)^{-1} Xs^T t
    # Use the dual form to avoid inverting a 2000x2000 matrix.
    lam = 1.0
    # K = Xs Xs^T  [N,N]
    K = Xs @ Xs.T
    # alpha = (K + λI)^{-1} t
    alpha = np.linalg.solve(K + lam * np.eye(N, dtype=np.float32), t)
    # w = Xs^T alpha  [V]
    w_bow = Xs.T @ alpha               # [V]

    # Bias in the standardized space
    b_bow = -np.dot(mu / sd, w_bow)

    # ---------------------------------------------------------------
    # 4. Embed the bag-of-words weights into the 32-dim embedding space.
    #    We need pooled = mean_t embed[t] to be a linear function of
    #    the counts.  The simplest exact construction is to set
    #    embed[t] = w_bow[t] * e0  (first coordinate), and then the
    #    head reads that coordinate.  However, LayerNorm is applied
    #    per token before pooling, which would destroy the magnitude.
    #    To avoid that, we pre-normalize each embedding row so that
    #    LayerNorm acts as identity (zero mean, unit variance).
    #    We can still encode the bag-of-words weights in the first
    #    coordinate by using a two-step construction:
    #    - set embed[t] = [w_bow[t], 0, ..., 0]
    #    - then subtract the row mean and divide by row std so that
    #      LayerNorm(embed[t]) = embed[t].
    #    This makes the pooled feature a linear function of the counts
    #    (up to the normalization constant per row, which we correct
    #    by fitting the head on the actual computed features).
    # ---------------------------------------------------------------
    E = np.zeros((V, 32), dtype=np.float32)
    E[:, 0] = w_bow
    # Normalize each row to zero mean and unit variance
    row_mean = E.mean(axis=1, keepdims=True)
    row_std  = E.std(axis=1, keepdims=True) + 1e-6
    E = (E - row_mean) / row_std
    w["embed.weight"][:] = E

    # ---------------------------------------------------------------
    # 5. Recompute the actual pooled features for the whole pool using
    #    the new embedding and the identity transformer.
    #    pooled = mean over valid tokens of LayerNorm(embed[token])
    #    Since we pre-normalized E, LayerNorm(E[t]) = E[t] (up to eps).
    # ---------------------------------------------------------------
    # Compute pooled features directly from E and the token ids.
    pooled = np.zeros((N, 32), dtype=np.float32)
    for i in range(N):
        ids = X_ids[i]
        valid = ids[ids != 0]
        if valid.size:
            pooled[i] = E[valid].mean(axis=0)
        else:
            pooled[i] = 0.0

    # ---------------------------------------------------------------
    # 6. Fit the head on the actual pooled features.
    #    This is a ridge regression from pooled -> one-hot targets.
    # ---------------------------------------------------------------
    # Center and standardize pooled features
    mu_p = pooled.mean(axis=0)
    Xp = pooled - mu_p
    sd_p = Xp.std(axis=0) + 1e-6
    Xp_s = Xp / sd_p

    # Targets: one-hot for two classes
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y.astype(np.int64)] = 1.0

    lam2 = 1.0
    # Solve for each output dimension: W = (Xp_s^T Xp_s + λI)^{-1} Xp_s^T Y
    # Xp_s is [N,32], so Xp_s^T Xp_s is 32x32.
    A = Xp_s.T @ Xp_s + lam2 * np.eye(32, dtype=np.float32)
    B = Xp_s.T @ Y                     # [32,2]
    W_head = np.linalg.solve(A, B)     # [32,2]

    # Convert back to original feature space:
    # logits = (pooled - mu_p)/sd_p @ W_head + b
    #        = pooled @ (W_head / sd_p) + (b - mu_p/sd_p @ W_head)
    head_weight = (W_head / sd_p[:, None]).T   # [2,32]
    head_bias   = - (mu_p / sd_p) @ W_head     # [2]

    w["head.weight"][:] = head_weight.astype(np.float32)
    w["head.bias"][:]   = head_bias.astype(np.float32)

    # ---------------------------------------------------------------
    # 7. Diagnostics
    # ---------------------------------------------------------------
    # Compute training accuracy with the new head on the pooled features
    logits = pooled @ head_weight.T + head_bias
    preds = np.argmax(logits, axis=1)
    acc = float((preds == y).mean())
    ce = float(-np.mean(np.log(np.exp(logits[np.arange(N), y.astype(np.int64)]) /
                               np.exp(logits).sum(axis=1) + 1e-12) + 1e-12))

    diag = {
        "train_accuracy": acc,
        "train_cross_entropy": ce,
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias": head_bias.tolist(),
        "embed_norm_mean": float(np.linalg.norm(E, axis=1).mean()),
        "bow_weight_norm": float(np.linalg.norm(w_bow)),
    }

    return {"weights": w, "diagnostics": diag}
