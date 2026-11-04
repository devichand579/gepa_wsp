import numpy as np

def edit(w, f):
    # Extract training data
    X_ids = f["x"]          # (4000, 64) int64
    y = f["y"].astype(np.float32)  # (4000,)
    N, L = X_ids.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build normalized bag-of-words features (mean over non-pad tokens)
    # We'll use counts / length. Length = number of non-zero tokens (pad=0).
    mask = (X_ids != 0)
    lengths = mask.sum(axis=1).astype(np.float32)
    lengths[lengths == 0] = 1.0  # avoid division by zero
    X_bow = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        # Count tokens
        counts = np.bincount(X_ids[i][mask[i]], minlength=V)
        X_bow[i] = counts / lengths[i]

    # Center and scale features for ridge regression
    X_mean = X_bow.mean(axis=0, keepdims=True)
    X_std = X_bow.std(axis=0, keepdims=True)
    X_std[X_std < 1e-6] = 1.0
    X_norm = (X_bow - X_mean) / X_std

    # Ridge regression closed form: solve (X^T X + lambda I) W = X^T y
    # y in {0,1}; we'll fit two classes via one-hot targets.
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y.astype(np.int64)] = 1.0
    # Center Y
    Y_mean = Y.mean(axis=0, keepdims=True)
    Yc = Y - Y_mean

    lam = 1.0
    A = X_norm.T @ X_norm + lam * np.eye(V, dtype=np.float32)
    B = X_norm.T @ Yc
    W_cls = np.linalg.solve(A, B)  # (V, 2)
    b_cls = Y_mean.flatten() - X_mean.flatten() @ W_cls  # (2,)

    # Now we want to map this linear classifier into the network.
    # Architecture after zeroing blocks: h = embed(ids) + pos; h = norm(h);
    # pooled = mean over tokens of h; logits = head(pooled).
    # We set pos.weight = 0, block proj/ff2 = 0, norm.weight=1, norm.bias=0.
    # To make LayerNorm approximately linear, we set embed.weight = [W_cls^T, C]
    # where C is a large constant. Then LayerNorm becomes affine.
    # We'll use 32 dims: first 2 dims are the classifier directions, remaining 30 are constant C.
    C = 100.0
    embed = np.zeros((V, 32), dtype=np.float32)
    embed[:, 0] = W_cls[:, 0]
    embed[:, 1] = W_cls[:, 1]
    embed[:, 2:] = C
    w["embed.weight"] = embed

    # Zero position embeddings
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # Zero all block parameters that affect the residual path
    for blk in ["block0", "block1"]:
        w[f"{blk}.proj.weight"] = np.zeros_like(w[f"{blk}.proj.weight"])
        w[f"{blk}.proj.bias"] = np.zeros_like(w[f"{blk}.proj.bias"])
        w[f"{blk}.ff2.weight"] = np.zeros_like(w[f"{blk}.ff2.weight"])
        w[f"{blk}.ff2.bias"] = np.zeros_like(w[f"{blk}.ff2.bias"])
        # Also zero qkv and ff1 to avoid any numerical noise
        w[f"{blk}.qkv.weight"] = np.zeros_like(w[f"{blk}.qkv.weight"])
        w[f"{blk}.qkv.bias"] = np.zeros_like(w[f"{blk}.qkv.bias"])
        w[f"{blk}.ff1.weight"] = np.zeros_like(w[f"{blk}.ff1.weight"])
        w[f"{blk}.ff1.bias"] = np.zeros_like(w[f"{blk}.ff1.bias"])
        w[f"{blk}.norm1.weight"] = np.ones_like(w[f"{blk}.norm1.weight"])
        w[f"{blk}.norm1.bias"] = np.zeros_like(w[f"{blk}.norm1.bias"])
        w[f"{blk}.norm2.weight"] = np.ones_like(w[f"{blk}.norm2.weight"])
        w[f"{blk}.norm2.bias"] = np.zeros_like(w[f"{blk}.norm2.bias"])

    # Final norm: keep as identity (weight=1, bias=0)
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # Now compute the actual pooled features after LayerNorm with the new embed.
    # We need to simulate the forward pass for the training data to fit the head.
    # Since blocks are zeroed, h = embed(ids) + pos = embed(ids).
    # Then h = LayerNorm(h) with weight=1, bias=0.
    # pooled = mean over non-pad tokens of h.
    # We can compute this efficiently.
    # First, get the embedding for each token: embed[ids] -> (N, L, 32)
    # But we only need the mean over tokens. We can compute the mean of embed[ids] over tokens,
    # then apply LayerNorm? No, LayerNorm is applied per token before pooling.
    # So we must compute LayerNorm per token, then average.
    # Let's compute the pooled features for all N examples.
    # We can do this in a vectorized way.
    # embed is (V, 32). For each example i, we have tokens ids[i] (L,). 
    # We compute E = embed[ids[i]] -> (L, 32). Then LayerNorm across dim=1.
    # Then mean over valid tokens.
    # We'll loop over N (4000) which is fine.
    pooled = np.zeros((N, 32), dtype=np.float32)
    eps = 1e-5
    for i in range(N):
        valid = mask[i]
        if not np.any(valid):
            continue
        tokens = X_ids[i][valid]
        E = embed[tokens]  # (L_valid, 32)
        mean = E.mean(axis=1, keepdims=True)
        var = E.var(axis=1, keepdims=True)
        E_norm = (E - mean) / np.sqrt(var + eps)
        pooled[i] = E_norm.mean(axis=0)

    # Now fit a linear head on these pooled features to predict y.
    # We can use ridge regression again, or just least squares.
    # Add a bias term by centering.
    P_mean = pooled.mean(axis=0, keepdims=True)
    P_std = pooled.std(axis=0, keepdims=True)
    P_std[P_std < 1e-6] = 1.0
    P_norm = (pooled - P_mean) / P_std

    # Fit head: logits = head.weight @ pooled + head.bias
    # We'll fit on P_norm to get better conditioning, then transform back.
    # Actually, we can just fit directly on pooled.
    # Use ridge regression with small lambda.
    lam_head = 1e-3
    A_head = pooled.T @ pooled + lam_head * np.eye(32, dtype=np.float32)
    B_head = pooled.T @ Yc
    W_head = np.linalg.solve(A_head, B_head)  # (32, 2)
    b_head = Y_mean.flatten() - P_mean.flatten() @ W_head  # (2,)

    # Set head weights: head.weight is (2, 32), so transpose W_head
    w["head.weight"] = W_head.T.astype(np.float32)
    w["head.bias"] = b_head.astype(np.float32)

    # Diagnostics
    # Compute training accuracy
    logits = pooled @ W_head + b_head
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    diagnostics = {
        "method": "linear BOW + large constant LayerNorm trick",
        "train_accuracy": float(acc),
        "embed_norm_mean": float(np.linalg.norm(embed, axis=1).mean()),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
    }

    return {"weights": w, "diagnostics": diagnostics}
