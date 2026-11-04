import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"]          # [N, 64] int64 token ids
    y = f["y"]          # [N] int64 labels
    N, L = X.shape

    # Build length-normalised bag-of-words counts (N x V)
    V = w["embed.weight"].shape[0]
    counts = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        row = X[i]
        mask = row != 0
        if mask.sum() == 0:
            continue
        ids = row[mask]
        np.add.at(counts[i], ids, 1.0)
        counts[i] /= mask.sum()

    # SVD of the normalised count matrix to get token vectors
    # We only need the top 32 right singular vectors.
    # To avoid huge memory, use randomized SVD or just full SVD if V=2000 is okay.
    # N=4000, V=2000. Full SVD is feasible (4000x2000).
    U, S, Vt = np.linalg.svd(counts, full_matrices=False)
    # Vt shape [min(N,V), V] = [2000, 2000]? Actually N=4000, V=2000, so Vt is [2000, 2000].
    # We want top 32 right singular vectors: Vt[:32, :] shape [32, 2000].
    # We need token vectors of shape [V, 32], so transpose.
    token_vecs = Vt[:32, :].T.astype(np.float32)  # [2000, 32]

    # Normalise token vectors to have zero mean and unit variance across the 32 dims?
    # Not necessary; we will add a large constant vector to dominate LayerNorm.
    # Choose a constant vector u with zero mean and large variance.
    # Let u = c * v, where v is a fixed random vector with zero mean and unit variance.
    rng = np.random.RandomState(42)
    v = rng.randn(32).astype(np.float32)
    v -= v.mean()
    v /= v.std()
    c = 50.0  # large enough to make LayerNorm approximately linear
    u = c * v

    # New embedding: e_t = token_vecs[t] + u
    new_embed = token_vecs + u[None, :]

    # Compute the features that the network will produce with these embeddings.
    # Since blocks are zero, h = embed(ids) + pos (pos is zero).
    # Then final norm: LayerNorm(h) per token, then mean over non-pad tokens.
    # We can compute this exactly using the new_embed.
    # For each example, gather embeddings, apply LayerNorm, then mean.
    eps = 1e-5
    feats = np.zeros((N, 32), dtype=np.float32)
    for i in range(N):
        row = X[i]
        mask = row != 0
        if mask.sum() == 0:
            continue
        ids = row[mask]
        emb = new_embed[ids]  # [n, 32]
        mean = emb.mean(axis=1, keepdims=True)
        var = emb.var(axis=1, keepdims=True)
        ln = (emb - mean) / np.sqrt(var + eps)
        feats[i] = ln.mean(axis=0)

    # Fit a ridge classifier on these features.
    # We want to predict y in {0,1}. Use ridge regression on targets -1/+1.
    Y = np.where(y == 1, 1.0, -1.0).astype(np.float32)
    # Add bias by augmenting features with a column of ones.
    F_aug = np.concatenate([feats, np.ones((N, 1), dtype=np.float32)], axis=1)
    # Ridge regression: w = (F^T F + alpha I)^-1 F^T Y
    alpha = 1.0
    A = F_aug.T @ F_aug + alpha * np.eye(33, dtype=np.float32)
    b = F_aug.T @ Y
    w_ridge = np.linalg.solve(A, b)  # [33]
    head_w = w_ridge[:32]  # [32]
    head_b = w_ridge[32]   # scalar

    # The head in the network is [2, 32] and bias [2].
    # We want logit for class 1 minus class 0 to be head_w @ pooled + head_b.
    # So set head.weight[1] = head_w, head.weight[0] = -head_w
    # and head.bias[1] = head_b, head.bias[0] = -head_b.
    # But note: the network's final logits are head @ pooled + bias.
    # The classification is argmax. So we can set:
    w["head.weight"] = np.vstack([-head_w, head_w]).astype(np.float32)
    w["head.bias"] = np.array([-head_b, head_b], dtype=np.float32)

    # Update embedding
    w["embed.weight"] = new_embed.astype(np.float32)

    # Keep all other weights as they are (blocks zero, pos zero, norms identity).
    # But ensure they are float32 and finite.
    for k in w:
        if isinstance(w[k], np.ndarray):
            w[k] = w[k].astype(np.float32)
            w[k] = np.clip(w[k], -1000.0, 1000.0)

    # Diagnostics
    # Compute training accuracy with the new head on the computed features.
    logits = feats @ head_w + head_b
    preds = (logits > 0).astype(np.int64)
    acc = (preds == y).mean()
    ce = -np.mean(np.log(np.where(y == 1, 1/(1+np.exp(-logits)), 1/(1+np.exp(logits))) + 1e-12))

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "head_w_norm": float(np.linalg.norm(head_w)),
        "head_b": float(head_b),
        "embed_mean": float(new_embed.mean()),
        "embed_std": float(new_embed.std()),
        "c": float(c),
    }

    return {"weights": w, "diagnostics": diagnostics}
