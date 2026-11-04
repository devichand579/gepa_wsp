import numpy as np

def edit(w, f):
    # Read full training pool
    X = f["x"].astype(np.int64)          # [4000, 64]
    y = f["y"].astype(np.int64)          # [4000]
    F = f["features"].astype(np.float32) # [4000, 32] current final hidden features
    N, D = F.shape

    # One-hot targets for least-squares / ridge
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    def fit_head(feats, alpha=1e-3):
        # feats: [N, D]
        # Add bias column
        A = np.concatenate([feats, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, D+1]
        # Ridge normal equations: (A^T A + alpha I) W = A^T Y
        AtA = A.T @ A
        AtA_reg = AtA + alpha * np.eye(D + 1, dtype=np.float32)
        AtY = A.T @ Y
        W = np.linalg.solve(AtA_reg, AtY)  # [D+1, 2]
        return W[:-1].T.astype(np.float32), W[-1].astype(np.float32)  # [2, D], [2]

    # Candidate 1: fit head on current features
    head_w1, head_b1 = fit_head(F, alpha=1e-3)
    logits1 = F @ head_w1.T + head_b1
    acc1 = np.mean(np.argmax(logits1, axis=1) == y)

    # Candidate 2: neutralize blocks, use current embed+pos, final norm, then fit head
    # We will compute features for this modified network on the full pool.
    # To do this, we need to run the forward pass with modified weights.
    # We'll create a copy of w and zero out block weights, set norms to identity.
    w2 = {k: v.copy() for k, v in w.items()}
    for blk in ["block0", "block1"]:
        for sub in ["qkv", "proj", "ff1", "ff2"]:
            w2[f"{blk}.{sub}.weight"][...] = 0.0
            w2[f"{blk}.{sub}.bias"][...] = 0.0
        for sub in ["norm1", "norm2"]:
            w2[f"{blk}.{sub}.weight"][...] = 1.0
            w2[f"{blk}.{sub}.bias"][...] = 0.0
    # final norm stays as is (weight=1, bias=0)
    # pos.weight: we can keep or zero. Let's keep it for now, or zero it.
    # We'll compute features for all 4000 using w2.

    def forward_features(w_dict, ids):
        # ids: [N, 64] int64
        N, L = ids.shape
        embed = w_dict["embed.weight"]  # [2000, 32]
        pos = w_dict["pos.weight"]      # [64, 32]
        h = embed[ids] + pos[None, :, :]  # [N, L, 32]
        # blocks are neutralized, so h unchanged
        # final norm
        gamma = w_dict["norm.weight"]   # [32]
        beta = w_dict["norm.bias"]      # [32]
        eps = 1e-5
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        h_norm = (h - mean) / np.sqrt(var + eps) * gamma + beta
        # masked mean
        mask = (ids != 0).astype(np.float32)  # [N, L]
        mask_sum = mask.sum(axis=1, keepdims=True) + 1e-9
        pooled = (h_norm * mask[..., None]).sum(axis=1) / mask_sum
        return pooled.astype(np.float32)  # [N, 32]

    F2 = forward_features(w2, X)
    head_w2, head_b2 = fit_head(F2, alpha=1e-3)
    logits2 = F2 @ head_w2.T + head_b2
    acc2 = np.mean(np.argmax(logits2, axis=1) == y)

    # Candidate 3: same as candidate 2 but also zero pos.weight
    w3 = {k: v.copy() for k, v in w2.items()}
    w3["pos.weight"][...] = 0.0
    F3 = forward_features(w3, X)
    head_w3, head_b3 = fit_head(F3, alpha=1e-3)
    logits3 = F3 @ head_w3.T + head_b3
    acc3 = np.mean(np.argmax(logits3, axis=1) == y)

    # Choose best candidate
    candidates = [
        (acc1, head_w1, head_b1, None),
        (acc2, head_w2, head_b2, w2),
        (acc3, head_w3, head_b3, w3),
    ]
    best_acc, best_head_w, best_head_b, best_w = max(candidates, key=lambda x: x[0])

    if best_w is not None:
        # Apply the modified weights
        for k in w:
            if k in best_w:
                w[k][...] = best_w[k]
    # Set head
    w["head.weight"][...] = best_head_w
    w["head.bias"][...] = best_head_b

    # Diagnostics
    diag = {
        "candidate_accuracies": {
            "current_features": float(acc1),
            "neutral_blocks_keep_pos": float(acc2),
            "neutral_blocks_zero_pos": float(acc3),
        },
        "chosen": int(np.argmax([acc1, acc2, acc3])),
        "head_weight_norm": float(np.linalg.norm(best_head_w)),
        "head_bias": best_head_b.tolist(),
    }

    return {"weights": w, "diagnostics": diag}
