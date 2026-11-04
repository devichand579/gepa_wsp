import numpy as np

def edit(w, f):
    # Extract training data
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = x.shape
    vocab_size = w["embed.weight"].shape[0]  # 2000

    # Build bag-of-words count matrix C [N, vocab_size]
    C = np.zeros((N, vocab_size), dtype=np.float32)
    for i in range(N):
        for t in x[i]:
            if t > 0:  # ignore padding (0)
                C[i, t] += 1.0

    # One-hot labels
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Ridge regression: solve (C^T C + lambda I) W = C^T Y
    # Use a small lambda for numerical stability
    lambda_reg = 1e-3
    CtC = C.T @ C
    CtC += lambda_reg * np.eye(vocab_size, dtype=np.float32)
    CtY = C.T @ Y
    # Solve for W [vocab_size, 2]
    W = np.linalg.solve(CtC, CtY)  # [2000, 2]

    # We want logits = W_head @ (1/N * E^T c) + b
    # We can set b = 0 and W_head = W^T (2x2000) but we need to factorise.
    # Actually, the network computes pooled = (1/N) * E^T c.
    # We want W_head @ pooled = W^T c (up to positive scaling).
    # So we need W_head @ E^T = W^T.
    # W^T is [2, 2000]. Its rank is at most 2.
    # We can factorise W^T = W_head @ E^T with W_head [2, 32] and E [2000, 32].
    # Use SVD of W^T: W^T = U S V^T, where U is [2,2], S is [2,2], V is [2000,2].
    # Then set W_head = U S, E = V. But we need E to be [2000, 32].
    # We can pad E with zeros to 32 columns, and pad W_head with zeros to 32 columns.
    # However, the network's embedding dimension is 32. We can put the 2 components in the first 2 dimensions.
    U, S, Vt = np.linalg.svd(W.T, full_matrices=False)  # W.T is [2, 2000]
    # U: [2, 2], S: [2], Vt: [2, 2000]
    # W.T = U @ diag(S) @ Vt
    # We want W_head @ E^T = W.T
    # Let W_head = U @ diag(S)  (2x2), E^T = Vt (2x2000) -> E = Vt.T (2000x2)
    W_head_2 = U @ np.diag(S)  # [2, 2]
    E_2 = Vt.T                 # [2000, 2]

    # Now we need to set the network weights.
    # 1. Embedding: set first 2 columns to E_2, rest to 0.
    embed = np.zeros((vocab_size, 32), dtype=np.float32)
    embed[:, :2] = E_2
    w["embed.weight"] = embed

    # 2. Positional embedding: set to zero (or keep as is? We want identity, so zero is fine)
    w["pos.weight"] = np.zeros((64, 32), dtype=np.float32)

    # 3. All transformer blocks: set to identity.
    # For each block, we need norm1, qkv, proj, norm2, ff1, ff2 to be identity.
    # To make block identity: we want output = input.
    # The block does: h = h + proj(attention(norm1(h)))
    #                 h = h + ff2(relu(ff1(norm2(h))))
    # If we set proj and ff2 to zero, then h remains unchanged.
    # So set qkv.weight=0, qkv.bias=0, proj.weight=0, proj.bias=0,
    # ff1.weight=0, ff1.bias=0, ff2.weight=0, ff2.bias=0.
    # Also set norm1.weight=1, norm1.bias=0, norm2.weight=1, norm2.bias=0.
    for block in ["block0", "block1"]:
        w[f"{block}.norm1.weight"] = np.ones(32, dtype=np.float32)
        w[f"{block}.norm1.bias"] = np.zeros(32, dtype=np.float32)
        w[f"{block}.qkv.weight"] = np.zeros((96, 32), dtype=np.float32)
        w[f"{block}.qkv.bias"] = np.zeros(96, dtype=np.float32)
        w[f"{block}.proj.weight"] = np.zeros((32, 32), dtype=np.float32)
        w[f"{block}.proj.bias"] = np.zeros(32, dtype=np.float32)
        w[f"{block}.norm2.weight"] = np.ones(32, dtype=np.float32)
        w[f"{block}.norm2.bias"] = np.zeros(32, dtype=np.float32)
        w[f"{block}.ff1.weight"] = np.zeros((64, 32), dtype=np.float32)
        w[f"{block}.ff1.bias"] = np.zeros(64, dtype=np.float32)
        w[f"{block}.ff2.weight"] = np.zeros((32, 64), dtype=np.float32)
        w[f"{block}.ff2.bias"] = np.zeros(32, dtype=np.float32)

    # 4. Final norm: identity
    w["norm.weight"] = np.ones(32, dtype=np.float32)
    w["norm.bias"] = np.zeros(32, dtype=np.float32)

    # 5. Head: set to W_head_2 padded to 32 columns, bias = 0
    head_weight = np.zeros((2, 32), dtype=np.float32)
    head_weight[:, :2] = W_head_2
    w["head.weight"] = head_weight
    w["head.bias"] = np.zeros(2, dtype=np.float32)

    # Diagnostics
    diagnostics = {
        "lambda_reg": lambda_reg,
        "W_norm": float(np.linalg.norm(W)),
        "W_head_2": W_head_2.tolist(),
        "E_2_mean": float(np.mean(E_2)),
        "E_2_std": float(np.std(E_2)),
    }

    return {"weights": w, "diagnostics": diagnostics}
