import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build length-normalized bag-of-words features
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        tokens = x[i]
        tokens = tokens[tokens != 0]  # exclude padding
        if tokens.size > 0:
            counts = np.bincount(tokens, minlength=V)
            X[i] = counts / float(tokens.size)

    # Add bias column
    X_aug = np.hstack([np.ones((N, 1), dtype=np.float32), X])  # [N, V+1]

    # Ridge regression closed-form
    lam = 1.0
    A = X_aug.T @ X_aug + lam * np.eye(V + 1, dtype=np.float32)
    b = X_aug.T @ (y.astype(np.float32) * 2.0 - 1.0)  # map 0->-1, 1->1
    w_cls = np.linalg.solve(A, b)

    bias = float(w_cls[0])
    token_weights = w_cls[1:].astype(np.float32)

    # Scale so that |a| <= 4 for all tokens
    max_abs = float(np.max(np.abs(token_weights)))
    C = max_abs / 4.0 if max_abs > 0 else 1.0
    if C == 0.0:
        C = 1.0

    # Construct embeddings: each token vector has mean 0, std 1, and first coord = a
    for t in range(V):
        a = token_weights[t] / C
        v = np.zeros(32, dtype=np.float32)
        v[0] = a
        v[1] = -a
        v[2] = np.sqrt(max(0.0, 16.0 - a * a))
        v[3] = -v[2]
        w["embed.weight"][t] = v

    # Zero positional embedding
    w["pos.weight"][:] = 0.0

    # Make both transformer blocks identity by zeroing residual projections and FFN outputs
    for blk in ["block0", "block1"]:
        w[f"{blk}.proj.weight"][:] = 0.0
        w[f"{blk}.proj.bias"][:] = 0.0
        w[f"{blk}.ff2.weight"][:] = 0.0
        w[f"{blk}.ff2.bias"][:] = 0.0

    # Final LayerNorm identity (our embeddings already have mean 0, std 1)
    w["norm.weight"][:] = 1.0
    w["norm.bias"][:] = 0.0

    # Head reads first coordinate and produces logits [-score, score]
    w["head.weight"][:] = 0.0
    w["head.weight"][0, 0] = -C
    w["head.weight"][1, 0] = C
    w["head.bias"][0] = -bias
    w["head.bias"][1] = bias

    # Ensure float32 and finite
    for k in w:
        w[k] = w[k].astype(np.float32)
        np.clip(w[k], -1000.0, 1000.0, out=w[k])

    diagnostics = {
        "ridge_lambda": lam,
        "max_abs_token_weight": max_abs,
        "scale_C": float(C),
        "bias": bias,
        "embed_norm_mean": float(np.mean(np.linalg.norm(w["embed.weight"], axis=1))),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
