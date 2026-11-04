import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- 1. Build length-normalised bag-of-words design matrix ----
    mask = (X != 0).astype(np.float64)   # [N,64]
    lens = mask.sum(axis=1)
    lens[lens == 0] = 1.0
    # counts per token, normalised by length
    F = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        m = mask[i]
        for t in range(T):
            if m[t] > 0:
                F[i, row[t]] += 1.0
    F /= lens[:, None]

    # ---- 2. Closed-form ridge regression for token scores ----
    yv = np.where(y == 1, 1.0, -1.0)
    # centre features (bias handled separately)
    Fm = F.mean(axis=0, keepdims=True)
    Fc = F - Fm
    yc = yv - yv.mean()
    lam = 1.0
    A = Fc.T @ Fc + lam * np.eye(V)
    b_vec = Fc.T @ yc
    try:
        w_tok = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        w_tok = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    b_tok = yv.mean() - Fm.ravel() @ w_tok

    # ---- 3. Scale token scores so they fit LayerNorm budget ----
    # We will place s'(v) in dim 1 and fillers in dims 2,3.
    # Constraint: s'^2 + u^2 + w^2 = 32 with u+w = -s'  =>  s'^2 <= 64/3
    max_abs = np.max(np.abs(w_tok))
    if max_abs < 1e-12:
        max_abs = 1.0
    budget = np.sqrt(64.0 / 3.0) * 0.95   # safe margin
    scale = budget / max_abs
    s = w_tok * scale
    b_scaled = b_tok * scale

    # ---- 4. Build embed rows satisfying zero-mean & unit-variance ----
    embed = np.zeros((V, D), dtype=np.float64)
    for v in range(V):
        sv = s[v]
        disc = 64.0 - 3.0 * sv * sv
        if disc < 0:
            disc = 0.0
        r = np.sqrt(disc)
        u = (-sv + r) / 2.0
        ww = (-sv - r) / 2.0
        embed[v, 0] = 0.0
        embed[v, 1] = sv
        embed[v, 2] = u
        embed[v, 3] = ww
        # remaining dims zero; mean = (0 + sv + u + ww)/32 = 0, norm^2 = sv^2+u^2+ww^2 = 32

    # ---- 5. Write weights ----
    w["embed.weight"] = embed.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # zero out both transformer blocks (keep identity behaviour)
    for blk in ["block0", "block1"]:
        for name in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = blk + "." + name
            if key in w:
                w[key] = np.zeros_like(w[key])
        for name in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
            key = blk + "." + name
            if key in w:
                if "weight" in name:
                    w[key] = np.ones_like(w[key])
                else:
                    w[key] = np.zeros_like(w[key])

    # final norm identity
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # head: logit_1 - logit_0 = mean_t s'(v_t) + b_scaled
    head = np.zeros((2, D), dtype=np.float64)
    head[1, 1] = 1.0
    w["head.weight"] = head.astype(np.float32)
    bias = np.zeros(2, dtype=np.float64)
    bias[1] = b_scaled
    w["head.bias"] = bias.astype(np.float32)

    # ---- 6. Diagnostics ----
    # recompute predictions on the pool using the new model
    pooled = np.zeros((N, D), dtype=np.float64)
    for i in range(N):
        m = mask[i]
        cnt = m.sum()
        if cnt == 0:
            continue
        acc = np.zeros(D, dtype=np.float64)
        for t in range(T):
            if m[t] > 0:
                acc += embed[X[i, t]]
        pooled[i] = acc / cnt
    logits = pooled @ head.T + bias
    preds = np.argmax(logits, axis=1)
    acc = float((preds == y).mean())

    diag = {
        "train_acc_bow": acc,
        "scale": float(scale),
        "b_scaled": float(b_scaled),
        "max_abs_score": float(max_abs),
        "embed_row_norm_mean": float(np.mean(np.linalg.norm(embed, axis=1))),
        "head_norm": float(np.linalg.norm(head)),
    }
    return {"weights": w, "diagnostics": diag}
