import numpy as np

def edit(w, f):
    rng = np.random.RandomState(0)
    X = f["x"].astype(np.int64)          # [N,64]
    Y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    D = 32
    mask = (X != 0).astype(np.float32)   # [N,64]
    L = np.maximum(mask.sum(1), 1.0)     # [N]

    # ---------- 1. Linear backbone: ridge on length-normalized BOW ----------
    V = int(w["embed.weight"].shape[0])
    counts = np.zeros((N, V), dtype=np.float64)
    for t in range(T):
        np.add.at(counts, (np.arange(N), X[:, t]), 1.0)
    A = counts / L[:, None]              # normalized BOW
    Ys = 2.0 * Y.astype(np.float64) - 1.0   # {-1,+1}

    best = None
    for lam in (0.3, 1.0, 3.0, 10.0):
        G = A.T @ A + lam * np.eye(V)
        rhs = A.T @ Ys
        wlin = np.linalg.solve(G, rhs)
        pred = np.sign(A @ wlin)
        acc = float((pred == Ys).mean())
        if best is None or acc > best[0]:
            best = (acc, wlin, lam)
    lin_acc, wlin, lam = best

    # ---------- 2. Encode wlin into embed rows so that d·LN(embed[v]) = s·wlin[v] ----------
    # LN output is zero-mean, unit-variance over 32 dims (norm sqrt(32)).
    # Realize z_v = sqrt(32)*(cos th * u + sin th * v) in a fixed 2-D orthonormal frame.
    u = np.zeros(D); u[0] = 1.0
    v2 = np.zeros(D); v2[1] = 1.0
    R = 0.80
    scale = np.sqrt(D) * R
    s = float(np.abs(wlin).max())
    s = s if s > 1e-9 else 1.0
    theta = np.arccos(np.clip(wlin / (scale * s), -1.0, 1.0))
    Z = (np.sqrt(D) * (np.cos(theta)[:, None] * u[None, :]
                       + np.sin(theta)[:, None] * v2[None, :]))   # [V,32]
    # invert LayerNorm: embed[v] = gamma * Z[v] + beta  (gamma=1,beta=0) -> embed = Z
    embed = Z.astype(np.float32)

    # head direction: d = head[1]-head[0] should be R*s*u  (so d·z_v = s*wlin[v])
    dvec = (R * s) * u
    head = np.zeros((2, D), dtype=np.float32)
    head[1] = 0.5 * dvec
    head[0] = -0.5 * dvec
    bias = np.zeros(2, dtype=np.float32)

    # ---------- 3. Document-context attention in block0 ----------
    # z = LN(embed) = Z (already zero-mean/unit-var). Uniform attention: q=k=0, v=z.
    # attn_i = mean_j z_j = zbar ; h_i = embed[token_i] + P zbar.
    # We search a few P scales and re-fit the head on the REALIZED pooled features.
    def realized_pooled(P):
        # zbar per example = mean_t Z[token_t] over non-pad
        Zt = Z[X]                                   # [N,64,32]
        m = mask[:, :, None]
        zbar = (Zt * m).sum(1) / L[:, None]         # [N,32]
        ctx = zbar @ P.T                            # [N,32]
        h = Zt + ctx[:, None, :]                    # [N,64,32]
        # final LayerNorm (weight 1, bias 0) per token
        mu = h.mean(2, keepdims=True)
        var = h.var(2, keepdims=True)
        hn = (h - mu) / np.sqrt(var + 1e-5)
        pooled = (hn * m).sum(1) / L[:, None]       # [N,32]
        return pooled

    bestP = None
    for pscale in (0.0, 0.05, 0.1, 0.2, 0.35):
        if pscale == 0.0:
            P = np.zeros((D, D))
        else:
            P = pscale * np.eye(D)
        pooled = realized_pooled(P)
        # closed-form ridge head on realized features
        G = pooled.T @ pooled + 1e-2 * np.eye(D)
        rhs = pooled.T @ Ys
        hw = np.linalg.solve(G, rhs)
        pred = np.sign(pooled @ hw)
        acc = float((pred == Ys).mean())
        if bestP is None or acc > bestP[0]:
            bestP = (acc, P, hw, pooled)

    accP, P, hw, pooled = bestP
    # use the better of {linear-only, context} by realized training accuracy
    if accP >= lin_acc:
        # set head from realized ridge (hw is [32] = head[1]-head[0] direction)
        head = np.zeros((2, D), dtype=np.float32)
        head[1] = 0.5 * hw.astype(np.float32)
        head[0] = -0.5 * hw.astype(np.float32)
        bias = np.zeros(2, dtype=np.float32)
        use_ctx = True
    else:
        P = np.zeros((D, D))
        use_ctx = False

    # ---------- 4. Write weights ----------
    w["embed.weight"] = embed.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # block0: norm1 identity, qkv q=k=0, v=identity, proj = P
    w["block0.norm1.weight"] = np.ones(D, dtype=np.float32)
    w["block0.norm1.bias"] = np.zeros(D, dtype=np.float32)
    qkv = np.zeros((96, D), dtype=np.float32)
    qkv[64:96, :] = np.eye(D, dtype=np.float32)      # v = z
    w["block0.qkv.weight"] = qkv
    w["block0.qkv.bias"] = np.zeros(96, dtype=np.float32)
    w["block0.proj.weight"] = P.astype(np.float32)
    w["block0.proj.bias"] = np.zeros(D, dtype=np.float32)
    w["block0.norm2.weight"] = np.ones(D, dtype=np.float32)
    w["block0.norm2.bias"] = np.zeros(D, dtype=np.float32)
    w["block0.ff1.weight"] = np.zeros_like(w["block0.ff1.weight"], dtype=np.float32)
    w["block0.ff1.bias"] = np.zeros_like(w["block0.ff1.bias"], dtype=np.float32)
    w["block0.ff2.weight"] = np.zeros_like(w["block0.ff2.weight"], dtype=np.float32)
    w["block0.ff2.bias"] = np.zeros_like(w["block0.ff2.bias"], dtype=np.float32)

    # block1 identity
    for nm in ["block1.norm1.weight", "block1.norm2.weight"]:
        w[nm] = np.ones(D, dtype=np.float32)
    for nm in ["block1.norm1.bias", "block1.norm2.bias"]:
        w[nm] = np.zeros(D, dtype=np.float32)
    for nm in ["block1.qkv.weight", "block1.proj.weight", "block1.ff1.weight", "block1.ff2.weight"]:
        w[nm] = np.zeros_like(w[nm], dtype=np.float32)
    for nm in ["block1.qkv.bias", "block1.proj.bias", "block1.ff1.bias", "block1.ff2.bias"]:
        w[nm] = np.zeros_like(w[nm], dtype=np.float32)

    w["norm.weight"] = np.ones(D, dtype=np.float32)
    w["norm.bias"] = np.zeros(D, dtype=np.float32)
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = bias.astype(np.float32)

    # clip / sanitize
    for k in w:
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[k] = arr.astype(np.float32)

    diag = {
        "lin_acc": float(lin_acc),
        "lam": float(lam),
        "ctx_acc": float(accP),
        "use_ctx": bool(use_ctx),
        "pscale": float(np.abs(P).max()),
        "wlin_absmax": float(np.abs(wlin).max()),
        "head_absmax": float(np.abs(head).max()),
    }
    return {"weights": w, "diagnostics": diag}
