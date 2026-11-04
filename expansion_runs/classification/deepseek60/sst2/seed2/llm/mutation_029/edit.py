import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D_dim = w["embed.weight"].shape[1]   # 32

    mask = (X != 0)
    L = mask.sum(axis=1).astype(np.float64)
    L[L == 0] = 1.0

    # ---- Build length-normalized bag-of-words features ----
    C = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    valid = (cols != 0)
    np.add.at(C, (rows[valid], cols[valid]), 1.0)
    Phi = C / L[:, None]                 # [N,V]

    # ---- Ridge fit on +/-1 targets (closed form, bias unpenalized) ----
    t = np.where(y == 1, 1.0, -1.0)
    A = np.concatenate([Phi, np.ones((N, 1))], axis=1)   # [N,V+1]
    G = A.T @ A
    lam = 1.0
    reg = np.eye(V + 1) * lam
    reg[-1, -1] = 0.0
    G = G + reg
    rhs = A.T @ t
    sol = np.linalg.solve(G, rhs)
    w_lin = sol[:V]                      # token weights
    b_lin = sol[V]                       # intercept

    # ---- Encode token weights into embed rows via LayerNorm ----
    # LayerNorm(e) = f  with f zero-mean, unit-std.  We want d.f = g_t.
    # Choose d = D * e_0, so g_t = D * f_t[0]  =>  f_t[0] = g_t / D.
    # Remaining 31 entries must have sum = -f_t[0] and sumsq = 32 - f_t[0]^2.
    g = w_lin.copy()
    gmax = np.max(np.abs(g))
    if gmax < 1e-12:
        gmax = 1.0
    # |f_t[0]| <= sqrt(31)  =>  D >= gmax/sqrt(31)
    D = gmax / np.sqrt(31.0) * 1.05
    if D < 1e-6:
        D = 1e-6
    a = g / D                            # f_t[0] targets, |a| <= sqrt(31)/1.05

    # Build the 31 remaining entries: 30 equal alpha, 1 = beta
    #   30*alpha + beta = S = -a
    #   30*alpha^2 + beta^2 = Q = 32 - a^2
    S = -a
    Q = 32.0 - a * a
    # 930 alpha^2 - 60 S alpha + (S^2 - Q) = 0
    disc = 3720.0 * Q - 120.0 * S * S
    disc = np.maximum(disc, 0.0)
    sq = np.sqrt(disc)
    alpha = (60.0 * S + sq) / 1860.0
    beta = S - 30.0 * alpha

    F = np.zeros((V, D_dim), dtype=np.float64)
    F[:, 0] = a
    F[:, 1:31] = alpha[:, None]
    F[:, 31] = beta

    # embed row e_t = s * f_t + m * ones  -> LayerNorm(e_t) = f_t for any s>0
    s = 1.0
    E = s * F
    E = E - E.mean(axis=1, keepdims=True)   # ensure zero-mean (numerical safety)
    std = E.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1e-8
    E = E / std

    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # ---- Zero the transformer blocks (keep passthrough) ----
    for name in list(w.keys()):
        if name.startswith("block0.") or name.startswith("block1."):
            if name.endswith(".weight"):
                w[name] = np.zeros_like(w[name])
            else:
                w[name] = np.zeros_like(w[name])
    # Norm weights inside blocks must be 1 (not 0) so LayerNorm is identity-ish;
    # but since qkv/proj/ff are zero, block output == input regardless. Keep them 1.
    for blk in ("block0", "block1"):
        w[blk + ".norm1.weight"] = np.ones_like(w[blk + ".norm1.weight"])
        w[blk + ".norm2.weight"] = np.ones_like(w[blk + ".norm2.weight"])

    # ---- Final norm: identity ----
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # ---- Head: d = head[1]-head[0] = D * e_0 ; bias diff = b_lin ----
    head = np.zeros((2, D_dim), dtype=np.float64)
    head[0, 0] = 0.0
    head[1, 0] = D
    w["head.weight"] = head.astype(np.float32)
    bias = np.zeros(2, dtype=np.float64)
    bias[0] = -b_lin / 2.0
    bias[1] = b_lin / 2.0
    w["head.bias"] = bias.astype(np.float32)

    # ---- Diagnostics ----
    diag = {
        "ridge_lambda": float(lam),
        "w_lin_absmax": float(np.max(np.abs(w_lin))),
        "w_lin_std": float(np.std(w_lin)),
        "b_lin": float(b_lin),
        "D": float(D),
        "f0_absmax": float(np.max(np.abs(a))),
        "embed_row_mean_absmax": float(np.max(np.abs(E.mean(axis=1)))),
        "embed_row_std_maxdev": float(np.max(np.abs(E.std(axis=1) - 1.0))),
    }
    return {"weights": w, "diagnostics": diag}
