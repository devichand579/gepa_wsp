import numpy as np

def edit(w, f):
    x = np.asarray(f["x"])            # [N,64] int64
    y = np.asarray(f["y"]).astype(np.int64)  # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]    # 2000
    D = w["embed.weight"].shape[1]    # 32

    # ---------- 1. Build normalized bag-of-words count matrix ----------
    # non-pad mask (pad id = 0)
    mask = (x != 0).astype(np.float32)          # [N,64]
    n_tok = np.maximum(mask.sum(axis=1), 1.0)   # [N]
    C = np.zeros((N, V), dtype=np.float32)
    rows = np.repeat(np.arange(N), L)
    cols = x.reshape(-1)
    valid = (cols != 0)
    np.add.at(C, (rows[valid], cols[valid]), 1.0)
    Cn = C / n_tok[:, None]                      # row-normalized counts

    # ---------- 2. Supervised ridge classifier on counts ----------
    # centered targets: +1 for positive (label 1), -1 for negative (label 0)
    t = np.where(y == 1, 1.0, -1.0).astype(np.float64)   # [N]
    # We fit a single weight vector wv (V,) so that Cn @ wv ~ t.
    # Ridge: wv = (Cn^T Cn + lam I)^{-1} Cn^T t
    G = Cn.T @ Cn                                # [V,V]
    lam = 1.0
    G = G + lam * np.eye(V, dtype=np.float64)
    rhs = Cn.T @ t                               # [V]
    wv = np.linalg.solve(G, rhs)                 # [V]

    # ---------- 3. Factor into embed (V x D) and head (2 x D) ----------
    # Effective logits ~ Cn @ wv.  We want head @ pooled with pooled ~ Cn @ embed.
    # Use a 2-column signal: embed_signal[:,0] = wv, embed_signal[:,1] = -wv
    # so head = [[1,0,...],[0,1,...]] reproduces wv.
    sig = np.zeros((V, D), dtype=np.float64)
    sig[:, 0] = wv
    sig[:, 1] = -wv

    # Make each row zero-mean and unit-std so LN(embed) ~ embed (identity).
    # Use two free dims (2 and 3) to absorb mean/std constraints.
    a = np.zeros(V, dtype=np.float64)
    b = np.zeros(V, dtype=np.float64)
    s0 = sig[:, 0]
    s1 = sig[:, 1]
    S = -(s0 + s1)                               # a+b must equal S
    T = float(D) - (s0 * s0 + s1 * s1)           # a^2+b^2 must equal T
    disc = np.maximum(2.0 * T - S * S, 0.0)
    root = np.sqrt(disc)
    a = 0.5 * (S + root)
    b = 0.5 * (S - root)
    sig[:, 2] = a
    sig[:, 3] = b
    # now each row of sig has mean 0 and sum-of-squares D -> std 1
    embed = sig.astype(np.float32)

    # ---------- 4. Write weights ----------
    w["embed.weight"] = embed
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    # final LayerNorm: identity-ish (w=1, b=0) since rows are zero-mean/unit-std
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # ---------- 5. Compute ACTUAL pooled features with real LayerNorm ----------
    # h = embed[x] + pos ; pos=0 ; LN per token ; masked mean
    h = embed[x]                                  # [N,64,D]
    mu = h.mean(axis=2, keepdims=True)
    var = h.var(axis=2, keepdims=True)
    hnorm = (h - mu) / np.sqrt(var + 1e-5)
    hnorm = hnorm * w["norm.weight"].reshape(1, 1, D) + w["norm.bias"].reshape(1, 1, D)
    m = mask[:, :, None]                          # [N,64,1]
    pooled = (hnorm * m).sum(axis=1) / n_tok[:, None]   # [N,D]

    # ---------- 6. Closed-form ridge fit of head on actual pooled features ----------
    P = pooled.astype(np.float64)                 # [N,D]
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    Yc = Y - Y.mean(axis=0, keepdims=True)        # center targets
    Pc = P - P.mean(axis=0, keepdims=True)        # center features
    lam_h = 1.0
    A = Pc.T @ Pc + lam_h * np.eye(D, dtype=np.float64)
    B = Pc.T @ Yc
    Whead = np.linalg.solve(A, B).T               # [2,D]
    bhead = (Y.mean(axis=0) - Whead @ P.mean(axis=0)).astype(np.float64)

    w["head.weight"] = Whead.astype(np.float32)
    w["head.bias"] = bhead.astype(np.float32)

    # ---------- diagnostics ----------
    logits = P @ Whead.T + bhead
    pred = logits.argmax(axis=1)
    train_acc = float((pred == y).mean())
    diag = {
        "train_accuracy": round(train_acc, 4),
        "ridge_lam_counts": lam,
        "ridge_lam_head": lam_h,
        "embed_row_mean_absmax": float(np.abs(embed.mean(axis=1)).max()),
        "embed_row_std_mean": float(embed.std(axis=1).mean()),
        "head_weight_norm": float(np.linalg.norm(Whead)),
        "head_bias": [round(float(v), 4) for v in bhead],
        "wv_norm": float(np.linalg.norm(wv)),
        "pooled_std_mean": float(P.std(axis=0).mean()),
    }
    return {"weights": w, "diagnostics": diag}
