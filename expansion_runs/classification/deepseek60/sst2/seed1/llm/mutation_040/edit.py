import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)      # [N,64]
    y = np.asarray(f["y"], dtype=np.int64)      # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]              # vocab
    D = w["embed.weight"].shape[1]              # 32
    mask = (X != 0)

    # ---- Build per-token features G[t] (D-dim, zero-mean, unit-variance) ----
    # Use class-conditional token statistics to get discriminative directions.
    # counts per class
    c1 = np.zeros(V, dtype=np.float64)
    c0 = np.zeros(V, dtype=np.float64)
    n1 = np.zeros(V, dtype=np.float64)
    n0 = np.zeros(V, dtype=np.float64)
    for i in range(N):
        ids = X[i][mask[i]]
        if y[i] == 1:
            np.add.at(c1, ids, 1.0)
            n1[ids] += 1.0
        else:
            np.add.at(c0, ids, 1.0)
            n0[ids] += 1.0
    # smoothed log-odds per token
    eps = 1.0
    p1 = (c1 + eps) / (n1 + 2 * eps)
    p0 = (c0 + eps) / (n0 + 2 * eps)
    score = np.log(p1 / p0)                     # vocab-dim sentiment direction
    freq = c0 + c1

    # Build D basis vocab-vectors g_k. First few are data-driven, rest random.
    rng = np.random.RandomState(12345)
    G = np.zeros((V, D), dtype=np.float64)      # G[t,k] = g_k[t]
    # direction 0: sentiment score (weighted by frequency to focus on seen tokens)
    dirs = []
    dirs.append(score * np.sqrt(freq + 1.0))
    # direction 1: frequency itself
    dirs.append(np.log1p(freq))
    # direction 2: positive-only mass
    dirs.append(np.log1p(c1))
    # direction 3: negative-only mass
    dirs.append(np.log1p(c0))
    while len(dirs) < D:
        dirs.append(rng.randn(V))
    for k in range(D):
        g = dirs[k].astype(np.float64)
        g = g - g.mean()
        nrm = np.linalg.norm(g)
        if nrm < 1e-8:
            g = rng.randn(V)
            g = g - g.mean()
            nrm = np.linalg.norm(g)
        G[:, k] = g / nrm

    # Now enforce per-token rows to be zero-mean and unit-variance.
    # We orthogonalize the columns so that rows become zero-mean & ~unit-var.
    # Center columns (already ~0 mean), then whiten rows via iterative approach:
    # Simpler: make columns orthonormal in the "row" sense is hard; instead
    # directly normalize each row to zero-mean unit-variance, accepting that
    # columns get slightly correlated (fine for a linear head fit).
    row_mean = G.mean(axis=1, keepdims=True)
    G = G - row_mean
    row_var = (G ** 2).mean(axis=1, keepdims=True)
    row_var[row_var < 1e-12] = 1.0
    G = G / np.sqrt(row_var)

    # ---- Set embed.weight = G (so LN(embed[t]) == G[t]) ----
    w["embed.weight"] = G.astype(np.float32)
    # keep pos zero, blocks zero, norm identity
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    for name in ["block0.qkv.weight","block0.qkv.bias","block0.proj.weight","block0.proj.bias",
                 "block0.ff1.weight","block0.ff1.bias","block0.ff2.weight","block0.ff2.bias",
                 "block1.qkv.weight","block1.qkv.bias","block1.proj.weight","block1.proj.bias",
                 "block1.ff1.weight","block1.ff1.bias","block1.ff2.weight","block1.ff2.bias"]:
        w[name] = np.zeros_like(w[name])
    for name in ["block0.norm1.weight","block0.norm2.weight","block1.norm1.weight","block1.norm2.weight"]:
        w[name] = np.ones_like(w[name])
    for name in ["block0.norm1.bias","block0.norm2.bias","block1.norm1.bias","block1.norm2.bias"]:
        w[name] = np.zeros_like(w[name])
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # ---- Compute pooled features exactly (LN is fixed point) ----
    # pooled_i = mean over non-pad of G[token]
    pooled = np.zeros((N, D), dtype=np.float64)
    for i in range(N):
        ids = X[i][mask[i]]
        if ids.size == 0:
            continue
        pooled[i] = G[ids].mean(axis=0)

    # ---- Closed-form ridge fit of head ----
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    # center features for bias handling
    mu = pooled.mean(axis=0, keepdims=True)
    Pc = pooled - mu
    lam = 1.0
    A = Pc.T @ Pc + lam * np.eye(D)
    B = Pc.T @ (Y - Y.mean(axis=0, keepdims=True))
    Wc = np.linalg.solve(A, B)                  # D x 2
    head_w = Wc.T                                # 2 x D
    head_b = Y.mean(axis=0) - mu @ Wc            # 2
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # ---- diagnostics ----
    logits = pooled @ head_w.T + head_b
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    diag = {
        "train_acc": acc,
        "pooled_dim_nonzero": int((pooled.std(axis=0) > 1e-6).sum()),
        "head_norm": float(np.linalg.norm(head_w)),
        "n_tokens_used": int(mask.sum()),
    }
    return {"weights": w, "diagnostics": diag}
