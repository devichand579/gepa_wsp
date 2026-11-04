import numpy as np
import math

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- Build length-normalized bag-of-words design matrix ----
    # count[v] over tokens, divided by number of non-pad tokens T
    mask = (X != 0).astype(np.float64)                 # [N,64]
    T = np.maximum(mask.sum(axis=1), 1.0)              # [N]
    # one-hot accumulation via bincount per row is slow; use matrix trick
    # Build sparse-ish counts with np.add.at on flattened indices
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    cnt = np.zeros((N, V), dtype=np.float64)
    np.add.at(cnt, (rows, cols), 1.0)
    Xf = cnt / T[:, None]                              # [N,V] mean-count features

    # ---- Targets: +1 / -1 ----
    t = np.where(y == 1, 1.0, -1.0)
    t_mean = t.mean()
    tc = t - t_mean

    # ---- Closed-form ridge, choose alpha by simple holdout CV ----
    # Center features
    Xm = Xf.mean(axis=0, keepdims=True)
    Xc = Xf - Xm

    # Precompute Gram once
    G = Xc.T @ Xc                                      # [V,V]
    b = Xc.T @ tc                                      # [V]

    # CV split (deterministic)
    rng = np.random.RandomState(1234)
    idx = rng.permutation(N)
    n_val = N // 5
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr = Xc[tr_idx]; ttr = tc[tr_idx]
    Xva = Xc[val_idx]; tva = tc[val_idx]
    Gtr = Xtr.T @ Xtr
    btr = Xtr.T @ ttr

    best_alpha = None
    best_acc = -1.0
    for alpha in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        A = Gtr + alpha * np.eye(V)
        try:
            s_tr = np.linalg.solve(A, btr)
        except np.linalg.LinAlgError:
            continue
        # bias
        b0 = ttr.mean() - (Xtr.mean(axis=0) @ s_tr)
        pred = Xva @ s_tr + b0
        acc = ((pred > 0).astype(np.int64) == (tva > 0).astype(np.int64)).mean()
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    if best_alpha is None:
        best_alpha = 1.0

    # ---- Final fit on all data ----
    A = G + best_alpha * np.eye(V)
    s = np.linalg.solve(A, b)                          # [V]
    bias = t_mean - (Xm.ravel() @ s)                   # scalar intercept

    # ---- Encode s(v) into embeddings ----
    # We want: LayerNorm(embed[v]) = u_v  with (head1-head0) @ u_v = s(v)
    # Choose head1-head0 = c * e_0  =>  u_v[0] = s(v)/c
    # u_v must be zero-mean, unit-variance.
    # Build u_v: u_v[0] = s(v)/c ; u_v[1..31] = -s(v)/(31c)*1 + alpha_v*p + beta_v*q
    # where p,q are fixed zero-sum orthonormal vectors in dims 1..31.
    c = 100.0
    s = np.clip(s, -0.95 * c, 0.95 * c)

    # Fixed orthonormal zero-sum basis for dims 1..31
    sub = D - 1                                        # 31
    # p: normalized zero-sum vector in dims 1..31
    p = np.ones(sub)
    p = p - p.mean()
    p = p / (np.linalg.norm(p) + 1e-12)
    # q: another zero-sum vector, orthogonal to p
    q = np.arange(sub).astype(np.float64)
    q = q - q.mean()
    q = q - (q @ p) * p
    q = q / (np.linalg.norm(q) + 1e-12)

    u = np.zeros((V, D), dtype=np.float64)
    u[:, 0] = s / c
    # base zero-sum part in dims 1..31
    base = -s / (sub * c)                              # [V]
    u[:, 1:] = base[:, None] * np.ones(sub)[None, :]
    # remaining energy to reach unit variance
    energy0 = (s / c) ** 2
    energy_base = (base ** 2) * sub
    rem = 1.0 - energy0 - energy_base
    rem = np.clip(rem, 0.0, None)
    # distribute rem into p and q directions
    # alpha_v^2 + beta_v^2 = rem ; choose alpha_v = sqrt(rem), beta_v = 0
    alpha_v = np.sqrt(rem)
    u[:, 1:] += alpha_v[:, None] * p[None, :]

    # Now u is (approx) zero-mean, unit-variance. Set embed = u.
    embed = u.astype(np.float32)
    embed = np.clip(embed, -1000.0, 1000.0)

    w["embed.weight"] = embed

    # ---- Set head so that head1 - head0 = c * e_0 ----
    head = np.zeros((2, D), dtype=np.float32)
    head[0, 0] = 0.0
    head[1, 0] = c
    # bias: logit1 - logit0 = bias
    w["head.weight"] = head
    hb = np.zeros(2, dtype=np.float32)
    hb[1] = bias
    w["head.bias"] = hb

    # ---- Keep everything else identity / zero ----
    for k in ["pos.weight"]:
        w[k] = np.zeros_like(w[k], dtype=np.float32)
    for k in ["block0.qkv.weight","block0.qkv.bias","block0.proj.weight","block0.proj.bias",
              "block0.ff1.weight","block0.ff1.bias","block0.ff2.weight","block0.ff2.bias",
              "block1.qkv.weight","block1.qkv.bias","block1.proj.weight","block1.proj.bias",
              "block1.ff1.weight","block1.ff1.bias","block1.ff2.weight","block1.ff2.bias"]:
        w[k] = np.zeros_like(w[k], dtype=np.float32)
    for k in ["block0.norm1.weight","block0.norm2.weight","block1.norm1.weight",
              "block1.norm2.weight","norm.weight"]:
        w[k] = np.ones_like(w[k], dtype=np.float32)
    for k in ["block0.norm1.bias","block0.norm2.bias","block1.norm1.bias",
              "block1.norm2.bias","norm.bias"]:
        w[k] = np.zeros_like(w[k], dtype=np.float32)

    # ---- Diagnostics ----
    diag = {
        "best_alpha": float(best_alpha),
        "cv_acc": float(best_acc),
        "s_absmax": float(np.abs(s).max()),
        "s_std": float(s.std()),
        "bias": float(bias),
        "c": float(c),
        "embed_rows_nonzero": int((np.abs(embed).sum(axis=1) > 0).sum()),
    }
    return {"weights": w, "diagnostics": diag}
