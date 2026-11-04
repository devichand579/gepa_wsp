import numpy as np

def edit(w, f):
    X = np.asarray(f["x"]).astype(np.int64)          # [N,64]
    y = np.asarray(f["y"]).astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]                   # 2000
    D = w["embed.weight"].shape[1]                   # 32
    eps = 1e-5

    # ---------- length-normalized bag-of-words counts ----------
    # counts[v] = (#occurrences of v)/L ; L = number of non-pad tokens
    counts = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        nz = row[row != 0]
        L = max(len(nz), 1)
        np.add.at(counts[i], row, 1.0 / L)
    # counts already includes pad (id 0) contribution; drop pad column for fitting
    F = counts[:, 1:]                                # [N, V-1] features (exclude pad)

    # ---------- ridge linear classifier with CV-selected lambda ----------
    # binary labels: use +1 for class 1, -1 for class 0
    t = np.where(y == 1, 1.0, -1.0)
    Fm = F.mean(axis=0)
    Fs = F.std(axis=0) + 1e-8
    Fz = (F - Fm) / Fs

    def fit_ridge(A, tvec, lam):
        # solve (A^T A + lam I) w = A^T t  (with intercept handled by centering)
        n, d = A.shape
        Ac = A - A.mean(axis=0, keepdims=True)
        tc = tvec - tvec.mean()
        G = Ac.T @ Ac + lam * np.eye(d)
        wv = np.linalg.solve(G, Ac.T @ tc)
        b = tvec.mean() - A.mean(axis=0) @ wv
        return wv, b

    # 5-fold CV to pick lambda
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_lam, best_acc = 1e-2, -1.0
    for lam in lambdas:
        accs = []
        for k in range(5):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            wv, b = fit_ridge(Fz[tr], t[tr], lam)
            pred = np.where(Fz[va] @ wv + b > 0, 1, 0)
            accs.append((pred == y[va]).mean())
        acc = float(np.mean(accs))
        if acc > best_acc:
            best_acc, best_lam = acc, lam

    wv, b = fit_ridge(Fz, t, best_lam)
    # per-token margin weight for token v (v=1..V-1): m_v = wv[v-1]/Fs[v-1]
    # plus intercept correction from centering
    m = np.zeros(V, dtype=np.float64)
    m[1:] = wv / Fs
    b_eff = b - np.sum(wv * Fm / Fs)

    # ---------- realize the linear classifier inside the transformer ----------
    # pooled = mean_t LayerNorm(embed[token_t]) = sum_v (n_v/L) u_v
    # margin = d @ pooled + b'  with d = head[0]-head[1]
    # choose d = a * g, g zero-mean unit-norm, and u_v = t_v g + sqrt(32 - t_v^2) r
    # with t_v = m_v / a  (clipped), r a fixed unit vector orthogonal to g and 1.
    Dd = D
    ones = np.ones(Dd)
    # direction g: zero-mean, unit norm
    g = np.zeros(Dd)
    g[0] = 1.0
    g[1] = -1.0
    g = g / np.linalg.norm(g)
    g = g - (g @ ones / Dd) * ones
    g = g / np.linalg.norm(g)
    # r: unit vector orthogonal to g and ones
    r = np.zeros(Dd)
    r[2] = 1.0
    r = r - (r @ g) * g - (r @ ones / Dd) * ones
    r = r / np.linalg.norm(r)

    a = max(np.max(np.abs(m)) / 0.95, 1e-6)          # keep |t_v| <= ~0.95
    t_v = np.clip(m / a, -0.95, 0.95)
    perp_scale = np.sqrt(np.maximum(Dd - t_v ** 2, 0.0))
    U = t_v[:, None] * g[None, :] + perp_scale[:, None] * r[None, :]   # [V,32]
    # ensure zero-mean (numerically)
    U = U - U.mean(axis=1, keepdims=True)

    # embed[v] = U[v]  (LayerNorm(U[v]) ~ U[v] since zero-mean/unit-var)
    embed = U.astype(np.float32)
    # keep pad row small but consistent (pad excluded from pooling anyway)
    embed[0] = 0.0

    # head: logits = head @ pooled + bias ; margin = (head0-head1)@pooled + (b0-b1)
    # set head0 = a/2 * g, head1 = -a/2 * g  => head0-head1 = a*g = d
    head = np.zeros((2, Dd), dtype=np.float32)
    head[0] = (a / 2.0) * g
    head[1] = -(a / 2.0) * g
    hb = np.zeros(2, dtype=np.float32)
    hb[0] = b_eff / 2.0
    hb[1] = -b_eff / 2.0

    w["embed.weight"] = embed
    w["head.weight"] = head
    w["head.bias"] = hb
    # keep everything else identity / zero as before
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    for name in ["block0.norm1.weight", "block0.norm2.weight",
                 "block1.norm1.weight", "block1.norm2.weight", "norm.weight"]:
        w[name] = np.ones_like(w[name])
    for name in ["block0.norm1.bias", "block0.norm2.bias",
                 "block1.norm1.bias", "block1.norm2.bias", "norm.bias"]:
        w[name] = np.zeros_like(w[name])
    for name in ["block0.qkv.weight", "block0.qkv.bias", "block0.proj.weight",
                 "block0.proj.bias", "block0.ff1.weight", "block0.ff1.bias",
                 "block0.ff2.weight", "block0.ff2.bias",
                 "block1.qkv.weight", "block1.qkv.bias", "block1.proj.weight",
                 "block1.proj.bias", "block1.ff1.weight", "block1.ff1.bias",
                 "block1.ff2.weight", "block1.ff2.bias"]:
        w[name] = np.zeros_like(w[name])

    # clip to safe range
    for k in list(w.keys()):
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[k] = arr.astype(np.float32)

    diag = {
        "cv_best_lambda": float(best_lam),
        "cv_best_acc": float(best_acc),
        "margin_absmax": float(np.max(np.abs(m))),
        "embed_absmax": float(np.max(np.abs(embed))),
        "head_absmax": float(np.max(np.abs(head))),
        "b_eff": float(b_eff),
    }
    return {"weights": w, "diagnostics": diag}
