import numpy as np
import math

def edit(w, f):
    X = f["x"]                      # [N,64] int64
    y = f["y"].astype(np.float64)   # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]  # 2000

    # ---- build normalized unigram count matrix (pad id 0 excluded) ----
    Xn = np.zeros((N, V), dtype=np.float64)
    mask = (X != 0)
    lengths = mask.sum(axis=1).astype(np.float64)
    lengths[lengths == 0] = 1.0
    for i in range(N):
        ids = X[i][mask[i]]
        if ids.size:
            np.add.at(Xn[i], ids, 1.0)
    Xn /= lengths[:, None]

    # ---- ridge regression on +/-1 labels, closed form ----
    t = 2.0 * y - 1.0
    # center features to get an intercept
    Xm = Xn.mean(axis=0, keepdims=True)
    Xc = Xn - Xm
    tm = t.mean()
    tc = t - tm

    best_alpha, best_cv = None, -1.0
    alphas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    # 4-fold CV
    idx = np.arange(N)
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    folds = np.array_split(idx, 4)
    for a in alphas:
        accs = []
        for k in range(4):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(4) if j != k])
            Xtr, ttr = Xc[tr], tc[tr]
            G = Xtr.T @ Xtr + a * np.eye(V)
            wv = np.linalg.solve(G, Xtr.T @ ttr)
            b = tm - Xm @ wv
            pred = (Xc[te] @ wv + b) > 0
            accs.append(((pred == (t[te] > 0)).mean()))
        cv = float(np.mean(accs))
        if cv > best_cv:
            best_cv, best_alpha = cv, a

    # ---- refit on all data with chosen alpha ----
    G = Xc.T @ Xc + best_alpha * np.eye(V)
    wv = np.linalg.solve(G, Xc.T @ tc)
    b = float(tm - Xm @ wv)

    # ---- encode per-token scores s(v) = wv[v] into embed dim 0 ----
    s = wv.copy()
    smax = float(np.max(np.abs(s)))
    # keep |s/c| <= sqrt(31) so zero-mean/unit-var construction is feasible
    c = max(1.0, smax / (math.sqrt(31.0) * 0.98))
    s = s / c

    E = np.zeros((V, 32), dtype=np.float32)
    for v in range(V):
        a0 = float(s[v])
        # remaining 31 dims: need sum = -a0 and sum of squares = 32 - a0^2
        rem_sum = -a0
        rem_sq = 32.0 - a0 * a0
        # distribute equally among 31 dims
        mu = rem_sum / 31.0
        # sum of (x_i - mu)^2 = rem_sq - 31*mu^2
        var = (rem_sq - 31.0 * mu * mu) / 31.0
        var = max(var, 0.0)
        sd = math.sqrt(var)
        # deterministic alternating pattern
        for j in range(31):
            E[v, 1 + j] = mu + (sd if (j % 2 == 0) else -sd)
        E[v, 0] = a0

    # ---- set head so that (head1-head0) = c * e0, bias = b ----
    head = np.zeros((2, 32), dtype=np.float32)
    head[0, 0] = -0.5 * c
    head[1, 0] = 0.5 * c
    bias = np.array([-0.5 * b, 0.5 * b], dtype=np.float32)

    w["embed.weight"] = E
    w["head.weight"] = head
    w["head.bias"] = bias

    # zero out pos and blocks (already zero, but be explicit)
    for k in ["pos.weight",
              "block0.qkv.weight", "block0.qkv.bias",
              "block0.proj.weight", "block0.proj.bias",
              "block0.ff1.weight", "block0.ff1.bias",
              "block0.ff2.weight", "block0.ff2.bias",
              "block1.qkv.weight", "block1.qkv.bias",
              "block1.proj.weight", "block1.proj.bias",
              "block1.ff1.weight", "block1.ff1.bias",
              "block1.ff2.weight", "block1.ff2.bias"]:
        if k in w:
            w[k] = np.zeros_like(w[k])

    for k in ["block0.norm1.weight", "block0.norm1.bias",
              "block0.norm2.weight", "block0.norm2.bias",
              "block1.norm1.weight", "block1.norm1.bias",
              "block1.norm2.weight", "block1.norm2.bias",
              "norm.weight", "norm.bias"]:
        if k in w:
            if "weight" in k:
                w[k] = np.ones_like(w[k])
            else:
                w[k] = np.zeros_like(w[k])

    diag = {
        "best_alpha": best_alpha,
        "cv_acc": best_cv,
        "s_absmax": float(np.max(np.abs(wv))),
        "s_std": float(np.std(wv)),
        "c": c,
        "bias": b,
        "embed_nonzero": int(np.count_nonzero(E)),
    }
    return {"weights": w, "diagnostics": diag}
