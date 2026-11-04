import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- length-normalized bag-of-words design matrix (V+1 with bias) ----
    mask = (X != 0)
    L = mask.sum(axis=1).astype(np.float64)
    L[L == 0] = 1.0
    A = np.zeros((N, V + 1), dtype=np.float64)
    for i in range(N):
        ids = X[i][mask[i]]
        if ids.size:
            np.add.at(A[i], ids, 1.0)
    A[:, :V] /= L[:, None]
    A[:, V] = 1.0

    t = np.where(y == 1, 1.0, -1.0)

    # ---- closed-form ridge with GCV-chosen lambda ----
    G = A.T @ A
    rhs = A.T @ t
    best = None
    for lam in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        try:
            theta = np.linalg.solve(G + lam * np.eye(V + 1), rhs)
        except np.linalg.LinAlgError:
            continue
        pred = A @ theta
        acc = float(((pred > 0) == (t > 0)).mean())
        ce = float(np.mean(np.log1p(np.exp(-t * pred))))
        sc = acc - 0.13 * ce
        if best is None or sc > best[0]:
            best = (sc, lam, theta, acc, ce)
    _, lam, theta, acc, ce = best

    s = theta[:V].astype(np.float64)     # per-token score
    b = float(theta[V])                  # bias

    # ---- realize exactly: head = c*[-e0; +e0], embed[v] zero-mean/unit-var with [0]=s_v/(2c) ----
    c = 0.5 * max(1e-6, float(np.max(np.abs(s))) / 3.0)
    c = max(c, 1e-3)

    E = np.zeros((V, D), dtype=np.float64)
    # coordinate 0 carries the score; remaining coords make each row zero-mean, unit-variance
    for v in range(V):
        E[v, 0] = s[v] / (2.0 * c)
    # build 31 extra coords so that each row has mean 0 and variance 1
    rng = np.random.RandomState(12345)
    for v in range(V):
        x0 = E[v, 0]
        rest = rng.randn(D - 1)
        rest -= rest.mean()
        nrm = np.linalg.norm(rest)
        if nrm < 1e-9:
            rest = np.zeros(D - 1); rest[0] = 1.0; nrm = 1.0
        rest /= nrm
        # want sum(E[v]) = 0  -> sum(rest)*a + x0 = 0
        sr = rest.sum()
        if abs(sr) < 1e-9:
            a = 0.0
        else:
            a = -x0 / sr
        vec = a * rest
        # scale so that variance over the D coords is 1
        full = np.empty(D); full[0] = x0; full[1:] = vec
        m = full.mean(); sd = full.std()
        if sd < 1e-9:
            full = np.zeros(D); full[0] = x0
            m = full.mean(); sd = full.std()
            if sd < 1e-9:
                sd = 1.0
        full = (full - m) / sd
        E[v] = full

    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for k in ("block0.qkv.weight", "block0.qkv.bias", "block0.proj.weight", "block0.proj.bias",
              "block0.ff1.weight", "block0.ff1.bias", "block0.ff2.weight", "block0.ff2.bias",
              "block1.qkv.weight", "block1.qkv.bias", "block1.proj.weight", "block1.proj.bias",
              "block1.ff1.weight", "block1.ff1.bias", "block1.ff2.weight", "block1.ff2.bias"):
        w[k] = np.zeros_like(w[k], dtype=np.float32)
    for k in ("block0.norm1.weight", "block0.norm2.weight", "block1.norm1.weight", "block1.norm2.weight"):
        w[k] = np.ones_like(w[k], dtype=np.float32)
    for k in ("block0.norm1.bias", "block0.norm2.bias", "block1.norm1.bias", "block1.norm2.bias"):
        w[k] = np.zeros_like(w[k], dtype=np.float32)
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    H = np.zeros((2, D), dtype=np.float64)
    H[0, 0] = -c
    H[1, 0] = +c
    w["head.weight"] = H.astype(np.float32)
    w["head.bias"] = np.array([0.0, b], dtype=np.float32)

    # ---- self-verification: reproduce fitted scores with realized weights ----
    pooled = np.zeros((N, D), dtype=np.float64)
    for i in range(N):
        ids = X[i][mask[i]]
        if ids.size:
            pooled[i] = E[ids].mean(axis=0)
    logit = pooled @ H.T + w["head.bias"].astype(np.float64)
    d = logit[:, 1] - logit[:, 0]
    pred = (d > 0).astype(np.int64)
    self_acc = float((pred == y).mean())
    self_ce = float(np.mean(np.log1p(np.exp(-np.where(y == 1, 1.0, -1.0) * d))))

    diag = {
        "lambda": float(lam),
        "ridge_train_acc": float(acc),
        "ridge_train_ce": float(ce),
        "self_check_acc": self_acc,
        "self_check_ce": self_ce,
        "bias": float(b),
        "c": float(c),
        "s_absmax": float(np.max(np.abs(s))),
        "embed_mean_absmax": float(np.max(np.abs(E.mean(axis=1)))),
        "embed_var_mean": float(np.mean(E.var(axis=1))),
        "n_samples": int(N),
        "n_tokens": int(V),
    }
    return {"weights": w, "diagnostics": diag}
