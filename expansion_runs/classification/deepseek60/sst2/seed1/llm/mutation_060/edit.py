import numpy as np

def edit(w, f):
    # ---- 0. read current parameters ----
    E = w["embed.weight"].astype(np.float64)      # [V,32]
    hw = w["head.weight"].astype(np.float64)      # [2,32]
    hb = w["head.bias"].astype(np.float64)        # [2]
    V, D = E.shape

    x = f["x"]; y = f["y"].astype(np.int64)       # [N,64], [N]
    bx = f["batch.x"]; by = f["batch.y"].astype(np.int64)
    N = x.shape[0]
    L = (x != 0).sum(1).astype(np.float64)
    L = np.maximum(L, 1.0)

    # ---- 1. verify the real forward pass on the paired batch ----
    def fwd(ids, E_, hw_, hb_):
        h = E_[ids]                                # [n,64,32]
        mu = h.mean(-1, keepdims=True)
        var = h.var(-1, keepdims=True)
        ln = (h - mu) / np.sqrt(var + 1e-5)        # norm layer (gamma=1,beta=0)
        m = (ids != 0).astype(np.float64)[..., None]
        pooled = (ln * m).sum(1) / np.maximum(m.sum(1), 1.0)
        return pooled @ hw_.T + hb_

    logits0 = fwd(bx, E, hw, hb)
    acc0 = (logits0.argmax(1) == by).mean()

    # ---- 2. closed-form bag-of-words fit (ridge normal equations) ----
    # design A[i,t] = count(i,t)/L_i  ->  score_i = A[i,:]@p + b
    A = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = x[i]
        row = row[row != 0]
        if row.size:
            np.add.at(A[i], row, 1.0)
    A /= L[:, None]

    t = 2.0 * y.astype(np.float64) - 1.0          # +1 / -1
    # center for intercept
    Amean = A.mean(0)
    Ac = A - Amean
    tmean = t.mean()
    tc = t - tmean

    best = None
    for lam in (1.0, 3.0, 10.0, 30.0, 100.0):
        G = Ac.T @ Ac + lam * np.eye(V)
        p = np.linalg.solve(G, Ac.T @ tc)
        b = tmean - Amean @ p
        # realized training accuracy with this (p,b)
        sc = A @ p + b
        acc = ((sc > 0).astype(np.int64) == y).mean()
        if best is None or acc > best[0]:
            best = (acc, p, b, lam)
    train_acc, p, b, lam = best

    # ---- 3. write into embed / head with exact LN-preserving geometry ----
    # û, v̂ orthonormal, mean-zero unit vectors in R^D
    def mean_zero_unit(seed_vec):
        v = seed_vec - seed_vec.mean()
        return v / np.linalg.norm(v)

    rng = np.random.RandomState(0)
    u = mean_zero_unit(rng.randn(D))
    v = rng.randn(D); v = v - v.mean()
    v = v - (v @ u) * u
    v = v / np.linalg.norm(v)

    # scale p so that |p_t| < sqrt(D) strictly (construction requires it)
    pmax = np.abs(p).max() if np.abs(p).max() > 0 else 1.0
    k = max(1.0, pmax / (0.9 * np.sqrt(D)))
    ps = p / k

    # embed[t] = ps_t*u + sqrt(D - ps_t^2)*v  ->  |embed[t]|=sqrt(D), mean 0
    q = np.sqrt(np.maximum(D - ps**2, 1e-6))
    Enew = ps[:, None] * u[None, :] + q[:, None] * v[None, :]
    # pad/unk rows: keep them finite & normalized too
    Enew[0] = np.sqrt(D) * v
    if V > 1:
        Enew[1] = np.sqrt(D) * v

    # head[1]-head[0] = k*u  ->  logit_diff = k*(A@p/k) + b = A@p + b
    hw_new = np.zeros_like(hw)
    hw_new[0] = -0.5 * k * u
    hw_new[1] = 0.5 * k * u
    hb_new = np.array([-0.5 * b, 0.5 * b])

    # ---- 4. verify realized accuracy on the paired batch ----
    logits1 = fwd(bx, Enew, hw_new, hb_new)
    acc1 = (logits1.argmax(1) == by).mean()

    # ---- 5. commit (keep original dtype/shape, finite, |.|<=1000) ----
    w["embed.weight"] = Enew.astype(np.float32)
    w["head.weight"] = hw_new.astype(np.float32)
    w["head.bias"] = hb_new.astype(np.float32)

    diag = {
        "acc_before_batch": float(acc0),
        "acc_after_batch": float(acc1),
        "train_acc_bow": float(train_acc),
        "lambda": float(lam),
        "p_absmax": float(np.abs(p).max()),
        "k": float(k),
        "bias": float(b),
        "embed_row_std_min": float(np.linalg.norm(Enew, axis=1).min() / np.sqrt(D)),
        "embed_row_mean_absmax": float(np.abs(Enew.mean(1)).max()),
        "head_w_norm": float(np.linalg.norm(hw_new[1] - hw_new[0])),
    }
    return {"weights": w, "diagnostics": diag}
