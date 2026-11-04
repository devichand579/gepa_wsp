import numpy as np

def edit(w, f):
    # ---------- helpers ----------
    def layernorm(x, weight, bias, eps=1e-5):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * weight + bias

    x_ids = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)              # [N]
    N, L = x_ids.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]

    # ---------- 1. normalized BOW features ----------
    # counts per token, length-normalized (mask pad=0)
    mask = (x_ids != 0).astype(np.float64)   # [N,64]
    length = np.maximum(mask.sum(axis=1), 1.0)
    # build sparse-ish count matrix via bincount per row is slow; use one-hot sum
    counts = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = x_ids.reshape(-1)
    np.add.at(counts, (rows, cols), 1.0)
    bow = counts / length[:, None]           # [N,V]

    # ---------- 2. closed-form ridge linear classifier on BOW ----------
    # targets +1 / -1
    t = np.where(y == 1, 1.0, -1.0)
    X = np.concatenate([bow, np.ones((N, 1))], axis=1)   # [N,V+1]
    # ridge with a few candidate lambdas, pick best in-sample accuracy (closed form)
    best = None
    for lam in (0.03, 0.1, 0.3, 1.0, 3.0):
        A = X.T @ X + lam * np.eye(X.shape[1])
        sol = np.linalg.solve(A, X.T @ t)
        pred = np.sign(X @ sol)
        acc = (pred == t).mean()
        if best is None or acc > best[0]:
            best = (acc, sol, lam)
    acc_lin, sol, lam = best
    a = sol[:-1]          # [V] token weights
    b = sol[-1]           # scalar bias

    # ---------- 3. encode linear backbone into embed / head ----------
    # We want d·LayerNorm(embed[v]) = a_v  with d = head[1]-head[0].
    # Build a 2-D zero-mean orthonormal frame p,q in R^D.
    p = np.zeros(D); p[0] = 1.0; p[1] = -1.0
    p = p / np.linalg.norm(p)
    q = np.zeros(D); q[2] = 1.0; q[3] = -1.0
    q = q / np.linalg.norm(q)
    # ensure orthonormal
    q = q - (q @ p) * p
    q = q / np.linalg.norm(q)

    maxa = np.max(np.abs(a)) + 1e-9
    R = maxa / (np.sqrt(D) * 0.99)          # so |a_v| <= R*sqrt(D)*0.99
    cosv = np.clip(a / (R * np.sqrt(D)), -0.999, 0.999)
    theta = np.arccos(cosv)
    U = np.sqrt(D) * (np.cos(theta)[:, None] * p[None, :] +
                      np.sin(theta)[:, None] * q[None, :])   # [V,D]
    # U rows are zero-mean, unit-variance
    w["embed.weight"] = U.astype(np.float32)

    d = R * p                                # [D]
    w["head.weight"] = np.stack([-0.5 * d, 0.5 * d]).astype(np.float32)
    w["head.bias"] = np.array([-0.5 * b, 0.5 * b], dtype=np.float32)

    # ---------- 4. keep blocks identity, norm identity ----------
    for k in list(w.keys()):
        if k.startswith("block0.") or k.startswith("block1."):
            w[k] = np.zeros_like(w[k])
    w["norm.weight"] = np.ones(D, dtype=np.float32)
    w["norm.bias"] = np.zeros(D, dtype=np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # ---------- 5. recompute realized pooled features & refit head (closed form) ----------
    # forward: h = embed[ids]; blocks identity; norm -> LayerNorm; pool
    emb = w["embed.weight"].astype(np.float64)          # [V,D]
    h = emb[x_ids]                                       # [N,64,D]
    hn = layernorm(h, np.ones(D), np.zeros(D))           # [N,64,D]
    pooled = (hn * mask[:, :, None]).sum(axis=1) / length[:, None]   # [N,D]

    # refit head by ridge on realized pooled features
    Xp = np.concatenate([pooled, np.ones((N, 1))], axis=1)
    best2 = None
    for lam2 in (0.03, 0.1, 0.3, 1.0, 3.0):
        A = Xp.T @ Xp + lam2 * np.eye(Xp.shape[1])
        sol2 = np.linalg.solve(A, Xp.T @ t)
        pred = np.sign(Xp @ sol2)
        acc = (pred == t).mean()
        if best2 is None or acc > best2[0]:
            best2 = (acc, sol2, lam2)
    acc_refit, sol2, lam2 = best2
    wd = sol2[:-1]
    bd = sol2[-1]
    w["head.weight"] = np.stack([-0.5 * wd, 0.5 * wd]).astype(np.float32)
    w["head.bias"] = np.array([-0.5 * bd, 0.5 * bd], dtype=np.float32)

    # ---------- 6. clip / sanitize ----------
    for k in w:
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[k] = arr.astype(np.float32)

    diag = {
        "acc_lin_bow": float(acc_lin),
        "lam_lin": float(lam),
        "acc_refit": float(acc_refit),
        "lam_refit": float(lam2),
        "a_absmax": float(np.max(np.abs(a))),
        "head_absmax": float(np.max(np.abs(w["head.weight"]))),
        "embed_absmax": float(np.max(np.abs(w["embed.weight"]))),
    }
    return {"weights": w, "diagnostics": diag}
