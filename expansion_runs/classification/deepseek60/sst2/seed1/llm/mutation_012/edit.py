import numpy as np
import math

def edit(w, f):
    X = np.asarray(f["x"])          # [4000,64] int64
    y = np.asarray(f["y"]).astype(np.int64)   # [4000]
    N, T = X.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]

    # ---- length-normalized bag-of-words design matrix ----
    mask = (X != 0)
    L = np.maximum(mask.sum(axis=1), 1).astype(np.float64)
    # accumulate counts per token
    F = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    keep = cols != 0
    np.add.at(F, (rows[keep], cols[keep]), 1.0)
    F /= L[:, None]

    # ---- closed-form ridge fit of a single score s = F @ wv + b ----
    # target: +1 for class 1, -1 for class 0
    t = np.where(y == 1, 1.0, -1.0).astype(np.float64)
    lam = 1.0
    # center features/targets for the bias
    Fm = F.mean(axis=0)
    tm = t.mean()
    Fc = F - Fm
    tc = t - tm
    A = Fc.T @ Fc + lam * np.eye(V)
    wv = np.linalg.solve(A, Fc.T @ tc)
    b = tm - Fm @ wv
    # scale so realized vectors stay well inside the unit sphere
    # max |wv| must satisfy |wv|/c <= sqrt(32) roughly; pick c accordingly
    c = 0.8420317798578142
    wv = wv.astype(np.float64)

    # ---- realize d[v] with head = c*[e0; e1] ----
    # logit_0 = c*d[v][0] , logit_1 = c*d[v][1]
    # want logit_1 - logit_0 = wv[v]  ->  d[v][1]-d[v][0] = wv[v]/c
    # choose d[v][0] = 0 , d[v][1] = wv[v]/c  (keeps both coords small)
    d = np.zeros((V, D), dtype=np.float64)
    d[:, 1] = wv / c

    # remaining coordinates 2..D-1 chosen so each row is zero-mean & unit-variance.
    # Need sum_{j>=2} d_j = -d_1  and  sum_{j>=2} d_j^2 = D - d_1^2.
    rem = D - 2
    s1 = d[:, 1]
    target_sum = -s1
    target_sq = D - s1 * s1
    # symmetric solution: all remaining coords equal a, plus a correction pair
    # use: a = target_sum/rem, then residual energy spread over two coords
    a = target_sum / rem
    base_sq = rem * a * a
    resid = target_sq - base_sq
    resid = np.maximum(resid, 0.0)
    # put residual into coordinates 2 and 3 as +/- delta
    delta = np.sqrt(resid / 2.0)
    d[:, 2] = a + delta
    d[:, 3] = a - delta
    if rem > 2:
        d[:, 4:] = a[:, None]

    # enforce exact zero mean / unit variance numerically
    d -= d.mean(axis=1, keepdims=True)
    nrm = np.sqrt((d * d).sum(axis=1, keepdims=True))
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    d = d / nrm * math.sqrt(D)

    # ---- write weights ----
    w["embed.weight"] = d.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    head = np.zeros((2, D), dtype=np.float32)
    head[0, 0] = c
    head[1, 1] = c
    w["head.weight"] = head
    w["head.bias"] = np.array([0.0, b], dtype=np.float32)

    # keep blocks/norms as-is (zero blocks, identity norm)
    diag = {
        "n_tokens": int(V),
        "n_samples": int(N),
        "mean_len": float(L.mean()),
        "ridge_lambda": float(lam),
        "c": float(c),
        "wv_absmax": float(np.abs(wv).max()),
        "wv_absmean": float(np.abs(wv).mean()),
        "bias": float(b),
        "d_var_mean": float((d * d).mean()),
        "d_mean_absmax": float(np.abs(d.mean(axis=1)).max()),
        "head_norm": float(np.linalg.norm(head)),
    }
    return {"weights": w, "diagnostics": diag}
