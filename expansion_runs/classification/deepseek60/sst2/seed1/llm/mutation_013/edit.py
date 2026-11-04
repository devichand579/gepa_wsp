import numpy as np

def edit(w, f):
    X = f["x"]                      # [N,64] int64
    y = f["y"].astype(np.float64)   # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]  # 2000

    # ---- length-normalized unigram count features p_v = count[v]/L ----
    mask = (X != 0).astype(np.float64)          # [N,64]
    L = mask.sum(1)                             # [N]
    L[L == 0] = 1.0
    P = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        m = mask[i]
        for t in range(T):
            if m[t] > 0:
                P[i, row[t]] += 1.0
    P /= L[:, None]                             # length-normalized counts

    # ---- closed-form ridge on ±1 labels, with GCV-chosen lambda ----
    # center features to fit an intercept implicitly
    mu = P.mean(0)
    Pc = P - mu
    t = 2.0 * y - 1.0                           # ±1 targets
    tc = t - t.mean()
    # SVD of centered design
    U, s, Vt = np.linalg.svd(Pc, full_matrices=False)
    s2 = s * s
    # GCV over a small grid of lambdas (closed form, no iterative search)
    best_lam, best_gcv = None, np.inf
    for lam in (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        # hat matrix trace for ridge in SVD basis
        d = s2 / (s2 + lam)
        # GCV = ||(I-H) t||^2 / (N - tr(H))^2
        resid = tc - U @ (d * (U.T @ tc))
        trH = d.sum()
        gcv = (resid @ resid) / max(N - trH, 1e-9) ** 2
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, lam
    lam = best_lam
    d = s2 / (s2 + lam)
    beta = Vt.T @ (d * (U.T @ tc))              # [V] ridge coefficients
    b0 = t.mean() - mu @ beta                   # intercept

    # ---- realize as per-token 2-vectors w_v = (c*d[v][0], c*d[v][1]) ----
    # head = c * [e0; e1];  d[v] zero-mean, unit-variance 32-vector.
    c = 0.8420317798578142                      # keep head scale from parent
    # desired first-two coords of d[v]
    a = beta / c                                # [V]
    # build d[v]: first two coords = a, remaining 30 chosen to keep
    # sum=0 and sum of squares = 32.
    D = np.zeros((V, 32), dtype=np.float64)
    D[:, 0] = a
    # distribute the remaining mass symmetrically so mean stays ~0
    rem = 32.0 - a * a
    rem = np.maximum(rem, 1e-6)
    # put residual variance into dims 2..31, with sum zero
    # simple choice: dims 2 and 3 carry ±sqrt(rem/2), rest zero
    half = np.sqrt(rem / 2.0)
    D[:, 2] = half
    D[:, 3] = -half
    # enforce exact zero mean by subtracting the row mean (tiny correction)
    D -= D.mean(1, keepdims=True)
    # renormalize to unit variance
    D /= np.sqrt((D * D).mean(1, keepdims=True))

    # ---- write weights ----
    w["embed.weight"] = D.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    for k in list(w.keys()):
        if k.startswith("block0.") or k.startswith("block1."):
            w[k] = np.zeros_like(w[k])
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])
    hw = np.zeros_like(w["head.weight"])
    hw[0, 0] = c
    hw[1, 1] = c
    w["head.weight"] = hw
    hb = np.zeros_like(w["head.bias"])
    hb[1] = b0
    w["head.bias"] = hb

    diag = {
        "lambda": float(lam),
        "beta_absmax": float(np.abs(beta).max()),
        "bias": float(b0),
        "d_var_mean": float((D * D).mean()),
        "d_mean_absmax": float(np.abs(D.mean(1)).max()),
    }
    return {"weights": w, "diagnostics": diag}
