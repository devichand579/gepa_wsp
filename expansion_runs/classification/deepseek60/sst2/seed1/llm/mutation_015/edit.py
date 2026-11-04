import math
import numpy as np

def edit(w, f):
    X = np.asarray(f["x"]).astype(np.int64)          # [N,64]
    y = np.asarray(f["y"]).astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]                   # 2000

    # ---- length-normalized bag-of-words features ----
    mask = (X != 0)
    L = np.maximum(mask.sum(axis=1), 1).astype(np.float64)   # [N]
    F = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    valid = cols != 0
    np.add.at(F, (rows[valid], cols[valid]), 1.0)
    F /= L[:, None]                                   # length-normalized counts

    t = np.where(y == 1, 1.0, -1.0)                   # ±1 targets

    # ---- closed-form ridge with GCV-selected lambda ----
    # center targets; fit intercept separately (unregularized)
    tbar = t.mean()
    tc = t - tbar
    # Gram matrix (2000x2000) — cheap
    G = F.T @ F                                       # [V,V]
    b = F.T @ tc                                      # [V]
    dg = np.diag(G).copy()

    def solve(lam):
        A = G + lam * np.eye(V)
        return np.linalg.solve(A, b)

    # generalized cross-validation over a small closed-form set of lambdas
    # (each solve is a direct linear solve, not an iterative optimizer)
    best_lam, best_w, best_gcv = None, None, np.inf
    for lam in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        wv = solve(lam)
        # hat-matrix trace approximation for GCV (closed form)
        # trace(H) = sum_i h_ii ; use diagonal of F (F^T F + lam I)^-1 F^T
        A = G + lam * np.eye(V)
        Ainv_diag = np.linalg.inv(A)                  # [V,V]
        # leverage h_ii = f_i^T Ainv f_i ; approximate via trace
        trH = float(np.sum(F * (F @ Ainv_diag)))      # sum_i f_i^T Ainv f_i
        resid = F @ wv - tc
        rss = float(resid @ resid)
        denom = max(N - trH, 1e-6)
        gcv = (rss / N) / (denom / N) ** 2
        if gcv < best_gcv:
            best_gcv, best_lam, best_w = gcv, lam, wv

    wv = best_w                                        # [V] per-token scalar score
    bias = tbar                                       # scalar intercept

    # ---- realize exactly: head = c*[e0;e1], embed[v]=d[v] ----
    # desired per-token 2-vector: w_v = (wv[v], -wv[v]) so that
    # logit0 - logit1 = 2*(head[0] @ pooled) + (bias0-bias1)
    # We want score s = wv·pooled + bias to drive class1 when s>0.
    # Set head[0] = c*e0, head[1] = -c*e0  -> head @ d = c*(d0, -d0)
    # then head[0]@pooled = c*pooled0 and logit0-logit1 = 2*c*pooled0 + (b0-b1).
    # We need c*pooled0 = wv·pooled  =>  pooled0 = (wv·pooled)/c.
    # pooled0 = Σ_v p_v d[v][0], so set d[v][0] = wv[v]/c.
    wmax = float(np.max(np.abs(wv))) if wv.size else 1.0
    c = max(2.0 * wmax, 1e-3)                          # keep d0 well inside unit sphere

    d = np.zeros((V, 32), dtype=np.float64)
    d[:, 0] = wv / c                                   # signal coordinate
    # remaining 31 coords must make each row zero-mean, unit-variance
    # sum of coords 1..31 = -d0 ; sum of squares = 32 - d0^2
    s = -d[:, 0]                                       # required sum over 31 coords
    q = 32.0 - d[:, 0] ** 2                            # required sum of squares
    # distribute over 31 coords: 30 equal (alpha) + 1 (beta)
    n = 31
    # 30*alpha + beta = s ; 30*alpha^2 + beta^2 = q
    # => 30 a^2 + (s-30a)^2 = q => 930 a^2 - 60 s a + s^2 - q = 0
    A2 = 930.0
    B2 = -60.0 * s
    C2 = s ** 2 - q
    disc = np.maximum(B2 ** 2 - 4.0 * A2 * C2, 0.0)
    alpha = (-B2 - np.sqrt(disc)) / (2.0 * A2)         # pick stable root
    beta = s - 30.0 * alpha
    d[:, 1:31] = alpha[:, None]
    d[:, 31] = beta

    # exact renormalization to guard against float error
    d = d - d.mean(axis=1, keepdims=True)
    d = d / np.sqrt((d ** 2).mean(axis=1, keepdims=True) + 1e-12)

    # ---- write weights ----
    ew = np.asarray(w["embed.weight"], dtype=np.float32)
    ew[:] = d.astype(np.float32)
    w["embed.weight"] = ew

    pw = np.asarray(w["pos.weight"], dtype=np.float32)
    pw[:] = 0.0
    w["pos.weight"] = pw

    hw = np.asarray(w["head.weight"], dtype=np.float32)
    hw[:] = 0.0
    hw[0, 0] = c
    hw[1, 0] = -c
    w["head.weight"] = hw

    hb = np.asarray(w["head.bias"], dtype=np.float32)
    # logit0 - logit1 = 2*c*pooled0 + (hb0 - hb1); want = 2*(wv·pooled) + 2*bias
    # since pooled0 = (wv·pooled)/c, 2*c*pooled0 = 2*(wv·pooled). So need hb0-hb1 = 2*bias.
    hb[0] = float(bias)
    hb[1] = float(-bias)
    w["head.bias"] = hb

    # ---- diagnostics ----
    diag = {
        "lambda": float(best_lam),
        "gcv": float(best_gcv),
        "bias": float(bias),
        "c": float(c),
        "wv_absmax": float(np.max(np.abs(wv))),
        "wv_absmean": float(np.mean(np.abs(wv))),
        "d_var_mean": float(np.mean((d ** 2).mean(axis=1))),
        "d_mean_absmax": float(np.max(np.abs(d.mean(axis=1)))),
        "n_samples": int(N),
        "n_tokens": int(V),
    }
    return {"weights": w, "diagnostics": diag}
