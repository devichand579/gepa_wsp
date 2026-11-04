import numpy as np
import math

def edit(w, f):
    # ---- read feedback ----
    x = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- build normalized unigram count features ----
    # counts per example
    counts = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = x[i]
        nz = row[row != 0]
        if nz.size == 0:
            continue
        np.add.at(counts[i], nz, 1.0)
    lengths = counts.sum(axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    X = counts / lengths                    # [N,V] normalized counts
    t = (y.astype(np.float64) * 2.0 - 1.0)  # ±1 targets

    # ---- ridge fit with CV over alpha ----
    # center X for intercept handling
    Xmean = X.mean(axis=0, keepdims=True)
    Xc = X - Xmean
    tmean = t.mean()

    def fit_ridge(Xc, t, alpha):
        # solve (Xc^T Xc + alpha I) beta = Xc^T (t - tmean)
        A = Xc.T @ Xc
        A[np.diag_indices_from(A)] += alpha
        rhs = Xc.T @ (t - tmean)
        beta = np.linalg.solve(A, rhs)
        b0 = tmean - Xmean.ravel() @ beta
        return beta, b0

    # CV
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    nfold = 5
    folds = np.array_split(idx, nfold)
    alphas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    best_alpha = alphas[0]
    best_cv = -1.0
    for alpha in alphas:
        accs = []
        for k in range(nfold):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(nfold) if j != k])
            Xtr = X[tr]; ttr = t[tr]
            Xte = X[te]; tte = t[te]
            Xm = Xtr.mean(axis=0, keepdims=True)
            Xc_tr = Xtr - Xm
            tm = ttr.mean()
            A = Xc_tr.T @ Xc_tr
            A[np.diag_indices_from(A)] += alpha
            rhs = Xc_tr.T @ (ttr - tm)
            beta = np.linalg.solve(A, rhs)
            b0 = tm - Xm.ravel() @ beta
            pred = (Xte @ beta + b0) > 0
            accs.append((pred == (tte > 0)).mean())
        cv = float(np.mean(accs))
        if cv > best_cv:
            best_cv = cv
            best_alpha = alpha

    beta, b0 = fit_ridge(Xc, t, best_alpha)
    # per-token score s(v) = beta[v]; intercept b0
    s = beta.astype(np.float64)             # [V]
    b = float(b0)

    # ---- construct embed so that LN(embed[v]) = e_v with d·e_v = s(v) ----
    # choose d direction (head difference). Use a fixed random-ish direction
    # but we will set head accordingly, so d is our choice.
    # We need e_v: 1·e_v = 0, ||e_v||^2 = D, d·e_v = s(v).
    # Pick d = e_0 (first basis vector) for simplicity -> then e_v[0] = s(v).
    # But we need zero mean and unit variance. Solve:
    #   e_v = [s(v), q_v, ...] with sum = 0 and sum of squares = D.
    # Use a 2-parameter family: e_v = s(v)*u + c_v * n, where u,n orthonormal,
    # u ⊥ 1, n ⊥ 1, n ⊥ u.  Then d = u (so d·e_v = s(v)).
    # Build u,n in the subspace orthogonal to 1 (dim D-1).
    D_ = D
    one = np.ones(D_, dtype=np.float64)
    # basis for subspace orthogonal to 1
    # use QR on [one, e0, e1, ...] to get orthonormal basis
    M = np.eye(D_, dtype=np.float64)
    M = M - np.outer(one, one) / D_          # projector
    # orthonormal basis via SVD of projector (rank D-1)
    U, S, Vt = np.linalg.svd(M)
    B = U[:, :D_-1]                           # [D, D-1] orthonormal, columns ⊥ 1
    u = B[:, 0].copy()                        # direction for score
    # ensure u is unit
    u = u / np.linalg.norm(u)
    # n direction orthogonal to u within span(B)
    # pick second column, orthogonalize
    n = B[:, 1].copy()
    n = n - (n @ u) * u
    n = n / np.linalg.norm(n)

    # scale so that max |s| fits: ||e_v||^2 = s(v)^2 + c_v^2 = D
    smax = np.max(np.abs(s)) + 1e-12
    # we need s(v)^2 <= D ; if not, scale down s (and b) uniformly
    scale = 1.0
    if smax > math.sqrt(D_) * 0.98:
        scale = (math.sqrt(D_) * 0.98) / smax
        s = s * scale
        b = b * scale
    # now build e_v
    E = np.zeros((V, D_), dtype=np.float64)
    for v in range(V):
        sv = s[v]
        rem = D_ - sv * sv
        if rem < 0:
            rem = 0.0
        cv = math.sqrt(rem)
        E[v] = sv * u + cv * n
    # verify zero mean / unit variance approximately
    # set embed
    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # ---- set head so that d = u (head1 - head0 = u) and bias carries b ----
    # logit1 - logit0 = (head1-head0)·pooled + (bias1-bias0)
    # want = u·pooled + b  (since pooled = mean e_v, u·pooled = mean s)
    hw = np.zeros_like(w["head.weight"])     # [2,32]
    hw[1] = 0.5 * u
    hw[0] = -0.5 * u
    w["head.weight"] = hw.astype(np.float32)
    hb = np.zeros_like(w["head.bias"])
    hb[1] = 0.5 * b
    hb[0] = -0.5 * b
    w["head.bias"] = hb.astype(np.float32)

    # ---- ensure all other params finite & small (keep blocks zero) ----
    for k in list(w.keys()):
        if k in ("embed.weight", "head.weight", "head.bias", "pos.weight"):
            continue
        arr = w[k]
        if not np.all(np.isfinite(arr)):
            w[k] = np.zeros_like(arr)
        else:
            w[k] = np.clip(arr, -1000.0, 1000.0).astype(np.float32)

    # ---- diagnostics ----
    diag = {
        "best_alpha": float(best_alpha),
        "cv_acc": float(best_cv),
        "scale": float(scale),
        "s_absmax": float(np.max(np.abs(s))),
        "bias": float(b),
        "embed_nonzero": int(np.count_nonzero(w["embed.weight"])),
        "head_norm": float(np.linalg.norm(w["head.weight"])),
    }
    return {"weights": w, "diagnostics": diag}
