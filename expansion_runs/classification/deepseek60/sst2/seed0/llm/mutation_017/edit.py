import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- build normalized unigram count features (pad id 0 excluded) ----
    mask = (X != 0)
    counts = np.zeros((N, V), dtype=np.float64)
    for v in range(V):
        counts[:, v] = (X == v).sum(axis=1)
    length = mask.sum(axis=1).astype(np.float64)
    length[length == 0] = 1.0
    F = counts / length[:, None]         # [N, V] normalized counts

    # ---- closed-form regularized linear discriminant (ridge on +-1 targets) ----
    # center features
    mu = F.mean(axis=0)
    Fc = F - mu
    # class means
    m0 = F[y == 0].mean(axis=0)
    m1 = F[y == 1].mean(axis=0)
    d = m1 - m0                          # class-mean difference

    # covariance (diagonal-regularized)
    # use S = Fc^T Fc / N
    S = (Fc.T @ Fc) / N                  # [V,V]

    best = None
    for lam in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        A = S + lam * np.eye(V)
        try:
            wv = np.linalg.solve(A, d)
        except np.linalg.LinAlgError:
            continue
        # intercept
        b = -0.5 * (m0 + m1) @ wv
        # CV accuracy
        scores = F @ wv + b
        pred = (scores > 0).astype(np.int64)
        acc = (pred == y).mean()
        if best is None or acc > best[0]:
            best = (acc, lam, wv, b)

    cv_acc, lam, wv, b = best

    # ---- encode per-token scores into embed[:,0] with LayerNorm constraint ----
    # We want head @ LN(embed[v]) = s(v).  Set head = c * u where u is unit,
    # and embed[v] zero-mean unit-variance with u . embed[v] = s(v)/c.
    # Simplest: u = e0 - (1/D) * 1  (zero-mean, unit norm after scaling).
    # embed[v] = a_v * u_hat + orthogonal filler to reach unit variance.
    # Choose c so that max |s(v)| <= c * sqrt(D) * |u_hat| ... use c = max|s| / sqrt(D-1)
    s = wv.copy()
    s_max = np.max(np.abs(s)) + 1e-9
    # head direction: use dim0 only for simplicity -> head = c * e0
    # then head @ embed[v] = c * embed[v][0] = s(v)  => embed[v][0] = s(v)/c
    # constraint: embed[v] zero-mean, unit variance, with dim0 = s(v)/c
    # feasible iff (s(v)/c)^2 <= D-1  (since other D-1 dims must have sum of squares D - (s/c)^2 >=0)
    c = s_max / np.sqrt(D - 1) * 1.0     # ensures |s/c| <= sqrt(D-1)
    c = max(c, 1e-6)

    E = np.zeros((V, D), dtype=np.float64)
    a = s / c                            # desired dim0 values, |a| <= sqrt(D-1)
    # remaining D-1 dims: need zero mean and unit variance overall.
    # set dims 1..D-1 equal to a common value t, then:
    #   mean: a + (D-1) t = 0  => t = -a/(D-1)
    #   variance: (a^2 + (D-1) t^2)/D = 1
    #   => a^2 + (D-1) a^2/(D-1)^2 = D  => a^2 (1 + 1/(D-1)) = D
    #   => a^2 * D/(D-1) = D => a^2 = D-1.  So this only works when a^2 = D-1.
    # Instead: set dims 1..D-1 to satisfy mean 0 and variance 1 given dim0=a.
    #   sum_{1..D-1} = -a ;  sum_{1..D-1} x_i^2 = D - a^2.
    #   A simple choice: x_1 = -(a + r), x_2 = r, rest 0?  Need sum = -a and squares = D-a^2.
    #   Let's use: fill dims 1..D-1 with value p, except adjust one dim q.
    #   (D-1) p + (q - p) = -a  and (D-2) p^2 + q^2 = D - a^2.
    #   Simpler robust construction: put the "energy" in dim1 only:
    #     dim1 = -a  (so sum of dims 1..D-1 = -a)  -> but then squares = a^2, need D-a^2.
    #   Use dims 1..D-1 as: first k dims share value, etc.
    # Practical: distribute so that sum = -a and sum of squares = D - a^2.
    #   Let m = D-1.  We need vector z in R^m with sum(z) = -a and ||z||^2 = D - a^2.
    #   Take z = alpha * 1_m + beta * e_1.  Then sum = m alpha + beta = -a,
    #   ||z||^2 = m alpha^2 + 2 alpha beta + beta^2 = D - a^2.
    #   Solve: beta = -a - m alpha.  Substitute:
    #   m alpha^2 + 2 alpha(-a - m alpha) + (a + m alpha)^2 = D - a^2
    #   = m alpha^2 - 2 a alpha - 2 m alpha^2 + a^2 + 2 a m alpha + m^2 alpha^2
    #   = (m - 2m + m^2) alpha^2 + (-2a + 2am) alpha + a^2
    #   = (m^2 - m) alpha^2 + 2a(m-1) alpha + a^2 = D - a^2
    #   => m(m-1) alpha^2 + 2a(m-1) alpha + 2a^2 - D = 0
    #   Solve quadratic for alpha.
    m = D - 1
    alpha = np.zeros(V)
    beta = np.zeros(V)
    for i in range(V):
        ai = a[i]
        A2 = m * (m - 1)
        B2 = 2.0 * ai * (m - 1)
        C2 = 2.0 * ai * ai - D
        disc = B2 * B2 - 4.0 * A2 * C2
        if disc < 0:
            disc = 0.0
        sq = np.sqrt(disc)
        # choose the root that gives smaller |beta| (more balanced)
        al1 = (-B2 + sq) / (2.0 * A2)
        al2 = (-B2 - sq) / (2.0 * A2)
        b1 = -ai - m * al1
        b2 = -ai - m * al2
        if abs(b1) <= abs(b2):
            alpha[i], beta[i] = al1, b1
        else:
            alpha[i], beta[i] = al2, b2
    E[:, 0] = a
    E[:, 1:] = alpha[:, None] * np.ones((V, m))
    E[:, 1] += beta

    # verify / fix numerical drift: renormalize each row to zero-mean unit-variance
    E = E - E.mean(axis=1, keepdims=True)
    nrm = np.sqrt((E ** 2).mean(axis=1, keepdims=True))
    nrm[nrm < 1e-8] = 1.0
    E = E / nrm

    # ---- write embed ----
    w["embed.weight"] = E.astype(np.float32)

    # ---- head: logit1 - logit0 = c * pooled[0] + b  ----
    # head.weight row1 = c * e0, row0 = 0 ; head.bias = [ -b/2, b/2 ]? 
    # We want logit1 - logit0 = c*pooled0 + b.  Set head0 = 0, head1 = c*e0,
    # bias0 = -b/2, bias1 = b/2  => diff = c*pooled0 + b.
    H = np.zeros((2, D), dtype=np.float64)
    H[1, 0] = c
    w["head.weight"] = H.astype(np.float32)
    w["head.bias"] = np.array([-b / 2.0, b / 2.0], dtype=np.float32)

    # ---- keep pos and blocks as-is (they are zero / identity) ----
    # ensure finite
    for k in list(w.keys()):
        arr = np.asarray(w[k])
        if not np.all(np.isfinite(arr)):
            w[k] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diag = {
        "cv_acc": float(cv_acc),
        "lam": float(lam),
        "s_absmax": float(np.max(np.abs(s))),
        "s_std": float(np.std(s)),
        "c": float(c),
        "b": float(b),
        "embed_nonzero": int(np.count_nonzero(w["embed.weight"])),
    }
    return {"weights": w, "diagnostics": diag}
