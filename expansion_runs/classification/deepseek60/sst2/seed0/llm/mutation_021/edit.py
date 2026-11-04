import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- normalised unigram count features (length-normalised) ----
    mask = (X != 0).astype(np.float32)
    length = np.maximum(mask.sum(1), 1.0)
    # count matrix
    C = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        row = X[i]
        nz = row[row != 0]
        if nz.size:
            np.add.at(C[i], nz, 1.0)
    C /= length[:, None]

    # ---- closed-form ridge on +/-1 targets, alpha chosen by K-fold CV ----
    t = 2.0 * y.astype(np.float32) - 1.0
    Cmean = C.mean(0, keepdims=True)
    Cc = C - Cmean
    # small ridge on the Gram matrix (V=2000, N=4000) -> use dual form
    alphas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)
    best = (None, -1.0)
    for a in alphas:
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Ctr = Cc[tr]; Cte = Cc[te]
            ttr = t[tr]
            # dual ridge: solve (Ctr Ctr^T + a I) b = ttr
            G = Ctr @ Ctr.T + a * np.eye(tr.size, dtype=np.float32)
            b = np.linalg.solve(G, ttr)
            pred = Cte @ (Ctr.T @ b)
            accs.append(np.mean((pred > 0) == (t[te] > 0)))
        m = float(np.mean(accs))
        if m > best[1]:
            best = (a, m)
    alpha = best[0]

    # final fit on all data
    G = Cc @ Cc.T + alpha * np.eye(N, dtype=np.float32)
    b = np.linalg.solve(G, t)
    wvec = Cc.T @ b                      # [V] per-token scores (unnormalised)
    intercept = float(-(Cmean @ wvec))   # bias so that score = wvec·C + intercept

    # ---- encode into embed[:,0] and head ----
    # We want s(v) = head_diff · LN(embed[v]) with head_diff = c * e0.
    # Choose c so that per-token scores are within the realisable range.
    s = wvec.astype(np.float32)
    smax = float(np.max(np.abs(s))) + 1e-8
    # realisable |LN(embed)_0| <= sqrt(D-1) ~ 5.57 ; use 0.9 of it
    c = 0.9 * np.sqrt(D - 1) / smax
    s_scaled = (s * c).astype(np.float32)   # desired dim-0 of LN(embed[v])

    # Build embed[v] with only dims 0 and 1 nonzero such that LN(embed[v])_0 = s_scaled[v].
    # embed[v] = (a, b, 0...0).  mean=(a+b)/D, var=((a-m)^2+(b-m)^2+(D-2)m^2)/D.
    # LN_0 = (a-m)/sqrt(var).  Solve for (a,b) given target z0=s_scaled[v].
    # Parametrise: a = m + z0*sigma, and choose b so that mean is m and variance is 1.
    # Simpler: set b = -a/(D-1) so mean=0, then var = (a^2 + a^2/(D-1))/D = a^2*(D/(D-1))/D = a^2/(D-1).
    # Then LN_0 = a/sqrt(a^2/(D-1)) = sign(a)*sqrt(D-1).  Not free -> instead use two free dims.
    # Use: embed = (a, b, r, r, ..., r) with r chosen so mean=0: r = -(a+b)/(D-2).
    # Then var = (a^2 + b^2 + (D-2) r^2)/D.  We want LN_0 = (a - 0)/sqrt(var) = z0
    # and LN_1 = b/sqrt(var) free.  Set sqrt(var)=1 by scaling: choose a = z0, b = 0,
    # r = -z0/(D-2).  Then var = (z0^2 + (D-2)*z0^2/(D-2)^2)/D = (z0^2 + z0^2/(D-2))/D
    #            = z0^2 * (D-1)/(D-2) / D.  Not 1 in general.
    # Robust approach: solve numerically per token for (a,b) with mean 0 and unit variance.
    embed = np.zeros((V, D), dtype=np.float32)
    Dm = D - 2
    for v in range(V):
        z0 = float(s_scaled[v])
        # want a = z0 * sigma, mean 0 -> b = -a - (D-2)*r ; and var = 1.
        # Set r = 0 for simplicity -> b = -a, then var = (a^2 + b^2)/D = 2a^2/D.
        # LN_0 = a / sqrt(2a^2/D) = sign(a)*sqrt(D/2).  Again fixed magnitude.
        # So we need r != 0 to tune variance independently of a.
        # Let a = z0 * sigma, and choose sigma via var equation with r = -(a+b)/(D-2), b free.
        # Take b = a (symmetric) -> r = -2a/(D-2).
        # var = (a^2 + a^2 + (D-2)*(4a^2/(D-2)^2))/D = (2a^2 + 4a^2/(D-2))/D
        #     = a^2*(2 + 4/(D-2))/D.
        # LN_0 = a/sqrt(var) = 1/sqrt((2+4/(D-2))/D)  -> fixed.  Not free.
        # Conclusion: with only 2 free dims the magnitude of LN_0 is fixed.
        # Therefore use 3 free dims (0,1,2) and set dims 3.. to a constant.
        # General construction: pick a unit vector u in the zero-mean subspace with u_0 = 1,
        # then LN(embed) = z0 * u works if ||u||^2 = D (unit variance) and mean(u)=0.
        # Build u: u_0 = 1, u_1 = -1, rest 0 -> mean 0, ||u||^2 = 2.  Scale to ||u||^2 = D:
        #   u = (1, -1, 0...)/sqrt(2) * sqrt(D).  Then u_0 = sqrt(D/2).  Fixed again.
        # The real fix: allow head_diff to be a general vector, not just e0.
        # Then s(v) = head_diff · z(v) and we can pick z(v) = z0(v)*u + w(v) with u fixed.
        # Simpler: just set head_diff = c*e0 and accept the fixed magnitude by scaling
        # the *scores* to lie on the achievable circle.  We do this by choosing
        # embed[v] = (a, b, 0...) with a,b solving mean 0 and unit variance AND a = z0.
        # mean 0 -> b = -a.  var = 2a^2/D = 1 -> a = sqrt(D/2).  So a is fixed.
        # => Instead encode score in the *angle*: set embed[v] = (cos, sin, 0...) scaled.
        #    mean = (cos+sin)/D, var = ((cos-m)^2+(sin-m)^2+(D-2)m^2)/D.
        #    LN_0 = (cos-m)/sqrt(var).  As (cos,sin) ranges over the circle this traces
        #    a bounded interval; we map our desired z0 into that interval.
        # Practical: use dims 0 and 1 as (cos θ, sin θ) and scale so LN is well-defined.
        # We simply set embed[v] = (z0, 0, 0, ..., 0) and rely on the fact that with
        # head_diff = c*e0 the realised score is sign(z0)*sqrt(D-1)*c.  To keep the
        # *magnitude* information we instead put the score into head_diff via a
        # rank-1 correction: we set head.weight directly from the fitted wvec.
        # --- final, correct and simple encoding ---
        embed[v, 0] = z0
        # fill remaining dims so that row is zero-mean and unit-variance:
        # choose dims 1..D-1 all equal to q, with z0 + (D-1) q = 0 -> q = -z0/(D-1)
        q = -z0 / (D - 1)
        embed[v, 1:] = q
    # Now LN(embed[v]) = (embed[v] - 0)/sigma_v with sigma_v = |z0|*sqrt(D/(D-1)) ... 
    # which again collapses magnitude.  To preserve magnitude we therefore ALSO
    # scale head_diff per-token is impossible (head is shared).  The only way to
    # keep magnitude is to make sigma_v constant, i.e. embed rows have equal norm.
    # We achieve this by normalising each row to unit norm BEFORE LayerNorm, which
    # is exactly what LayerNorm does anyway -> magnitude is inherently lost.
    # Hence the correct realisation is: the model IS a sign model unless we use
    # head_diff with a component along the *mean* direction.  We therefore encode
    # the linear scores directly into head.weight and keep embed rows as fixed
    # zero-mean/unit-variance basis vectors, one per token, chosen so that
    # head_diff · LN(embed[v]) = s(v).  Since LN(embed[v]) has fixed norm sqrt(D),
    # we can realise any s(v) with |s(v)| <= ||head_diff||*sqrt(D) by choosing
    # LN(embed[v]) = s(v)/||head_diff||^2 * head_diff + orthogonal part.
    # Build that explicitly:
    hd = np.zeros(D, dtype=np.float32)
    hd[0] = 1.0
    hd[1] = 1.0
    hd = hd - hd.mean()                 # zero-mean direction
    hd = hd / np.linalg.norm(hd)
    # choose head_diff magnitude so max |s| fits
    smax2 = float(np.max(np.abs(s))) + 1e-8
    scale = 0.9 * np.sqrt(D) / smax2
    head_diff = (hd * scale).astype(np.float32)   # [D]
    # for each token build z(v) = s(v)/scale * hd + orthogonal, zero-mean unit-variance
    embed = np.zeros((V, D), dtype=np.float32)
    # orthogonal basis in zero-mean subspace, orthogonal to hd
    # build a fixed orthonormal set spanning the complement
    B = np.eye(D, dtype=np.float32)
    B = B - B.mean(0, keepdims=True)             # project to zero-mean subspace
    # Gram-Schmidt against hd
    comp = []
    for j in range(D):
        vj = B[:, j].copy()
        vj -= hd * float(hd @ vj)
        for u in comp:
            vj -= u * float(u @ vj)
        n = np.linalg.norm(vj)
        if n > 1e-6:
            comp.append(vj / n)
    comp = np.array(comp, dtype=np.float32)      # [D-1, D]
    for v in range(V):
        target = float(s[v]) / scale             # component along hd
        # z = target*hd + sqrt(D - target^2) * (some unit vector in comp)
        rem2 = D - target * target
        if rem2 < 0:
            rem2 = 0.0
        rem = np.sqrt(rem2)
        # distribute the remaining norm equally over the complement basis
        z = target * hd
        if comp.shape[0] > 0 and rem > 0:
            z = z + rem / np.sqrt(comp.shape[0]) * comp.sum(0)
        embed[v] = z.astype(np.float32)
    # head: logit = head @ pooled ; we want logit1-logit0 = head_diff · pooled + intercept
    head = np.zeros((2, D), dtype=np.float32)
    head[1] = head_diff
    head[0] = -head_diff
    bias = np.array([-intercept / 2.0, intercept / 2.0], dtype=np.float32)

    w["embed.weight"] = embed.astype(np.float32)
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = bias.astype(np.float32)

    # keep everything else as-is (already zero/identity) but ensure finite & bounded
    for k, arr in list(w.items()):
        a = np.asarray(arr, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.clip(a, -1000.0, 1000.0)
        w[k] = a.astype(np.float32)

    diag = {
        "alpha": float(alpha),
        "cv_acc": float(best[1]),
        "s_absmax": float(np.max(np.abs(s))),
        "scale": float(scale),
        "intercept": float(intercept),
        "embed_nonzero": int(np.count_nonzero(w["embed.weight"])),
    }
    return {"weights": w, "diagnostics": diag}
