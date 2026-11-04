import numpy as np

def edit(w, f):
    X = np.asarray(f["x"])            # [N,64] int64
    y = np.asarray(f["y"]).astype(np.float64)   # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]    # 2000
    D = w["embed.weight"].shape[1]    # 32

    # ---- build length-normalized unigram count features ----
    # counts[v] = (# occurrences of token v) / (# non-pad tokens)
    mask = (X != 0)
    lens = mask.sum(axis=1).astype(np.float64)
    lens[lens == 0] = 1.0
    counts = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        m = mask[i]
        if m.any():
            np.add.at(counts[i], row[m], 1.0)
        counts[i] /= lens[i]

    # target: +/-1 for the two classes (ridge regression on class label)
    t = 2.0 * y - 1.0

    # ---- closed-form ridge with K-fold CV to pick alpha ----
    def ridge_fit(A, b, alpha):
        # solve (A^T A + alpha I) w = A^T b  (A is [n,V])
        G = A.T @ A
        G[np.diag_indices_from(G)] += alpha
        rhs = A.T @ b
        try:
            return np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(G, rhs, rcond=None)[0]

    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    K = 5
    folds = np.array_split(perm, K)
    alphas = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    best_alpha, best_cv = alphas[0], -1.0
    for alpha in alphas:
        correct = 0
        for k in range(K):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(K) if j != k])
            wv = ridge_fit(counts[tr], t[tr], alpha)
            pred = counts[va] @ wv
            correct += int(((pred > 0).astype(np.float64) == y[va]).sum())
        cv = correct / N
        if cv > best_cv:
            best_cv, best_alpha = cv, alpha

    # final fit on all data
    wv = ridge_fit(counts, t, best_alpha)

    # ---- encode per-token scores into embed[:,0] ----
    # We want s(v) = wv[v] realized as (head1-head0)·embed[v].
    # Choose head direction d = (c, 0, ..., 0) with c chosen so embed[:,0]
    # stays within the feasible range for a zero-mean/unit-variance vector.
    s = wv.astype(np.float64)
    s = s - s.mean()                       # center (intercept goes to head.bias)
    smax = np.max(np.abs(s)) + 1e-12
    # feasible |a| < sqrt(D); pick c so that max |a| = 0.9*sqrt(D)
    c = 0.9 * np.sqrt(D) / smax
    a = s / c                              # embed[:,0] values, |a| <= 0.9*sqrt(D)

    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = a
    # fill remaining dims so each row is zero-mean and unit-variance
    # row sum must be -a ; sum of squares must be D - a^2
    rem = D - 1
    # choose remaining entries equal to x (all same) => rem*x = -a => x = -a/rem
    # then sum of squares = rem*x^2 = a^2/rem ; need a^2/rem = D - a^2
    # => a^2 (1 + 1/rem) = D => a^2 = D*rem/(rem+1) = D*(D-1)/D = D-1
    # That only works for a specific a. Instead use a two-value construction.
    # General: pick b on dims 1..D-1 with sum = -a and sum of squares = D - a^2.
    # Let dims 1..D-2 be equal to p, dim D-1 be q.
    # (D-2)*p + q = -a ; (D-2)*p^2 + q^2 = D - a^2
    # Solve: q = -a - (D-2)p ; substitute.
    m = D - 2
    # (m)*p^2 + (a + m*p)^2 = D - a^2
    # m p^2 + a^2 + 2 a m p + m^2 p^2 = D - a^2
    # (m + m^2) p^2 + 2 a m p + 2 a^2 - D = 0
    A2 = m + m * m
    B2 = 2.0 * a * m
    C2 = 2.0 * a * a - D
    disc = B2 * B2 - 4.0 * A2 * C2
    disc = np.maximum(disc, 0.0)
    p = (-B2 + np.sqrt(disc)) / (2.0 * A2)
    q = -a - m * p
    E[:, 1:D - 1] = p[:, None]
    E[:, D - 1] = q

    # renormalize each row exactly to zero-mean / unit-variance
    E = E - E.mean(axis=1, keepdims=True)
    std = E.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    E = E / std

    # ---- set head so that head1 - head0 = (c, 0, ..., 0) ----
    head = np.zeros((2, D), dtype=np.float64)
    head[1, 0] = c / 2.0
    head[0, 0] = -c / 2.0
    # intercept from the centering of s
    b0 = float((wv - s).mean())  # = s.mean() before centering
    head_bias = np.array([-b0 / 2.0, b0 / 2.0], dtype=np.float64)

    # ---- write back (float32, finite, |.|<=1000) ----
    w["embed.weight"] = np.clip(E.astype(np.float32), -1000.0, 1000.0)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for name in ["block0.norm1.weight", "block0.norm2.weight",
                 "block1.norm1.weight", "block1.norm2.weight",
                 "norm.weight"]:
        w[name] = np.ones_like(w[name], dtype=np.float32)
    for name in ["block0.norm1.bias", "block0.norm2.bias",
                 "block1.norm1.bias", "block1.norm2.bias",
                 "norm.bias"]:
        w[name] = np.zeros_like(w[name], dtype=np.float32)
    for name in ["block0.qkv.weight", "block0.qkv.bias",
                 "block0.proj.weight", "block0.proj.bias",
                 "block0.ff1.weight", "block0.ff1.bias",
                 "block0.ff2.weight", "block0.ff2.bias",
                 "block1.qkv.weight", "block1.qkv.bias",
                 "block1.proj.weight", "block1.proj.bias",
                 "block1.ff1.weight", "block1.ff1.bias",
                 "block1.ff2.weight", "block1.ff2.bias"]:
        w[name] = np.zeros_like(w[name], dtype=np.float32)
    w["head.weight"] = np.clip(head.astype(np.float32), -1000.0, 1000.0)
    w["head.bias"] = np.clip(head_bias.astype(np.float32), -1000.0, 1000.0)

    diag = {
        "best_alpha": float(best_alpha),
        "cv_acc": float(best_cv),
        "s_absmax": float(np.max(np.abs(s))),
        "s_std": float(np.std(s)),
        "c": float(c),
        "b0": float(b0),
        "embed_nonzero": int(np.count_nonzero(w["embed.weight"])),
        "head_norm": float(np.linalg.norm(head)),
    }
    return {"weights": w, "diagnostics": diag}
