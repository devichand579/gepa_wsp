import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)          # (4000,32) frozen head input
    y = np.asarray(f["y"], dtype=np.int64)                   # (4000,)
    ids = np.asarray(f["x"], dtype=np.int64)                 # (4000,64)
    N, D = X.shape
    diag = {}

    def acc_of(scores):
        pred = (scores[:, 1] > scores[:, 0]).astype(np.int64)
        return float((pred == y).mean())

    # ---------- candidate 1: refit head on frozen features ----------
    # antisymmetric parameterisation: logits = [g, -g] with g = wf @ x + bf
    Ypm = np.where(y == 1, 1.0, -1.0)                        # (N,)
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc.T @ Xc
    cands = []

    # current head as baseline
    hw = np.asarray(w["head.weight"], dtype=np.float64)
    hb = np.asarray(w["head.bias"], dtype=np.float64)
    w_cur = 0.5 * (hw[1] - hw[0])
    b_cur = 0.5 * (hb[1] - hb[0])
    cands.append(("current", w_cur, b_cur))

    # ridge to +-1 targets, several lambdas
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        wf = np.linalg.solve(G + lam * np.eye(D), Xc.T @ Ypm)
        bf = -float(Xc.mean(axis=0) @ wf)
        cands.append((f"ridge{lam}", wf, bf))

    # LDA (pooled covariance) direction
    m0 = X[y == 0].mean(axis=0)
    m1 = X[y == 1].mean(axis=0)
    S0 = np.cov(X[y == 0].T) + np.cov(X[y == 1].T)
    try:
        wl = np.linalg.solve(S0 + 1e-3 * np.eye(D), m1 - m0)
        bl = -float(0.5 * (m0 + m1) @ wl)
        cands.append(("lda", wl, bl))
    except np.linalg.LinAlgError:
        pass

    # threshold recalibration of the current direction (1-D closed form)
    s_cur = X @ w_cur + b_cur
    order = np.argsort(s_cur)
    ys = y[order]
    # best threshold on the sorted scores
    best_t, best_a = 0.0, -1.0
    for i in range(N + 1):
        # predict 1 for indices >= i
        a = (ys[i:].sum() + (i - ys[:i].sum())) / N
        if a > best_a:
            best_a = a
            best_t = s_cur[order[i]] if i < N else s_cur[order[-1]] + 1.0
    cands.append(("thresh", w_cur, best_t))

    # ---------- candidate 2: exact linear BOW ----------
    # token-count matrix (mean over tokens), fit S (2000,2) by ridge
    V = 2000
    counts = np.zeros((N, V), dtype=np.float64)
    mask = ids != 0
    for i in range(N):
        row = ids[i][mask[i]]
        if row.size:
            np.add.at(counts[i], row, 1.0 / row.size)
    Yoh = np.zeros((N, 2), dtype=np.float64)
    Yoh[np.arange(N), y] = 1.0
    C = counts - counts.mean(axis=0, keepdims=True)
    Gc = C.T @ C
    best_bow = None
    for lam in [1e-2, 1e-1, 1.0, 10.0]:
        S = np.linalg.solve(Gc + lam * np.eye(V), C.T @ (Yoh - Yoh.mean(axis=0)))
        b = Yoh.mean(axis=0) - counts.mean(axis=0) @ S
        sc = counts @ S + b
        a = acc_of(sc)
        if best_bow is None or a > best_bow[0]:
            best_bow = (a, S, b, lam)

    # ---------- evaluate candidates on the pool ----------
    best = None
    for name, wf, bf in cands:
        sc = np.stack([X @ wf + bf, -(X @ wf + bf)], axis=1)
        a = acc_of(sc)
        if best is None or a > best[0]:
            best = (a, name, wf, bf)
    diag["head_candidates"] = {n: round(acc_of(np.stack([X @ wf + bf, -(X @ wf + bf)], 1)), 4)
                               for n, wf, bf in cands}
    diag["bow_acc"] = round(best_bow[0], 4) if best_bow else None

    # choose between refit-head and BOW construction
    use_bow = best_bow is not None and best_bow[0] > best[0] + 1e-4

    if use_bow:
        _, S, b, lam = best_bow
        # build embed rows: zero-mean, unit-norm, so LN(v)=v exactly
        a_scale = 1.0
        # keep |u|,|v| small enough for the quadratic to be solvable
        smax = max(np.abs(S).max(), 1e-6)
        a_scale = max(1.0, smax / 3.0)
        E = np.zeros((V, D), dtype=np.float64)
        for i in range(V):
            u = S[i, 0] / a_scale
            v = S[i, 1] / a_scale
            m = u + v
            R = D - u * u - v * v
            disc = 870.0 * R - 29.0 * m * m
            if disc < 0:
                disc = 0.0
            c = (-29.0 * m + np.sqrt(disc)) / 870.0
            d = -m - 29.0 * c
            row = np.empty(D)
            row[0] = u
            row[1] = v
            row[2:31] = c
            row[31] = d
            # renormalise exactly to zero-mean / unit-norm
            row = row - row.mean()
            nrm = np.sqrt((row * row).sum())
            if nrm > 1e-9:
                row = row * (np.sqrt(D) / nrm)
            E[i] = row
        w["embed.weight"] = E.astype(np.float32)
        hw_new = np.zeros((2, D), dtype=np.float64)
        hw_new[0, 0] = a_scale
        hw_new[1, 1] = a_scale
        w["head.weight"] = hw_new.astype(np.float32)
        w["head.bias"] = b.astype(np.float32)
        diag["chosen"] = "bow"
        diag["bow_lambda"] = lam
        diag["bow_scale"] = a_scale
    else:
        _, name, wf, bf = best
        hw_new = np.stack([-wf, wf], axis=0)
        w["head.weight"] = hw_new.astype(np.float32)
        w["head.bias"] = np.array([-bf, bf], dtype=np.float32)
        diag["chosen"] = name

    diag["pool_acc_chosen"] = round(best_bow[0] if use_bow else best[0], 4)
    diag["class1_rate"] = round(float((y == 1).mean()), 4)
    return {"weights": w, "diagnostics": diag}
