import numpy as np

def edit(w, f):
    X = f["x"].reshape(-1, 784).astype(np.float64)
    Y = f["y"].astype(np.int64)
    N = X.shape[0]
    W1 = w["fc1.weight"].astype(np.float64)
    b1 = w["fc1.bias"].astype(np.float64)
    W2 = w["fc2.weight"].astype(np.float64)
    b2 = w["fc2.bias"].astype(np.float64)

    Y1h = np.zeros((N, 10))
    Y1h[np.arange(N), Y] = 1.0

    # ---------- current forward (sanity check against cached features) ----------
    A = X @ W1.T + b1
    H = np.maximum(A, 0.0)
    cached = f["features"].astype(np.float64)
    feat_err = float(np.abs(H - cached).max())

    # ---------- helper: ridge solve with target prior (closed form) ----------
    def ridge_solve(Xd, T, lam, Wprior, bprior):
        n, d = Xd.shape
        Xc = Xd - Xd.mean(0)
        mu = Xd.mean(0)
        A_ = Xc
        G = A_.T @ A_ + lam * np.eye(d)
        B = A_.T @ (T - T.mean(0))
        Wn = np.linalg.solve(G, B)
        bn = T.mean(0) - mu @ Wn
        # shrink toward prior
        Wn = Wprior + Wn
        bn = bprior + bn
        return Wn, bn

    def acc_ce(Wn, bn, Xd, T, ytrue):
        L = Xd @ Wn.T + bn
        L = L - L.max(1, keepdims=True)
        P = np.exp(L)
        P /= P.sum(1, keepdims=True)
        pred = P.argmax(1)
        acc = float((pred == ytrue).mean())
        ce = float(-np.log(np.clip(P[np.arange(len(ytrue)), ytrue], 1e-12, None)).mean())
        return acc, ce

    # ---------- augmentation: small shifts (zero padded) ----------
    def shift_batch(Xb, dy, dx):
        out = np.zeros_like(Xb)
        ys = slice(max(0, dy), 28 + min(0, dy))
        yd = slice(max(0, -dy), 28 + min(0, -dy))
        xs = slice(max(0, dx), 28 + min(0, dx))
        xd = slice(max(0, -dx), 28 + min(0, -dx))
        out[:, yd, xd] = Xb[:, ys, xs]
        return out

    Xim = X.reshape(-1, 28, 28)
    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    Xa_list = []
    for dy, dx in shifts:
        if dy == 0 and dx == 0:
            Xa_list.append(Xim)
        else:
            Xa_list.append(shift_batch(Xim, dy, dx))
    Xa = np.concatenate([a.reshape(N, -1) for a in Xa_list], 0)
    Aa = Xa @ W1.T + b1
    Ha = np.maximum(Aa, 0.0)
    Ta = np.tile(Y1h, (len(shifts), 1))

    # ---------- candidate heads on current features ----------
    cands = []  # (name, W2, b2, acc, ce)

    # incumbent
    cands.append(("incumbent", W2, b2) + acc_ce(W2, b2, H, Y1h, Y))

    # ridge on features (no prior), lambda grid
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        Wn, bn = ridge_solve(H, Y1h, lam, np.zeros((10, 128)), np.zeros(10))
        cands.append(("ridge_l%.4g" % lam, Wn, bn) + acc_ce(Wn, bn, H, Y1h, Y))

    # ridge toward current fc2 (shrinkage of incumbent), lambda grid
    for lam in [1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0]:
        Wn, bn = ridge_solve(H, Y1h, lam, W2, b2)
        cands.append(("ridge_prior_l%.4g" % lam, Wn, bn) + acc_ce(Wn, bn, H, Y1h, Y))

    # ridge on augmented features, toward current fc2
    for lam in [1.0, 3.0, 10.0, 30.0, 100.0]:
        Wn, bn = ridge_solve(Ha, Ta, lam, W2, b2)
        cands.append(("ridge_aug_l%.4g" % lam, Wn, bn) + acc_ce(Wn, bn, H, Y1h, Y))

    # LDA / shared-covariance Gaussian classifier on features (shrinkage)
    def lda_solve(Xd, ytrue, shrink):
        d = Xd.shape[1]
        mus = np.zeros((10, d))
        Sw = np.zeros((d, d))
        for c in range(10):
            m = ytrue == c
            Xc = Xd[m]
            mus[c] = Xc.mean(0)
            Z = Xc - mus[c]
            Sw += Z.T @ Z
        Sw /= max(1, N - 10)
        tr = np.trace(Sw) / d
        Sw = (1 - shrink) * Sw + shrink * tr * np.eye(d)
        P = np.linalg.inv(Sw)
        Wn = mus @ P
        bn = -0.5 * np.sum(Wn * mus, 1) + np.log(np.bincount(ytrue, minlength=10) / N)
        return Wn, bn

    for sh in [0.01, 0.03, 0.1, 0.3, 0.6]:
        Wn, bn = lda_solve(H, Y, sh)
        cands.append(("lda_s%.3g" % sh, Wn, bn) + acc_ce(Wn, bn, H, Y1h, Y))

    # ---------- new hidden layer: re-threshold detector rows + keep others ----------
    # detector rows = rows whose hidden activation is class-selective
    act = H > 0
    det = []
    for j in range(128):
        col = act[:, j]
        if col.sum() < 5:
            continue
        cnt = np.bincount(Y[col], minlength=10)
        if cnt.max() / max(1, col.sum()) > 0.9:
            det.append(j)
    det = np.array(det, dtype=int)

    def build_hidden(thr_per_det, seed):
        rng = np.random.default_rng(seed)
        rows = []
        biases = []
        for j in det:
            col = A[:, j]
            pos = col[Y == np.argmax(np.bincount(Y[act[:, j]], minlength=10))]
            qs = np.quantile(pos, np.linspace(0.2, 0.8, thr_per_det)) if len(pos) > 5 else np.array([0.0])
            for t in np.atleast_1d(qs):
                rows.append(W1[j])
                biases.append(b1[j] - t)
        # remaining capacity: keep original non-detector rows (they carry extra info)
        used = len(rows)
        for j in range(128):
            if used >= 128:
                break
            if j in det:
                continue
            rows.append(W1[j])
            biases.append(b1[j])
            used += 1
        # if still short, add random projections of the whitened-ish pixel space
        while used < 128:
            v = rng.normal(size=784)
            v /= np.linalg.norm(v)
            rows.append(v * 0.05)
            biases.append(-0.5)
            used += 1
        Wn1 = np.array(rows[:128])
        bn1 = np.array(biases[:128])
        return Wn1, bn1

    def eval_newfc1(Wn1, bn1, lam_list, tag):
        Hn = np.maximum(X @ Wn1.T + bn1, 0.0)
        best = None
        for lam in lam_list:
            Wn, bn = ridge_solve(Hn, Y1h, lam, np.zeros((10, 128)), np.zeros(10))
            a, c = acc_ce(Wn, bn, Hn, Y1h, Y)
            if best is None or (a, -c) > (best[2], -best[3]):
                best = (Wn, bn, a, c, lam)
        Wn, bn, a, c, lam = best
        cands.append((tag, Wn, bn, a, c))
        return best

    if len(det) > 0:
        for ndet_thr in [2, 3, 5]:
            Wn1, bn1 = build_hidden(ndet_thr, 7)
            eval_newfc1(Wn1, bn1, [0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
                        "thresh%d" % ndet_thr)

    # ---------- honest 5-fold CV comparison of top candidates (closed-form folds) ----------
    rng = np.random.default_rng(0)
    fold = rng.permutation(N) % 5

    def cv_score(Wn, bn, Xd):
        # if candidate uses new fc1, Xd must be its features; here Xd = H (current features)
        tot = 0.0
        for k in range(5):
            tr = fold != k
            te = ~tr
            Wf, bf = ridge_solve(Xd[tr], Y1h[tr], 10.0, np.zeros((10, Xd.shape[1])), np.zeros(10))
            a, _ = acc_ce(Wf, bf, Xd[te], Y1h[te], Y[te])
            tot += a * te.sum()
        return tot / N

    # ---------- pick best ----------
    # rank by pool accuracy, tie-break by CE
    cands.sort(key=lambda t: (-t[3], t[4]))
    best = cands[0]
    name, Wn2, bn2, acc, ce = best

    # ---------- diagnostics ----------
    diag = {
        "feat_max_err": round(feat_err, 6),
        "n_detector_rows": int(len(det)),
        "detector_idx": det[:20].tolist(),
        "incumbent_acc": round(float((f["logits"].argmax(1) == Y).mean()), 4),
        "candidates": [(c[0], round(c[3], 4), round(c[4], 4)) for c in cands],
        "chosen": name,
        "chosen_acc": round(acc, 4),
        "chosen_ce": round(ce, 4),
        "fc2_absmax": round(float(np.abs(Wn2).max()), 4),
        "fc2_bias": [round(float(v), 4) for v in bn2],
        "logit_std": round(float((H @ Wn2.T + bn2).std()), 4),
    }

    # ---------- write back ----------
    out = {}
    for k, v in w.items():
        out[k] = np.array(v, dtype=np.float32, copy=True)
    out["fc2.weight"] = np.clip(Wn2, -1000, 1000).astype(np.float32)
    out["fc2.bias"] = np.clip(bn2, -1000, 1000).astype(np.float32)
    return {"weights": out, "diagnostics": diag}
