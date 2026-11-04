import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"]).astype(np.int64)
    n, d = X.shape
    C = 10

    # ---------- class statistics ----------
    mu = np.zeros((C, d))
    for c in range(C):
        m = y == c
        if m.sum() > 0:
            mu[c] = X[m].mean(0)
    mu_bar = mu.mean(0)
    Sw = np.zeros((d, d))
    for c in range(C):
        m = y == c
        if m.sum() > 0:
            Z = X[m] - mu[c]
            Sw += Z.T @ Z
    Sw /= max(n - C, 1)

    # ---------- whitening (regularized eigendecomposition) ----------
    ev, U = np.linalg.eigh(Sw)
    ev = np.maximum(ev, 0.0)
    tr = ev.sum()
    ridge = 0.05 * tr / d
    ev_r = ev + ridge
    keep = ev_r > (1e-6 * tr / d)
    Vk = U[:, keep]
    dk = ev_r[keep]
    Wk = Vk / np.sqrt(dk)                      # d x k
    Z = X @ Wk                                 # whitened coords
    k = Z.shape[1]

    # class means in whitened space
    mz = np.zeros((C, k))
    for c in range(C):
        m = y == c
        if m.sum() > 0:
            mz[c] = Z[m].mean(0)
    mz_bar = mz.mean(0)
    Mb = (mz - mz_bar).T @ (mz - mz_bar)       # between-class scatter (whitened)
    eb, Ub = np.linalg.eigh(Mb)
    order = np.argsort(eb)[::-1]
    Ub = Ub[:, order]
    # Fisher score of each whitened direction
    fisher = np.zeros(k)
    for j in range(k):
        v = Ub[:, j]
        num = 0.0
        for c in range(C):
            num += (v @ (mz[c] - mz_bar)) ** 2
        fisher[j] = num / max(v @ v, 1e-12)
    fisher = np.nan_to_num(fisher)

    # ---------- direction bank ----------
    dirs = []          # list of 784-vectors
    tags = []
    # (a) one-vs-rest LDA directions (whitened class-mean directions)
    for c in range(C):
        v = Wk @ (mz[c] - mz_bar)
        nrm = np.linalg.norm(v)
        if nrm > 1e-9:
            dirs.append(v / nrm); tags.append(("lda", c, -1))
    # (b) pairwise LDA directions
    pairs = []
    for a in range(C):
        for b in range(a + 1, C):
            v = Wk @ (mz[a] - mz[b])
            nrm = np.linalg.norm(v)
            if nrm > 1e-9:
                pairs.append((a, b, v / nrm))
    pairs.sort(key=lambda t: -np.linalg.norm(t[2]))
    for a, b, v in pairs:
        dirs.append(v); tags.append(("pair", a, b))
    # (c) top Fisher whitened-PCA directions
    for j in range(min(k, 40)):
        v = Wk @ Ub[:, j]
        nrm = np.linalg.norm(v)
        if nrm > 1e-9:
            dirs.append(v / nrm); tags.append(("pca", j, -1))

    Dm = np.array(dirs)                        # ndir x 784
    ndir = Dm.shape[0]

    # ---------- ridge with closed-form LOO ----------
    Y1 = np.zeros((n, C))
    Y1[np.arange(n), y] = 1.0

    def ridge_fit(F, lam):
        G = F.T @ F
        b = F.T @ Y1
        A = G + lam * np.eye(F.shape[1])
        try:
            Wi = np.linalg.inv(A)
        except Exception:
            Wi = np.linalg.pinv(A)
        return Wi @ b, Wi, G

    def loo_select(F):
        G = F.T @ F
        b = F.T @ Y1
        tr = np.trace(G) / F.shape[1]
        lams = tr * np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0])
        best = None
        for lam in lams:
            A = G + lam * np.eye(F.shape[1])
            try:
                Wi = np.linalg.inv(A)
            except Exception:
                continue
            Wc = Wi @ b
            H = F @ Wi @ F.T
            dg = np.clip(np.diag(H), 0.0, 0.999)
            err = ((Y1 - H @ Y1) / (1.0 - dg)[:, None])
            mse = float(np.mean(err ** 2))
            if best is None or mse < best[0]:
                best = (mse, lam, Wc, Wi)
        return best

    def cv_acc(F, lam, folds=5):
        idx = np.arange(n)
        acc = 0.0
        for kk in range(folds):
            te = idx % folds == kk
            tr = ~te
            G = F[tr].T @ F[tr]
            b = F[tr].T @ Y1[tr]
            A = G + lam * np.eye(F.shape[1])
            try:
                Wi = np.linalg.inv(A)
            except Exception:
                Wi = np.linalg.pinv(A)
            Wc = Wi @ b
            pred = F[te] @ Wc
            acc += float(np.mean(np.argmax(pred, 1) == y[te]))
        return acc / folds

    # ---------- candidate designs ----------
    # each design: list of (dir_index, n_thresholds, include_raw_pair)
    designs = []
    designs.append(("lda+pair", list(range(10)) + list(range(10, 10 + 30)), 1, True))
    designs.append(("lda+pair2", list(range(10)) + list(range(10, 10 + 45)), 1, True))
    designs.append(("lda+pair3", list(range(10)) + list(range(10, 10 + 45)), 2, True))
    designs.append(("lda+pca", list(range(10)) + list(range(55, 55 + 20)), 2, True))
    designs.append(("lda+pair+pca", list(range(10)) + list(range(10, 10 + 25)) + list(range(55, 55 + 12)), 1, True))

    results = []
    for name, dsel, nthr, raw in designs:
        cols = []
        meta = []
        for di in dsel:
            p = X @ Dm[di]
            qs = [0.5, 0.8, 0.92]
            ts = []
            for j in range(nthr):
                q = qs[min(j, len(qs) - 1)]
                ts.append(float(np.quantile(p, q)))
            if raw:
                cols.append(np.maximum(p, 0.0)); meta.append((di, None, 1))
                cols.append(np.maximum(-p, 0.0)); meta.append((di, None, -1))
            for t in ts:
                cols.append(np.maximum(p - t, 0.0)); meta.append((di, t, 1))
                cols.append(np.maximum(t - p, 0.0)); meta.append((di, t, -1))
        F = np.array(cols).T
        if F.shape[1] > 128:
            # keep the 128 most useful columns via one-shot ridge magnitude selection
            mse, lam, Wc, Wi = loo_select(F)
            score = np.abs(Wc).sum(0)
            keep = np.argsort(score)[::-1][:128]
            keep = np.sort(keep)
            F = F[:, keep]
            meta = [meta[i] for i in keep]
        if F.shape[1] > 128:
            F = F[:, :128]; meta = meta[:128]
        mse, lam, Wc, Wi = loo_select(F)
        a = cv_acc(F, lam)
        results.append((a, name, F, meta, lam, mse))

    results.sort(key=lambda t: -t[0])
    best_acc, best_name, F, meta, lam, loo_mse = results[0]

    # ---------- build fc1 / fc2 ----------
    nf = F.shape[1]
    mu_f = F.mean(0)
    sd_f = F.std(0)
    sd_f = np.where(sd_f < 1e-8, 1.0, sd_f)
    Fs = (F - mu_f) / sd_f

    W1 = np.zeros((128, 784))
    B1 = np.zeros(128)
    for j, (di, t, sgn) in enumerate(meta):
        if t is None:
            W1[j] = sgn * Dm[di]
            B1[j] = 0.0
        else:
            W1[j] = sgn * Dm[di]
            B1[j] = -sgn * t
    # fold standardization into fc1: relu(W1 x + B1) = sd_f * Fs + sd_f*mu_f
    # => standardized feature = relu((W1/sd)x + B1/sd) ; we need (F - mu)/sd
    # relu(W1 x + B1)/sd = relu((W1/sd) x + B1/sd) = F/sd ; subtract mu/sd via bias
    W1 = W1 / sd_f[:, None]
    B1 = B1 / sd_f - mu_f / sd_f

    # fc2: ridge on standardized features
    G = Fs.T @ Fs
    b = Fs.T @ Y1
    A = G + lam * np.eye(nf)
    try:
        Wi = np.linalg.inv(A)
    except Exception:
        Wi = np.linalg.pinv(A)
    W2_full = Wi @ b
    # fold feature mean into fc2 bias: Fs = (F-mu)/sd  =>  F = Fs*sd + mu
    W2 = W2_full * sd_f[None, :]
    b2 = Y1.mean(0) - (W2_full @ (mu_f / sd_f))
    # dead units for the rest
    if nf < 128:
        W1[nf:] = 0.0
        B1[nf:] = -1.0
    W2 = np.zeros((10, 128))
    W2[:, :nf] = W2_full * sd_f[None, :]
    b2 = np.zeros(10)
    b2 = Y1.mean(0) - (W2_full @ (mu_f / sd_f))

    # ---------- verify with explicit forward pass ----------
    H = np.maximum(X @ W1.T + B1[None, :], 0.0)
    L = H @ W2.T + b2[None, :]
    pred = np.argmax(L, 1)
    acc = float(np.mean(pred == y))
    lse = np.log(np.sum(np.exp(L - L.max(1, keepdims=True)), 1)) + L.max(1) - L[np.arange(n), y]
    ce = float(np.mean(lse))

    # ---------- write out ----------
    out = {}
    for kk, vv in w.items():
        out[kk] = np.array(vv, dtype=np.float32).copy()
    out["fc1.weight"] = np.clip(W1, -1000.0, 1000.0).astype(np.float32)
    out["fc1.bias"] = np.clip(B1, -1000.0, 1000.0).astype(np.float32)
    out["fc2.weight"] = np.clip(W2, -1000.0, 1000.0).astype(np.float32)
    out["fc2.bias"] = np.clip(b2, -1000.0, 1000.0).astype(np.float32)

    diag = {
        "design": best_name,
        "cv_acc": round(float(best_acc), 4),
        "train_acc": round(acc, 4),
        "train_ce": round(ce, 4),
        "loo_mse": round(float(loo_mse), 5),
        "ridge_lambda": float(lam),
        "n_units_used": int(nf),
        "all_cv": {r[1]: round(float(r[0]), 4) for r in results},
        "fc1_abs_max": round(float(np.abs(out["fc1.weight"]).max()), 4),
        "fc2_abs_max": round(float(np.abs(out["fc2.weight"]).max()), 4),
        "fc1_bias_min_max": [round(float(out["fc1.bias"].min()), 3), round(float(out["fc1.bias"].max()), 3)],
        "fc2_bias": [round(float(v), 3) for v in out["fc2.bias"]],
        "finite": bool(np.all(np.isfinite(out["fc1.weight"])) and np.all(np.isfinite(out["fc2.weight"]))),
    }
    return {"weights": out, "diagnostics": diag}
