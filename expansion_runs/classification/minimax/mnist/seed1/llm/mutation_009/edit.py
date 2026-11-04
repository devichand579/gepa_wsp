import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(len(f["x"]), -1)   # [N,784]
    y = np.asarray(f["y"], dtype=np.int64)
    N, D = X.shape
    K = 10

    W1 = np.asarray(w["fc1.weight"], dtype=np.float64)
    b1 = np.asarray(w["fc1.bias"], dtype=np.float64)
    W2 = np.asarray(w["fc2.weight"], dtype=np.float64)
    b2 = np.asarray(w["fc2.bias"], dtype=np.float64)

    # ---------------- whitening / PCA ----------------
    mu = X.mean(axis=0)
    Xc = X - mu
    # economical SVD of centered data
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / max(N - 1, 1)
    order = np.argsort(ev)[::-1]
    ev = ev[order]; Vt = Vt[order]; S = S[order]
    keep = min(160, Vt.shape[0])
    ev = ev[:keep]; Vt = Vt[:keep]
    ev = np.maximum(ev, 1e-8)
    P = Vt.T                                   # [784,keep] principal directions
    Z = Xc @ P                                 # [N,keep] coordinates
    Zs = Z / np.sqrt(ev)                      # whitened coordinates (unit var)
    Zs = Zs / (Zs.std(axis=0, keepdims=True) + 1e-12)   # exact unit variance

    # ---------------- class structure ----------------
    onehot = np.zeros((N, K)); onehot[np.arange(N), y] = 1.0
    cls_mean = np.zeros((K, D))
    for c in range(K):
        m = y == c
        cls_mean[c] = X[m].mean(axis=0) if m.sum() > 0 else mu
    # class means in whitened space
    cm = (cls_mean - mu) @ P / np.sqrt(ev)
    cm = cm / (np.linalg.norm(cm, axis=1, keepdims=True) + 1e-12)

    # ---------------- feature construction (candidate A) ----------------
    rng = np.random.default_rng(7)

    def build(kind, seed):
        r = np.random.default_rng(seed)
        cols = []          # list of [N] feature columns (pre-activation or post)
        Wrows = []         # corresponding fc1 rows
        brows = []
        if kind == "sup":
            # (a) class-centroid directions in whitened space, rectified
            for c in range(K):
                v = cm[c]
                cols.append(Zs @ v)
                Wrows.append(v); brows.append(0.0)
            # (b) template cosine similarities (raw pixels), rectified at thresholds
            cn = cls_mean / (np.linalg.norm(cls_mean, axis=1, keepdims=True) + 1e-12)
            xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            cos = xn @ cn.T                      # [N,K]
            for c in range(K):
                for t in (0.55, 0.65, 0.75):
                    cols.append(cos[:, c] - t)
                    Wrows.append(cn[c]); brows.append(t)
            # (c) random ReLU projections of whitened space
            for j in range(128 - len(cols)):
                v = r.standard_normal(Zs.shape[1]); v /= np.linalg.norm(v)
                cols.append(Zs @ v)
                Wrows.append(v); brows.append(0.0)
        else:  # "pca": whitened PCA + random ReLU (baseline family)
            for j in range(128):
                v = r.standard_normal(Zs.shape[1]); v /= np.linalg.norm(v)
                cols.append(Zs @ v)
                Wrows.append(v); brows.append(0.0)
        A = np.stack(cols, axis=1)               # [N,128] pre-activations
        Wn = np.stack(Wrows, axis=0)             # [128,784] in whitened coords
        # convert whitened-space rows back to raw-pixel rows:
        # (Zs @ v) = ((x-mu) @ P / sqrt(ev)) @ v  ->  w = P @ (v / sqrt(ev))
        Wraw = P @ (Wn / np.sqrt(ev)[:, None]).T
        Wraw = Wraw.T
        return A, Wraw, np.asarray(brows, dtype=np.float64)

    # ---------------- closed-form ridge readout ----------------
    def ridge_fit(H, T, lam):
        """H:[N,p] (with bias col appended), T:[N,K]; returns W:[p+1,K]"""
        Np = H.shape[0]
        Hm = np.concatenate([H, np.ones((Np, 1))], axis=1)
        G = Hm.T @ Hm
        G[np.diag_indices_from(G)] += lam
        B = Hm.T @ T
        return np.linalg.solve(G, B)

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def metrics(logits, yy):
        pred = logits.argmax(axis=1)
        acc = float((pred == yy).mean())
        p = softmax(logits)
        ce = float(-np.log(np.maximum(p[np.arange(len(yy)), yy], 1e-12)).mean())
        return acc, ce

    # ---------------- split for model selection ----------------
    idx = rng.permutation(N)
    ntr = int(0.8 * N)
    tr, va = idx[:ntr], idx[ntr:]

    def evaluate(A, Wraw, brows, lam, tgt_scale, tr, va, full=False):
        pre = A @ np.eye(1)  # placeholder
        # pre-activations of the candidate hidden layer on all samples
        H = np.maximum(A, 0.0)
        # NOTE: A rows already are pre-activations; rectify
        H = np.maximum(A, 0.0)
        # ridge to scaled one-hot targets
        T = tgt_scale * (onehot - 1.0 / K)
        Wf = ridge_fit(H[tr], T[tr], lam)
        logits_va = H[va] @ Wf[:-1] + Wf[-1]
        acc, ce = metrics(logits_va, y[va])
        return acc, ce, Wf

    best = None
    for kind, seed in (("sup", 11), ("sup", 23), ("pca", 31)):
        A, Wraw, brows = build(kind, seed)
        # rectify pre-activations (bias handled separately: A already includes bias)
        for lam in (1e-4, 1e-2, 1.0, 10.0, 100.0):
            for ts in (1.0, 2.0, 4.0):
                acc, ce, Wf = evaluate(A, Wraw, brows, lam, ts, tr, va)
                key = (acc - 0.03 * ce)
                if best is None or key > best[0]:
                    best = (key, kind, seed, lam, ts, acc, ce)
    _, kind, seed, lam, ts, vacc, vce = best

    # ---------------- refit on full pool with chosen config ----------------
    A, Wraw, brows = build(kind, seed)
    H = np.maximum(A, 0.0)
    T = ts * (onehot - 1.0 / K)
    Wf = ridge_fit(H, T, lam)
    W2_new = Wf[:-1].T                      # [10,128]
    b2_new = Wf[-1]                         # [10]

    # ---------------- build fc1 rows from raw-pixel weights ----------------
    # A = X @ Wraw.T + brows  (approximately; verify numerically)
    A_chk = X @ Wraw.T + brows[None, :]
    err = float(np.abs(A_chk - A).max())
    # correct residual by adding a rank-1 correction in raw space if needed
    if err > 1e-6:
        R = A - A_chk                                  # [N,128]
        # solve least squares for correction rows:  X @ C.T ~= R  (ridge)
        G = X.T @ X
        G[np.diag_indices_from(G)] += 1e-3 * np.trace(G) / D
        C = np.linalg.solve(G, X.T @ R)                # [784,128]
        Wraw = Wraw + C.T
        A_chk2 = X @ Wraw.T + brows[None, :]
        err2 = float(np.abs(A_chk2 - A).max())
    else:
        err2 = 0.0

    # ---------------- final sanity metrics on full pool ----------------
    Hf = np.maximum(A, 0.0)
    logits_new = Hf @ W2_new.T + b2_new
    acc_full, ce_full = metrics(logits_new, y)

    # ---------------- write out ----------------
    W1_new = np.asarray(Wraw, dtype=np.float32)
    b1_new = np.asarray(brows, dtype=np.float32)
    W2o = np.asarray(W2_new, dtype=np.float32)
    b2o = np.asarray(b2_new, dtype=np.float32)

    for t in (W1_new, b1_new, W2o, b2o):
        np.clip(t, -1000.0, 1000.0, out=t)

    w["fc1.weight"] = W1_new
    w["fc1.bias"] = b1_new
    w["fc2.weight"] = W2o
    w["fc2.bias"] = b2o

    diag = {
        "selected_kind": kind,
        "selected_seed": int(seed),
        "ridge_lam": float(lam),
        "target_scale": float(ts),
        "val_acc": float(vacc),
        "val_ce": float(vce),
        "pool_acc": float(acc_full),
        "pool_ce": float(ce_full),
        "prev_pool_acc": 0.8525,
        "prev_pool_ce": 1.8836,
        "recon_err_before": err,
        "recon_err_after": err2,
        "fc1_absmax": float(np.abs(W1_new).max()),
        "fc2_absmax": float(np.abs(W2o).max()),
        "feat_mean": float(Hf.mean()),
        "feat_std": float(Hf.std()),
        "feat_zerofrac": float((Hf == 0).mean()),
        "logit_std": float(logits_new.std()),
        "n_pca": int(keep),
        "split": [int(len(tr)), int(len(va))],
    }
    return {"weights": w, "diagnostics": diag}
