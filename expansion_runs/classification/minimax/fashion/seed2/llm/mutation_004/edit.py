import numpy as np

def edit(w, f):
    # ---------------- design conv1: 16 delta (shift) filters, bias 0 ----------------
    C1 = 16
    shifts = []
    for dy in range(4):
        for dx in range(4):
            shifts.append((dy, dx))
    shifts = shifts[:C1]                      # 16 shifts in [0,3]^2
    W1 = np.zeros((C1, 1, 3, 3), dtype=np.float64)
    for k, (dy, dx) in enumerate(shifts):
        W1[k, 0, dy, dx] = 1.0
    b1 = np.zeros((C1,), dtype=np.float64)

    # ---------------- design conv2: 32 filters selecting one conv1 channel -------
    C2 = 32
    W2 = np.zeros((C2, C1, 3, 3), dtype=np.float64)
    b2 = np.zeros((C2,), dtype=np.float64)
    # 16 conv1 channels x 2 conv2 offsets (0,0) and (1,1)  -> 32 combos
    for m in range(C2):
        k = m % C1
        o = m // C1
        W2[m, k, o, o] = 1.0

    # ---------------- numpy forward pass (exact same ops as the net) --------------
    def conv2d(x, W, b, pad=1):
        # x: [N,Cin,H,W], W: [Cout,Cin,3,3]
        N, Cin, H, Wd = x.shape
        Cout = W.shape[0]
        xp = np.zeros((N, Cin, H + 2 * pad, Wd + 2 * pad), dtype=np.float64)
        xp[:, :, pad:pad + H, pad:pad + Wd] = x
        # im2col via stride tricks
        s = xp.strides
        cols = np.lib.stride_tricks.as_strided(
            xp, shape=(N, Cin, 3, 3, H, Wd), strides=(s[0], s[1], s[2], s[3], s[2], s[3]))
        cols = cols.reshape(N, Cin * 9, H * Wd)
        Wf = W.reshape(Cout, Cin * 9)
        out = np.einsum('oc,ncp->nop', Wf, cols) + b[:, None]
        return out.reshape(N, Cout, H, Wd)

    def maxpool(x):
        N, C, H, Wd = x.shape
        return x.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))

    X = f["x"].astype(np.float64)[:, 0]          # [N,28,28]
    y = f["y"].astype(np.int64)
    N = X.shape[0]

    h1 = np.maximum(conv2d(X[:, None], W1, b1), 0.0)
    h1p = maxpool(h1)                             # [N,16,14,14]
    h2 = np.maximum(conv2d(h1p, W2, b2), 0.0)    # [N,32,14,14]
    h2p = maxpool(h2)                             # [N,32,7,7]
    F = h2p.reshape(N, C2 * 49)                   # [N,1568] final features

    # ---------------- closed-form ridge fit of head on one-hot targets -----------
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    mu = F.mean(axis=0)
    Fs = F - mu
    G = Fs.T @ Fs                                # [1568,1568]
    d = G.shape[0]
    tr = np.trace(G)
    # closed-form 5-fold CV over a small lambda grid (no gradients, no SGD)
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    folds = np.array_split(perm, 5)
    lams = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    lam_rel = lams
    best_lam, best_err = None, np.inf
    for lr in lam_rel:
        lam = lr * (tr / d + 1e-12)
        errs = []
        for fo in folds:
            mask = np.ones(N, dtype=bool)
            mask[fo] = False
            Xtr = Fs[mask]
            Ytr = Y[mask]
            mtr = Xtr.mean(axis=0)
            Xc = Xtr - mtr
            A = Xc.T @ Xc + lam * np.eye(d)
            B = Xc.T @ (Ytr - Ytr.mean(axis=0))
            Wf = np.linalg.solve(A, B)
            btr = Ytr.mean(axis=0) - mtr @ Wf
            Xte = Fs[fo] - mtr
            P = Xte @ Wf + btr
            errs.append(np.mean(np.argmax(P, axis=1) != y[fo]))
        e = float(np.mean(errs))
        if e < best_err:
            best_err, best_lam = e, lam

    lam = best_lam
    A = G + lam * np.eye(d)
    B = Fs.T @ (Y - Y.mean(axis=0))
    Wh = np.linalg.solve(A, B)
    bh = Y.mean(axis=0) - mu @ Wh

    # ---------------- shrinkage-LDA alternative (pick better by CV) --------------
    # class means in feature space, shared covariance with shrinkage
    cls_mean = np.zeros((10, d))
    for c in range(10):
        m = (y == c)
        cls_mean[c] = Fs[m].mean(axis=0) if m.any() else 0.0
    Sw = np.zeros((d, d))
    for c in range(10):
        m = (y == c)
        if m.any():
            Z = Fs[m] - cls_mean[c]
            Sw += Z.T @ Z
    Sw /= max(N - 10, 1)
    trS = np.trace(Sw) / d
    best_lda = None
    best_lda_err = np.inf
    for alpha in [0.05, 0.1, 0.2, 0.4, 0.7]:
        Sr = (1 - alpha) * Sw + alpha * trS * np.eye(d)
        Sr += 1e-6 * trS * np.eye(d)
        errs = []
        for fo in folds:
            mask = np.ones(N, dtype=bool)
            mask[fo] = False
            trm = mask
            cm = np.zeros((10, d))
            for c in range(10):
                mm = trm & (y == c)
                cm[c] = Fs[mm].mean(axis=0) if mm.any() else 0.0
            Swf = np.zeros((d, d))
            for c in range(10):
                mm = trm & (y == c)
                if mm.any():
                    Z = Fs[mm] - cm[c]
                    Swf += Z.T @ Z
            Swf /= max(int(trm.sum()) - 10, 1)
            Sr2 = (1 - alpha) * Swf + alpha * (np.trace(Swf) / d) * np.eye(d)
            Sr2 += 1e-6 * (np.trace(Swf) / d + 1e-12) * np.eye(d)
            P0 = np.linalg.solve(Sr2, cm.T)      # [d,10]
            b0 = -0.5 * np.einsum('ct,dc->t', cm, P0) + np.log(
                np.maximum(np.bincount(y[trm], minlength=10) / max(int(trm.sum()), 1), 1e-9))
            Xte = Fs[fo]
            P = Xte @ P0 + b0
            errs.append(np.mean(np.argmax(P, axis=1) != y[fo]))
        e = float(np.mean(errs))
        if e < best_lda_err:
            best_lda_err, best_lda = e, alpha

    alpha = best_lda
    Sr = (1 - alpha) * Sw + alpha * trS * np.eye(d)
    Sr += 1e-6 * (trS + 1e-12) * np.eye(d)
    P0 = np.linalg.solve(Sr, cls_mean.T)
    b0 = -0.5 * np.einsum('ct,dc->t', cls_mean, P0) + np.log(
        np.maximum(np.bincount(y, minlength=10) / N, 1e-9))

    # ---------------- pick the better of ridge / LDA by CV error ----------------
    use_lda = best_lda_err < best_err
    if use_lda:
        Wf = P0.T.copy()                          # [10,d]
        bf = b0.copy()
    else:
        Wf = Wh.T.copy()                          # [10,d]
        bf = bh.copy()

    # ---------------- write back, keep shapes / dtype / bounds ------------------
    w["conv1.weight"] = W1.astype(np.float32)
    w["conv1.bias"] = b1.astype(np.float32)
    w["conv2.weight"] = W2.astype(np.float32)
    w["conv2.bias"] = b2.astype(np.float32)
    w["head.weight"] = np.clip(Wf, -1000.0, 1000.0).astype(np.float32)
    w["head.bias"] = np.clip(bf, -1000.0, 1000.0).astype(np.float32)

    # in-sample diagnostics
    P = F @ Wf.T + bf
    acc = float(np.mean(np.argmax(P, axis=1) == y))
    ce = float(np.mean(-np.log(np.maximum(
        np.exp(P - P.max(axis=1, keepdims=True)) / np.exp(P - P.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True),
        1e-12)[np.arange(N), y])))

    diag = {
        "design": "conv1=16 delta shifts (bias0); conv2=32 channel/offset selectors; head=closed-form fit",
        "feature_dim": int(d),
        "ridge_cv_err": round(best_err, 4),
        "lda_cv_err": round(best_lda_err, 4),
        "chosen": "lda" if use_lda else "ridge",
        "lambda_rel": float(lam / (tr / d + 1e-12)),
        "lda_alpha": float(alpha),
        "train_acc_insample": round(acc, 4),
        "train_ce_insample": round(ce, 4),
        "feat_mean": round(float(F.mean()), 5),
        "feat_std": round(float(F.std()), 5),
        "head_w_absmean": round(float(np.abs(Wf).mean()), 6),
        "head_w_absmax": round(float(np.abs(Wf).max()), 6),
        "conv1_w_absmax": 1.0,
        "conv2_w_absmax": 1.0,
        "class_counts": [int(c) for c in np.bincount(y, minlength=10)],
    }
    return {"weights": w, "diagnostics": diag}
