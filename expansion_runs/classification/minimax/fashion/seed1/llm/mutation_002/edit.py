def edit(w, f):
    import numpy as np

    # ---------------- helpers ----------------
    def conv2d(x, W, b, pad=1):
        # x [N,C,H,W], W [O,C,3,3], b [O] -> [N,O,H,W] (cross-correlation)
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        out = np.zeros((N, O, H, Wd), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                patch = xp[:, :, i:i + H, j:j + Wd]
                out += np.tensordot(patch, W[:, :, i, j].astype(np.float64),
                                    axes=([1], [1])).transpose(0, 3, 1, 2)
        out += b.astype(np.float64).reshape(1, O, 1, 1)
        return out

    def relu(a):
        return np.maximum(a, 0.0)

    def maxpool2(a):
        N, C, H, Wd = a.shape
        a = a.reshape(N, C, H // 2, 2, Wd // 2, 2)
        return a.max(axis=(3, 5))

    def forward_feats(x, c1w, c1b, c2w, c2b):
        h = maxpool2(relu(conv2d(x, c1w, c1b)))
        h = maxpool2(relu(conv2d(h, c2w, c2b)))
        return h.reshape(h.shape[0], -1)

    # ---------------- 1. verify my forward pass against supplied activations ----------------
    ver = {}
    try:
        bx = f["layer.conv1.input0"].astype(np.float64)
        o1 = conv2d(bx, w["conv1.weight"].astype(np.float64), w["conv1.bias"].astype(np.float64))
        ver["conv1_out_maxdiff"] = float(np.max(np.abs(o1 - f["layer.conv1.output"].astype(np.float64))))
        h1 = maxpool2(relu(o1))
        ver["conv2_in_maxdiff"] = float(np.max(np.abs(h1 - f["layer.conv2.input0"].astype(np.float64))))
        o2 = conv2d(h1, w["conv2.weight"].astype(np.float64), w["conv2.bias"].astype(np.float64))
        ver["conv2_out_maxdiff"] = float(np.max(np.abs(o2 - f["layer.conv2.output"].astype(np.float64))))
        h2 = maxpool2(relu(o2))
        ver["head_in_maxdiff"] = float(np.max(np.abs(h2 - f["layer.head.input0"].astype(np.float64))))
        lg = h2 @ w["head.weight"].astype(np.float64).T + w["head.bias"].astype(np.float64)
        ver["head_out_maxdiff"] = float(np.max(np.abs(lg - f["layer.head.output"].astype(np.float64))))
    except Exception as e:
        ver["verify_error"] = str(e)

    # ---------------- 2. design new conv1 filters (16 x 1 x 3 x 3) ----------------
    # oriented first/second derivative (Gabor-like) + blur + center-surround + LoG
    k = np.arange(-1, 2, dtype=np.float64)
    yy, xx = np.meshgrid(k, k, indexing="ij")   # row=y, col=x

    def gabor(theta, sigma_u, sigma_v, freq, phase):
        ct, st = np.cos(theta), np.sin(theta)
        u = xx * ct + yy * st
        v = -xx * st + yy * ct
        env = np.exp(-0.5 * (u / sigma_u) ** 2 - 0.5 * (v / sigma_v) ** 2)
        return env * np.cos(2 * np.pi * freq * u + phase)

    def dog(theta, sa, sb):
        ct, st = np.cos(theta), np.sin(theta)
        u = xx * ct + yy * st
        return np.exp(-0.5 * (u / sa) ** 2) - np.exp(-0.5 * (u / sb) ** 2)

    filts = []
    filts.append(np.ones((3, 3)) / 9.0)                       # 0: local mean
    filts.append(dog(0.0, 0.8, 1.8))                          # 1: center-surround
    for i in range(6):                                        # 2..7: 1st derivative, 6 orient
        th = np.pi * i / 6.0
        filts.append(gabor(th, 1.6, 0.85, 0.55, 0.0))
    for i in range(6):                                        # 8..13: 2nd derivative, 6 orient
        th = np.pi * i / 6.0
        filts.append(gabor(th, 1.5, 0.8, 0.75, np.pi / 2))
    filts.append(np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) / 8.0)   # 14: LoG (spot)
    filts.append(np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 8.0)     # 15: LoG (corner)

    c1w = np.zeros((16, 1, 3, 3), dtype=np.float64)
    for i, fl in enumerate(filts):
        c1w[i, 0] = fl
    c1b = np.zeros(16, dtype=np.float64)

    # ---------------- 3. design new conv2 filters (32 x 16 x 3 x 3) ----------------
    # 0..15 : 3x3 box smoothing of conv1 channel j
    # 16..31: 3x3 box smoothing of conv1 channel j, minus 0.5 * box of channel (j+1)%16
    box = np.ones((3, 3)) / 9.0
    c2w = np.zeros((32, 16, 3, 3), dtype=np.float64)
    for j in range(16):
        c2w[j, j] = box
        c2w[16 + j, j] = box
        c2w[16 + j, (j + 1) % 16] = -0.5 * box
    c2b = np.zeros(32, dtype=np.float64)

    # ---------------- 4. recompute features for the whole permitted pool ----------------
    X = f["x"].astype(np.float64)
    Y = f["y"].astype(np.int64)
    N = X.shape[0]
    F = forward_feats(X, c1w, c1b, c2w, c2b)      # [N, 1568]
    F = np.maximum(F, 0.0)

    # ---------------- 5. closed-form ridge head (normal equations + Cholesky) ----------------
    Y1 = np.zeros((N, 10), dtype=np.float64)
    Y1[np.arange(N), Y] = 1.0
    muF = F.mean(axis=0)
    Fc = F - muF
    Yc = Y1 - Y1.mean(axis=0)
    G = Fc.T @ Fc
    tr = float(np.trace(G))
    d = G.shape[0]
    lam = 0.05 * tr / d            # single fixed shrinkage (no hyper-parameter search)
    A = G + lam * np.eye(d)
    B = Fc.T @ Yc
    try:
        L = np.linalg.cholesky(A)
        Wt = np.linalg.solve(L.T, np.linalg.solve(L, B))     # [D,10]
    except np.linalg.LinAlgError:
        Wt = np.linalg.lstsq(A, B, rcond=None)[0]
    bhead = Y1.mean(axis=0) - muF @ Wt
    headW = Wt.T.copy()                                       # [10, D]

    # ---------------- 6. diagnostics: in-sample + 5-fold CV (single fixed lambda) ----------
    def acc_of(logits, yy):
        return float(np.mean(np.argmax(logits, axis=1) == yy))

    train_logits = F @ headW.T + bhead
    diag = {}
    diag["train_acc"] = acc_of(train_logits, Y)
    diag["train_ce"] = float(np.mean(np.log(np.exp(train_logits - train_logits.max(1, keepdims=True)).sum(1))
                                     + train_logits.max(1) - train_logits[np.arange(N), Y]))
    # 5-fold CV (evaluation only, closed-form fit per fold)
    idx = np.arange(N)
    folds = idx % 5
    cv_correct = 0
    for fo in range(5):
        tr = folds != fo
        te = ~tr
        Ft, Yt = F[tr], Y1[tr]
        mF = Ft.mean(axis=0); FcT = Ft - mF
        YcT = Yt - Yt.mean(axis=0)
        At = FcT.T @ FcT + lam * np.eye(d)
        Bt = FcT.T @ YcT
        try:
            Lt = np.linalg.cholesky(At)
            Wt2 = np.linalg.solve(Lt.T, np.linalg.solve(Lt, Bt))
        except np.linalg.LinAlgError:
            Wt2 = np.linalg.lstsq(At, Bt, rcond=None)[0]
        b2 = Yt.mean(axis=0) - mF @ Wt2
        cv_correct += int(np.sum(np.argmax(F[te] @ Wt2 + b2, axis=1) == Y[te]))
    diag["cv5_acc"] = cv_correct / float(N)
    diag["lambda"] = lam
    diag["trace_over_d"] = tr / d
    diag["feature_mean"] = float(F.mean())
    diag["feature_std"] = float(F.std())
    diag["feat_nonzero_frac"] = float(np.mean(F > 0))
    diag["head_abs_max"] = float(np.max(np.abs(headW)))
    diag["old_acc_check"] = acc_of(f["logits"].astype(np.float64), Y)
    for kk, vv in ver.items():
        diag[kk] = vv

    # ---------------- 7. write back ----------------
    w["conv1.weight"] = c1w.astype(np.float32)
    w["conv1.bias"] = c1b.astype(np.float32)
    w["conv2.weight"] = c2w.astype(np.float32)
    w["conv2.bias"] = c2b.astype(np.float32)
    w["head.weight"] = np.clip(headW, -1000.0, 1000.0).astype(np.float32)
    w["head.bias"] = np.clip(bhead, -1000.0, 1000.0).astype(np.float32)

    return {"weights": w, "diagnostics": diag}
