def edit(w, f):
    import numpy as np

    # ---------------- helpers: forward pass (numpy) ----------------
    def conv2d(x, W, b):
        # x [N,C,H,W], W [O,C,3,3], zero padding 1, stride 1
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float64)
        xp[:, :, 1:H + 1, 1:Wd + 1] = x
        out = np.zeros((N, O, H, Wd), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                patch = xp[:, :, i:i + H, j:j + Wd]          # [N,C,H,W]
                for o in range(O):
                    out[:, o] += patch[:, 0] * 0.0          # placeholder (no-op)
        # vectorized version (memory friendly, per output channel)
        out = np.zeros((N, O, H, Wd), dtype=np.float64)
        for o in range(O):
            acc = np.zeros((N, H, Wd), dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    acc += np.tensordot(xp[:, :, i:i + H, j:j + Wd], W[o, :, i, j], axes=([1], [0]))
            out[:, o] = acc
        out += b.reshape(1, O, 1, 1)
        return out

    def maxpool2d(a):
        # 2x2 stride 2, no padding; H,W even here
        N, C, H, Wd = a.shape
        H2, W2 = H // 2, Wd // 2
        r = a[:, :, 0:2 * H2:2, 0:2 * W2:2]
        r = r.reshape(N, C, H2, 2, W2, 2)
        return r.max(axis=(3, 5))

    def forward(x, cw1, cb1, cw2, cb2, hw, hb):
        h1 = np.maximum(conv2d(x, cw1, cb1), 0.0)
        h1p = maxpool2d(h1)
        h2 = np.maximum(conv2d(h1p, cw2, cb2), 0.0)
        h2p = maxpool2d(h2)
        feats = h2p.reshape(h2p.shape[0], -1)
        logits = feats @ hw.T + hb
        return feats, logits

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float64)          # [N,1,28,28]
    y = np.asarray(f["y"], dtype=np.int64)
    N = x.shape[0]
    X = x.reshape(N, -1)                               # [N,784]

    cw1 = np.asarray(w["conv1.weight"], dtype=np.float64)
    cb1 = np.asarray(w["conv1.bias"], dtype=np.float64)
    cw2 = np.asarray(w["conv2.weight"], dtype=np.float64)
    cb2 = np.asarray(w["conv2.bias"], dtype=np.float64)
    hw = np.asarray(w["head.weight"], dtype=np.float64)
    hb = np.asarray(w["head.bias"], dtype=np.float64)

    diag = {}

    # ---------------- sanity: reproduce cached forward pass ----------------
    try:
        feats_chk, logits_chk = forward(x[:64], cw1, cb1, cw2, cb2, hw, hb)
        f_ref = np.asarray(f["features"][:64], dtype=np.float64)
        l_ref = np.asarray(f["logits"][:64], dtype=np.float64)
        diag["fwd_feat_maxerr"] = float(np.max(np.abs(feats_chk - f_ref)))
        diag["fwd_logit_maxerr"] = float(np.max(np.abs(logits_chk - l_ref)))
    except Exception as e:
        diag["fwd_error"] = str(e)

    # ---------------- class statistics ----------------
    classes = np.unique(y)
    K = 10
    M = np.zeros((K, 784))
    for c in classes:
        M[c] = X[y == c].mean(axis=0)
    Mbar = X.mean(axis=0)
    Mc = M.reshape(K, 28, 28)
    Mbar2 = Mbar.reshape(28, 28)

    # ---------------- candidate conv1 filter designs ----------------
    def design_conv1_diag():
        """Class-discriminative 3x3 filters: for each class c, the 3x3 patch
        (centered at the class's most discriminative location) of (M_c - Mbar),
        plus a few generic edge/center-surround filters."""
        filters = []
        # discriminative location: maximize |M_c - Mbar| local energy
        for c in range(K):
            D = (Mc[c] - Mbar2)
            # local 3x3 energy map of the difference image
            e = np.zeros((26, 26))
            for i in range(3):
                for j in range(3):
                    e += np.abs(D[i:i + 26, j:j + 26])
            loc = np.unravel_index(np.argmax(e), e.shape)
            r, c2 = loc[0] + 1, loc[1] + 1
            patch = D[r - 1:r + 2, c2 - 1:c2 + 2].copy()
            filters.append(patch)
        # generic filters to fill remaining slots
        gen = []
        for i in range(3):
            for j in range(3):
                gen.append(np.eye(3)[i] * np.eye(3)[j])
        while len(filters) < 16:
            filters.append(gen[(len(filters) - K) % len(gen)])
        F = np.stack(filters[:16])[:, None, :, :]      # [16,1,3,3]
        return F

    def design_conv1_mixed():
        """Half class-discriminative templates, half random-but-fixed filters
        (seeded, deterministic) to keep feature diversity."""
        F = design_conv1_diag()
        rng = np.random.RandomState(12345)
        extra = rng.randn(6, 1, 3, 3) * 0.25
        return np.concatenate([F[:10], extra], axis=0)

    def design_conv2_random(seed):
        """Random projection conv2: each output filter mixes all 16 conv1 channels
        with random 3x3 spatial taps; keeps 1568 diverse pooled features."""
        rng = np.random.RandomState(seed)
        W2 = rng.randn(32, 16, 3, 3) / np.sqrt(16 * 9.0)
        b2 = np.zeros(32)
        return W2, b2

    # ---------------- ridge head fitting (closed form) ----------------
    def ridge_fit(Feat, Y, lam):
        n, d = Feat.shape
        mu = Feat.mean(axis=0)
        Xc = Feat - mu
        ybar = Y.mean(axis=0)
        T = Y - ybar
        G = Xc.T @ Xc
        G.flat[:: d + 1] += lam * n / d
        B = np.linalg.solve(G, Xc.T @ T)
        intercept = ybar - mu @ B
        return B, intercept

    def acc_of(Feat, Y, B, intercept):
        P = Feat @ B + intercept
        return float((P.argmax(axis=1) == Y.argmax(axis=1)).mean())

    def build_candidates():
        cands = []
        # (a) head-only on the given (current) features
        cands.append(("head_only_current", cw1, cb1, cw2, cb2,
                      np.asarray(f["features"], dtype=np.float64)))
        # (b) discriminative conv1 + random conv2
        cands.append(("diag_conv1", design_conv1_diag(), np.zeros(16),
                      *design_conv2_random(7), None))
        # (c) mixed conv1 + random conv2
        cands.append(("mixed_conv1", design_conv1_mixed(), np.zeros(16),
                      *design_conv2_random(11), None))
        return cands

    # one-hot targets
    Y1 = np.zeros((N, K))
    Y1[np.arange(N), y] = 1.0

    # validation split (deterministic)
    rng = np.random.RandomState(2024)
    perm = rng.permutation(N)
    nval = 400
    val_idx = perm[:nval]
    tr_idx = perm[nval:]

    results = []
    for name, c1, b1, c2, b2, feats in build_candidates():
        if feats is None:
            feats, _ = forward(x, c1, b1, c2, b2, hw, hb)
        # scale conv1 to keep activation magnitudes sane
        s1 = float(np.max(np.abs(c1))) if np.max(np.abs(c1)) > 0 else 1.0
        # normalise conv2 so conv2 output std ~ 0.1
        best = None
        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            B, ic = ridge_fit(feats[tr_idx], Y1[tr_idx], lam)
            a_tr = acc_of(feats[tr_idx], Y1[tr_idx], B, ic)
            a_va = acc_of(feats[val_idx], Y1[val_idx], B, ic)
            rec = (a_va, a_tr, lam, B, ic)
            if best is None or a_va > best[0]:
                best = rec
        a_va, a_tr, lam, B, ic = best
        results.append((a_va, a_tr, name, c1, b1, c2, b2, feats, lam, B, ic))

    results.sort(key=lambda t: -t[0])
    a_va, a_tr, name, c1, b1, c2, b2, feats, lam, B, ic = results[0]

    # ---------------- final head fit on ALL data with chosen lambda ----------------
    Bf, icf = ridge_fit(feats, Y1, lam)
    train_acc = acc_of(feats, Y1, Bf, icf)
    pred = (feats @ Bf + icf).argmax(axis=1)
    per_class = {}
    for c in range(K):
        m = y == c
        per_class[c] = float((pred[m] == c).mean()) if m.sum() else 0.0

    # ---------------- write weights ----------------
    def to32(a):
        a = np.asarray(a, dtype=np.float64)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.clip(a, -1000.0, 1000.0)
        return a.astype(np.float32)

    w["conv1.weight"] = to32(c1)
    w["conv1.bias"] = to32(b1)
    w["conv2.weight"] = to32(c2)
    w["conv2.bias"] = to32(b2)
    w["head.weight"] = to32(Bf)
    w["head.bias"] = to32(icf)

    diag.update({
        "chosen": name,
        "ridge_lambda": float(lam),
        "val_acc": float(a_va),
        "train_acc_split": float(a_tr),
        "train_acc_full": float(train_acc),
        "per_class_acc": per_class,
        "candidates": {r[2]: {"val": float(r[0]), "train": float(r[1])} for r in results},
        "head_w_std": float(np.std(Bf)),
        "head_w_absmax": float(np.max(np.abs(Bf))),
        "feat_mean": float(np.mean(feats)),
        "feat_std": float(np.std(feats)),
        "n_train": int(N),
    })
    return {"weights": w, "diagnostics": diag}
