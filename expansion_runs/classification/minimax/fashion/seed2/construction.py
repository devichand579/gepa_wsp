def edit(w, f):
    import numpy as np

    # ---------------- helpers ----------------
    def conv2d(x, W, b, pad=1):
        # x: [N,C,H,W], W: [O,C,3,3] -> [N,O,H,W]
        N, C, H, Wd = x.shape
        O = W.shape[0]
        if pad:
            xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        else:
            xp = x
        Hp, Wp = xp.shape[2], xp.shape[3]
        cols = np.empty((N, C, 3, 3, Hp, Wp), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                cols[:, :, i, j] = xp[:, :, i:i + Hp, j:j + Wp]
        out = np.tensordot(W.reshape(O, C * 9), cols.reshape(N, C * 9, Hp, Wp),
                            axes=(1, 1))
        out = out + b.reshape(1, O, 1, 1)
        return out.astype(np.float32)

    def relu(a):
        return np.maximum(a, 0.0)

    def maxpool(a):
        # [N,C,H,W] -> [N,C,H//2,W//2]
        N, C, H, Wd = a.shape
        r = a.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))
        return r

    def forward(x, c1w, c1b, c2w, c2b):
        h1 = maxpool(relu(conv2d(x, c1w, c1b)))
        h2 = maxpool(relu(conv2d(h1, c2w, c2b)))
        return h2.reshape(x.shape[0], -1)

    # ---------------- design conv1: 8 orientations x 2 polarities ----------------
    d1 = np.array([-1.0, 0.0, 1.0])          # derivative along direction a
    s1 = np.array([1.0, 2.0, 1.0])           # smoothing along perpendicular b
    idx = np.array([-1.0, 0.0, 1.0])

    def orient_filter(theta, kind):
        ca, sa = np.cos(theta), np.sin(theta)
        # a = (row dir, col dir); b = perpendicular
        # build 3x3 separable: derivative along a, smooth along b
        # rows index i (row), cols index j (col)
        # derivative along a: d1[i] * ca + d1[j] * sa  (approx separable sum)
        # smoothing along b: s1[i] * (-sa) + s1[j] * (ca)
        D = np.outer(d1 * ca, np.ones(3)) + np.outer(np.ones(3), d1 * sa)
        S = np.outer(s1 * (-sa), np.ones(3)) + np.outer(np.ones(3), s1 * ca)
        if kind == 0:      # edge: derivative along a, smooth along b
            F = D * S
        else:              # line/energy: smooth along a, derivative along b
            F = S * D.T
        F = F - F.mean()
        n = np.sqrt((F * F).sum())
        if n < 1e-8:
            n = 1.0
        return F / n

    c1 = np.zeros((16, 1, 3, 3), dtype=np.float64)
    for k in range(8):
        th = k * np.pi / 8.0
        c1[2 * k, 0] = orient_filter(th, 0)
        c1[2 * k + 1, 0] = orient_filter(th, 1)
    c1b = np.zeros(16, dtype=np.float64)

    # ---------------- design conv2: orientation energy x 4 spatial kernels ----------------
    # kernels: delta, 3x3 box, horizontal 1x3, vertical 3x1
    k_delta = np.zeros((3, 3)); k_delta[1, 1] = 1.0
    k_box = np.ones((3, 3)) / 9.0
    k_h = np.zeros((3, 3)); k_h[1, :] = 1.0 / 3.0
    k_v = np.zeros((3, 3)); k_v[:, 1] = 1.0 / 3.0
    kernels = [k_delta, k_box, k_h, k_v]

    c2 = np.zeros((32, 16, 3, 3), dtype=np.float64)
    for k in range(8):
        for m in range(4):
            j = 4 * k + m
            c2[j, 2 * k] = kernels[m]
            c2[j, 2 * k + 1] = kernels[m]
    c2b = np.zeros(32, dtype=np.float64)

    # ---------------- compute features on the permitted pool ----------------
    x = f["x"]
    y = np.asarray(f["y"]).astype(np.int64)
    N = x.shape[0]
    feats = np.empty((N, 1568), dtype=np.float64)
    bs = 250
    for s in range(0, N, bs):
        e = min(N, s + bs)
        feats[s:e] = forward(x[s:e].astype(np.float32),
                             c1.astype(np.float32), c1b.astype(np.float32),
                             c2.astype(np.float32), c2b.astype(np.float32)).astype(np.float64)

    # ---------------- closed-form heads ----------------
    def standardize(F):
        mu = F.mean(axis=0)
        sd = F.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return (F - mu) / sd, mu, sd

    Fz, mu_f, sd_f = standardize(feats)
    Fz = np.clip(Fz, -8.0, 8.0)

    Y = np.zeros((N, 10))
    Y[np.arange(N), y] = 1.0

    def lda_fit(F, Y, lam):
        C = F.shape[1]
        means = np.zeros((10, C))
        Sw = np.zeros((C, C))
        for c in range(10):
            m = y == c
            Fc = F[m]
            mc = Fc.mean(axis=0)
            means[c] = mc
            Z = Fc - mc
            Sw += Z.T @ Z
        Sw /= max(N - 10, 1)
        tr = np.trace(Sw) / C
        Sw = (1.0 - lam) * Sw + lam * tr * np.eye(C)
        try:
            Wi = np.linalg.inv(Sw)
        except Exception:
            Wi = np.linalg.pinv(Sw)
        Wl = Wi @ means.T
        bl = -0.5 * np.sum(Wl * means.T, axis=0) + np.log(np.bincount(y, minlength=10) / N)
        return Wl.T, bl

    def ridge_fit(F, Y, lam):
        C = F.shape[1]
        A = F.T @ F + lam * np.trace(F.T @ F) / C * np.eye(C)
        B = F.T @ Y
        Wr = np.linalg.solve(A, B)
        br = Y.mean(axis=0) - F.mean(axis=0) @ Wr
        return Wr, br

    # single held-out split for model selection (closed-form fits only)
    rs = np.arange(N)
    tr_idx = rs[rs % 5 != 0]
    va_idx = rs[rs % 5 == 0]

    def acc(F, yv, W, b):
        return float(np.mean((F @ W + b).argmax(axis=1) == yv))

    cands = []
    for lam in (0.02, 0.05, 0.1, 0.2, 0.4):
        Wl, bl = lda_fit(Fz[tr_idx], Y[tr_idx], lam)
        cands.append(("lda", lam, Wl, bl, acc(Fz[va_idx], y[va_idx], Wl, bl)))
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        Wr, br = ridge_fit(Fz[tr_idx], Y[tr_idx], lam)
        cands.append(("ridge", lam, Wr, br, acc(Fz[va_idx], y[va_idx], Wr, br)))
    cands.sort(key=lambda t: -t[4])
    kind, lam, Wsel, bsel, val_acc = cands[0]

    # refit chosen model on the full permitted pool
    if kind == "lda":
        Wf, bf = lda_fit(Fz, Y, lam)
    else:
        Wf, bf = ridge_fit(Fz, Y, lam)
    train_acc = acc(Fz, y, Wf, bf)

    # ---------------- write weights ----------------
    out = dict(w)
    out["conv1.weight"] = c1.astype(np.float32)
    out["conv1.bias"] = c1b.astype(np.float32)
    out["conv2.weight"] = c2.astype(np.float32)
    out["conv2.bias"] = c2b.astype(np.float32)

    # head operates on raw (unstandardized) features:
    # logits = (F - mu)/sd @ Wf + bf = F @ (Wf/sd) + (bf - (mu/sd)@Wf)
    Wraw = Wf / sd_f[:, None]
    braw = bf - (mu_f / sd_f) @ Wf
    out["head.weight"] = Wraw.T.astype(np.float32)
    out["head.bias"] = braw.astype(np.float32)

    # safety: finite, bounded
    for k in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
              "head.weight", "head.bias"):
        a = np.asarray(out[k], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = np.clip(a, -1000.0, 1000.0)
        out[k] = a.astype(np.float32)

    # per-class training accuracy
    pred = (feats @ Wraw + braw).argmax(axis=1)
    per_class = {int(c): float(np.mean(pred[y == c] == c)) for c in range(10)}

    diag = {
        "method": kind,
        "lambda": float(lam),
        "val_acc": round(val_acc, 4),
        "train_acc": round(train_acc, 4),
        "per_class_train_acc": per_class,
        "feat_mean": round(float(feats.mean()), 5),
        "feat_std": round(float(feats.std()), 5),
        "feat_frac_zero": round(float((feats == 0).mean()), 4),
        "logit_std": round(float((feats @ Wraw + braw).std()), 4),
        "head_w_absmean": round(float(np.abs(Wraw).mean()), 6),
        "head_w_absmax": round(float(np.abs(Wraw).max()), 4),
        "cands": [(c[0], c[1], round(c[4], 4)) for c in cands],
    }
    return {"weights": out, "diagnostics": diag}
