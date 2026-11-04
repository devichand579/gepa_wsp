import numpy as np

def edit(w, f):
    # ---------------- helpers (numpy only) ----------------
    def conv2d(x, W, b, pad=1):
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        cols = np.empty((N, C * 9, H * Wd), dtype=np.float32)
        idx = 0
        for c in range(C):
            for i in range(3):
                for j in range(3):
                    cols[:, idx, :] = xp[:, c, i:i + H, j:j + Wd].reshape(N, -1)
                    idx += 1
        out = cols.reshape(N, C * 9, H * Wd) @ W.reshape(O, C * 9).T
        out = out + b[None, :, None]
        return out.reshape(N, O, H, Wd)

    def maxpool2(x):
        N, C, H, Wd = x.shape
        a = x[:, :, 0:H:2, 0:Wd:2]
        b_ = x[:, :, 1:H:2, 0:Wd:2]
        c_ = x[:, :, 0:H:2, 1:Wd:2]
        d_ = x[:, :, 1:H:2, 1:Wd:2]
        return np.maximum(np.maximum(a, b_), np.maximum(c_, d_))

    def forward_feats(x, cw1, cb1, cw2, cb2, chunk=200):
        outs = []
        for s in range(0, x.shape[0], chunk):
            xb = x[s:s + chunk]
            h = np.maximum(conv2d(xb, cw1, cb1, 1), 0.0)
            h = maxpool2(h)
            h = np.maximum(conv2d(h, cw2, cb2, 1), 0.0)
            h = maxpool2(h)
            outs.append(h.reshape(h.shape[0], -1))
        return np.concatenate(outs, 0).astype(np.float64)

    # ---------------- hand-crafted conv1 filter bank (16 x 1 x 3 x 3) ----------------
    Dx = np.array([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])
    Dy = np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]])
    Sx = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    Sy = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])

    f1 = []
    # 0..7 : 8 unsigned oriented derivative filters (22.5 deg steps)
    for k in range(8):
        th = np.pi * k / 8.0
        f1.append(np.cos(th) * Dx + np.sin(th) * Dy)
    # 8..11 : 4 smoothed (Sobel) orientations
    for k in range(4):
        th = np.pi * k / 4.0
        f1.append(np.cos(th) * Sx + np.sin(th) * Sy)
    # 12 : box smooth ; 13 : gaussian smooth
    f1.append(np.ones((3, 3)) / 9.0)
    f1.append(np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0)
    # 14 : identity (intensity) ; 15 : centre-surround high-pass
    f1.append(np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]))
    f1.append(np.array([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]) / 8.0)
    cw1 = np.stack(f1).astype(np.float32)[:, None, :, :]
    cb1 = np.zeros(16, dtype=np.float32)

    # ---------------- hand-crafted conv2 routing (32 x 16 x 3 x 3) ----------------
    box = np.ones((3, 3)) / 9.0
    f2 = np.zeros((32, 16, 3, 3), dtype=np.float64)
    for c in range(16):                       # 0..15 : local energy of each conv1 channel
        f2[c, c] = box
    for g in range(4):                       # 16..19 : grouped orientation energy (2 dirs each)
        f2[16 + g, 2 * g] = box
        f2[16 + g, 2 * g + 1] = box
    for g in range(4):                       # 20..23 : grouped energy, centre-weighted spatial kernel
        kern = np.array([[0.5, 1.0, 0.5], [1.0, 2.0, 1.0], [0.5, 1.0, 0.5]]) / 8.0
        f2[20 + g, 2 * g] = kern
        f2[20 + g, 2 * g + 1] = kern
    f2[24, 0:8] = box                        # 24 : total unsigned edge energy
    f2[25, 0:8] = box * 0.5
    f2[26, 8:12] = box                       # 26 : smoothed-orientation energy
    f2[27, 12] = box                         # 27 : smoothed intensity (box)
    f2[28, 13] = box                         # 28 : smoothed intensity (gauss)
    f2[29, 14] = box                         # 29 : mean intensity
    f2[30, 15] = box                         # 30 : local high-pass energy
    f2[31, 14] = box * 0.5
    cw2 = f2.astype(np.float32)
    cb2 = np.zeros(32, dtype=np.float32)

    # ---------------- closed-form head fit on the permitted pool ----------------
    x = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"]).astype(np.int64)
    N = x.shape[0]
    X = forward_feats(x, cw1, cb1, cw2, cb2)

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd

    classes = np.unique(y)
    C = len(classes)
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    M = Z.mean(axis=0)
    Zc = Z - M
    Sw = np.zeros((Z.shape[1], Z.shape[1]), dtype=np.float64)
    mus = np.zeros((C, Z.shape[1]), dtype=np.float64)
    for ci, c in enumerate(classes):
        idx = np.where(y == c)[0]
        mc = Zc[idx].mean(axis=0)
        mus[ci] = mc
        D = Zc[idx] - mc
        Sw += D.T @ D
    Sw /= N
    d = Z.shape[1]
    tr = np.trace(Sw) / d
    alpha = 0.10
    Sr = (1.0 - alpha) * Sw + alpha * tr * np.eye(d)
    Sr += 1e-6 * tr * np.eye(d)

    prior = np.array([(y == c).sum() for c in classes], dtype=np.float64) / N
    prior = np.maximum(prior, 1e-6)
    prior = prior / prior.sum()

    try:
        Sr_inv = np.linalg.inv(Sr)
    except np.linalg.LinAlgError:
        Sr_inv = np.linalg.pinv(Sr)
    Wc = Sr_inv @ mus.T                      # [d, C]
    b = -0.5 * np.sum(mus * Wc.T, axis=1) + np.log(prior)

    # fold standardisation back into the linear head
    hw = (Wc / sd[:, None]).T.astype(np.float32)   # [C, d]
    hb = (b - (mu / sd) @ Wc).astype(np.float32)

    # ---------------- write back ----------------
    w["conv1.weight"] = cw1
    w["conv1.bias"] = cb1
    w["conv2.weight"] = cw2
    w["conv2.bias"] = cb2
    w["head.weight"] = hw
    w["head.bias"] = hb

    # ---------------- diagnostics ----------------
    logits = Z @ Wc + b[None, :]
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.maximum(p[np.arange(N), y], 1e-12)).mean())
    sep = float(np.mean(np.abs(mus - mus.mean(axis=0, keepdims=True))))
    eff_rank = float(np.exp(-np.sum(p_ * np.log(p_ + 1e-300))
                           for p_ in [np.linalg.svd(Zc, compute_uv=False) ** 2 /
                                      (Zc.shape[0] * np.maximum(np.linalg.svd(Zc, compute_uv=False) ** 2, 1e-30)).sum()]) if False else 0.0)
    sv = np.linalg.svd(Zc, compute_uv=False)
    ev = sv ** 2
    ev = ev / max(ev.sum(), 1e-30)
    eff_rank = float(np.exp(-np.sum(ev * np.log(ev + 1e-300))))
    per_class = {}
    for ci, c in enumerate(classes):
        idx = np.where(y == c)[0]
        per_class[str(int(c))] = round(float((pred[idx] == c).mean()), 4)

    diag = {
        "train_accuracy": round(acc, 4),
        "train_cross_entropy": round(ce, 4),
        "per_class_accuracy": per_class,
        "feature_mean_separation": round(sep, 5),
        "feature_effective_rank": round(eff_rank, 2),
        "shrinkage_alpha": alpha,
        "head_weight_abs_max": round(float(np.abs(hw).max()), 5),
        "head_bias": [round(float(v), 4) for v in hb],
        "conv1_weight_abs_max": round(float(np.abs(cw1).max()), 4),
        "conv2_weight_abs_max": round(float(np.abs(cw2).max()), 4),
        "n_train": int(N),
    }
    return {"weights": w, "diagnostics": diag}
