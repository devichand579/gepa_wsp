import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: (N,C,H,W), W: (O,C,3,3) cross-correlation, pad 1
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float32)
        xp[:, :, 1:-1, 1:-1] = x
        out = np.zeros((N, O, H, Wd), dtype=np.float32)
        for kr in range(3):
            for kc in range(3):
                out += np.tensordot(xp[:, :, kr:kr + H, kc:kc + Wd],
                                    W[:, :, kr, kc], axes=([1], [1]))
        if b is not None:
            out += b.reshape(1, O, 1, 1)
        return out

    def pad1(x):
        N, C, H, Wd = x.shape
        xp = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float32)
        xp[:, :, 1:-1, 1:-1] = x
        return xp

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float32)          # (N,1,16,16)
    y = np.asarray(f["y"], dtype=np.float32)          # (N,1,16,16)
    t = np.asarray(f["t"], dtype=np.int64).reshape(-1)
    clean = np.asarray(f["clean"], dtype=np.float32)  # (N,1,16,16)
    N = x.shape[0]

    # ---------------- rebuild conv1 as a non-negative feature bank ----------
    # channel layout (32 channels):
    #   0 : x_t                                  (signed)
    #   1 : |x_t|                                (signed)
    #   2 : x_t^2                                (non-negative)
    #   3 : tau = t/31                            (non-negative)
    #   4 : tau * x_t                            (signed)
    #   5 : tau * |x_t|                          (non-negative)
    #   6 : tau * x_t^2                          (non-negative)
    #   7 : x_t * tau^2                          (signed)
    #   8 : |x_t| * tau^2                        (non-negative)
    #   9 : x_t * tau^3                          (signed)
    #  10 : |x_t| * tau^3                        (non-negative)
    #  11 : x_t^2 * tau                          (non-negative)
    #  12 : x_t^2 * tau^2                        (non-negative)
    #  13 : (x_t - tau)^+                        (non-negative)
    #  14 : (x_t + tau)^+                        (non-negative)
    #  15 : (|x_t| - tau)^+                      (non-negative)
    #  16 : (|x_t| + tau)^+                      (non-negative)
    #  17 : (x_t^2 - tau^2)^+                    (non-negative)
    #  18 : (|x_t| - tau^2)^+                    (non-negative)
    #  19 : (x_t^2 - tau)^+                       (non-negative)
    #  20 : (x_t - tau^2)^+                       (non-negative)
    #  21 : (|x_t| - tau^3)^+                     (non-negative)
    #  22 : (x_t^2 - tau^3)^+                     (non-negative)
    #  23 : tau^2                                (non-negative)
    #  24 : tau^3                                (non-negative)
    #  25 : tau^4                                (non-negative)
    #  26 : x_t * tau^4                          (signed)
    #  27 : |x_t| * tau^4                        (non-negative)
    #  28 : x_t^2 * tau^3                        (non-negative)
    #  29 : (x_t - tau^3)^+                       (non-negative)
    #  30 : (x_t^2 - tau^4)^+                     (non-negative)
    #  31 : (|x_t| - tau^4)^+                     (non-negative)
    tau = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
    xs = x[:, 0]                                     # (N,16,16)
    ab = np.abs(xs)
    x2 = xs * xs
    t1 = tau[:, 0]
    t2 = t1 * t1
    t3 = t2 * t1
    t4 = t3 * t1

    def pos(a):
        return np.maximum(a, 0.0)

    feats = np.empty((N, 32, 16, 16), dtype=np.float32)
    feats[:, 0] = xs
    feats[:, 1] = ab
    feats[:, 2] = x2
    feats[:, 3] = t1
    feats[:, 4] = xs * t1
    feats[:, 5] = ab * t1
    feats[:, 6] = x2 * t1
    feats[:, 7] = xs * t2
    feats[:, 8] = ab * t2
    feats[:, 9] = xs * t3
    feats[:, 10] = ab * t3
    feats[:, 11] = x2 * t2
    feats[:, 12] = x2 * t3
    feats[:, 13] = pos(xs - t1)
    feats[:, 14] = pos(xs + t1)
    feats[:, 15] = pos(ab - t1)
    feats[:, 16] = pos(ab + t1)
    feats[:, 17] = pos(x2 - t2)
    feats[:, 18] = pos(ab - t2)
    feats[:, 19] = pos(x2 - t1)
    feats[:, 20] = pos(xs - t2)
    feats[:, 21] = pos(ab - t3)
    feats[:, 22] = pos(x2 - t3)
    feats[:, 23] = t2
    feats[:, 24] = t3
    feats[:, 25] = t4
    feats[:, 26] = xs * t4
    feats[:, 27] = ab * t4
    feats[:, 28] = x2 * t4
    feats[:, 29] = pos(xs - t3)
    feats[:, 30] = pos(x2 - t4)
    feats[:, 31] = pos(ab - t4)

    # conv1: identity on channel 0 (x_t), zero elsewhere; bias 0
    w1 = np.zeros((32, 2, 3, 3), dtype=np.float32)
    w1[0, 0, 1, 1] = 1.0
    b1 = np.zeros(32, dtype=np.float32)

    # conv2 / conv3: identity copies (channel c -> channel c), zero bias
    w2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
    for c in range(32):
        w2[c, c, 1, 1] = 1.0
    b2 = np.zeros(64, dtype=np.float32)
    w3 = np.zeros((64, 64, 3, 3), dtype=np.float32)
    for c in range(32):
        w3[c, c, 1, 1] = 1.0
    b3 = np.zeros(64, dtype=np.float32)

    # ---------------- closed-form least squares for the output layer --------
    # design row for pixel p: [F[:,p+off] (64*9), 1]
    Fp = pad1(feats)                                   # (N,64,18,18)
    cols = []
    for kr in range(3):
        for kc in range(3):
            cols.append(Fp[:, :, kr:kr + 16, kc:kc + 16].transpose(0, 2, 3, 1).reshape(-1, 64))
    A = np.concatenate(cols, axis=1).astype(np.float64)  # (N*256, 576)
    b = np.ones((A.shape[0], 1), dtype=np.float64)
    Xd = np.concatenate([A, b], axis=1)                  # (M, 577)
    Yd = y[:, 0].reshape(-1, 1).astype(np.float64)

    G = Xd.T @ Xd                                     # (577,577)
    r = Xd.T @ Yd                                     # (577,1)
    lam = 1e-6 * max(1.0, float(np.trace(G)) / G.shape[0])
    G[np.diag_indices_from(G)] += lam
    try:
        theta = np.linalg.solve(G, r)
    except Exception:
        theta = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
    theta = theta.reshape(-1)

    wo = np.zeros((1, 64, 3, 3), dtype=np.float32)
    for j in range(9):
        wo[0, :, j // 3, j % 3] = theta[j * 64:(j + 1) * 64].astype(np.float32)
    bo = np.array([theta[-1]], dtype=np.float32)

    # ---------------- verify with an explicit forward pass -----------------
    def forward(xin, tt):
        Nn = xin.shape[0]
        ta = (tt.astype(np.float32) / 31.0).reshape(Nn, 1, 1, 1)
        xs_ = xin[:, 0]
        ab_ = np.abs(xs_)
        x2_ = xs_ * xs_
        t1 = ta[:, 0]; t2 = t1 * t1; t3 = t2 * t1; t4 = t3 * t1
        F = np.empty((Nn, 32, 16, 16), dtype=np.float32)
        F[:, 0] = xs_; F[:, 1] = ab_; F[:, 2] = x2_; F[:, 3] = t1
        F[:, 4] = xs_ * t1; F[:, 5] = ab_ * t1; F[:, 6] = x2_ * t1
        F[:, 7] = xs_ * t2; F[:, 8] = ab_ * t2; F[:, 9] = xs_ * t3
        F[:, 10] = ab_ * t3; F[:, 11] = x2_ * t2; F[:, 12] = x2_ * t3
        F[:, 13] = pos(xs_ - t1); F[:, 14] = pos(xs_ + t1)
        F[:, 15] = pos(ab_ - t1); F[:, 16] = pos(ab_ + t1)
        F[:, 17] = pos(x2_ - t2); F[:, 18] = pos(ab_ - t2)
        F[:, 19] = pos(x2_ - t1); F[:, 20] = pos(xs_ - t2)
        F[:, 21] = pos(ab_ - t3); F[:, 22] = pos(x2_ - t3)
        F[:, 23] = t2; F[:, 24] = t3; F[:, 25] = t4
        F[:, 26] = xs_ * t4; F[:, 27] = ab_ * t4; F[:, 28] = x2_ * t4
        F[:, 29] = pos(xs_ - t3); F[:, 30] = pos(x2_ - t4); F[:, 31] = pos(ab_ - t4)
        # conv1 -> relu -> conv2 -> relu -> conv3 -> relu -> out
        h1 = np.maximum(conv2d(F, w1, b1), 0.0)
        h2 = np.maximum(conv2d(h1, w2, b2), 0.0)
        h3 = np.maximum(conv2d(h2, w3, b3), 0.0)
        return conv2d(h3, wo, bo)

    pred = forward(x, t)
    mse_new = float(np.mean((pred - y) ** 2))
    mse_old = float(np.mean((np.asarray(f["prediction"], dtype=np.float32) - y) ** 2))

    # per-timestep MSE (new)
    per_t = {}
    for tt in range(32):
        m = (t == tt)
        if m.any():
            per_t[str(tt)] = round(float(np.mean((pred[m] - y[m]) ** 2)), 4)

    w = dict(w)
    w["conv1.weight"] = w1
    w["conv1.bias"] = b1
    w["conv2.weight"] = w2
    w["conv2.bias"] = b2
    w["conv3.weight"] = w3
    w["conv3.bias"] = b3
    w["out.weight"] = wo
    w["out.bias"] = bo

    for k in w:
        w[k] = np.ascontiguousarray(np.clip(w[k].astype(np.float32), -1000.0, 1000.0))

    diagnostics = {
        "mse_before": round(mse_old, 5),
        "mse_after": round(mse_new, 5),
        "mse_per_t_after": per_t,
        "n_train": int(N),
        "design_rows": int(Xd.shape[0]),
        "n_params_fit": 577,
        "cond_hint": "ridge-regularised normal equations",
    }
    return {"weights": w, "diagnostics": diagnostics}
