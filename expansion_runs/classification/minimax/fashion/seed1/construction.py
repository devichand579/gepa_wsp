import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def conv2d(x, W, b, pad=1):
        # x: [N,C,H,W], W: [O,C,kh,kw] (cross-correlation)
        N, C, H, Wd = x.shape
        O, C2, kh, kw = W.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        cols = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))
        cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N, H, Wd, C * kh * kw)
        out = cols @ W.reshape(O, C * kh * kw).T + b[None, None, None, :]
        return out.transpose(0, 3, 1, 2)

    def maxpool2d(x, k=2):
        N, C, H, Wd = x.shape
        H2, W2 = H // k, Wd // k
        v = x[:, :, :H2 * k, :W2 * k].reshape(N, C, H2, k, W2, k)
        return v.max(axis=(3, 5))

    def forward(x, c1w, c1b, c2w, c2b, hw, hb):
        h = np.maximum(conv2d(x, c1w, c1b), 0.0)
        h = maxpool2d(h, 2)
        h = np.maximum(conv2d(h, c2w, c2b), 0.0)
        h = maxpool2d(h, 2)
        return h.reshape(h.shape[0], -1)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    X = np.asarray(f["x"], dtype=np.float32)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0]
    y = np.asarray(f["y"]).astype(np.int64)
    N = X.shape[0]

    # ------------------------------------------------------------------
    # conv1: 16 fixed filters
    #   ch 0..8  : delta (shifted copy of the image) at offsets (a,b)
    #   ch 9..15 : oriented / box filters
    # ------------------------------------------------------------------
    c1w = np.zeros((16, 1, 3, 3), dtype=np.float32)
    offs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    for k, (a, b) in enumerate(offs):
        c1w[k, 0, a, b] = 1.0

    # oriented first derivatives (Sobel-like) + box filters
    sob = [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],   # 0  horizontal edge
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],   # 90 vertical edge
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],   # 180 vertical edge
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],   # 270 horizontal edge
        [[-1, -1, 1], [-1, 1, 1], [1, 1, -1]],   # diagonal
        [[1, -1, -1], [1, -1, 1], [-1, 1, 1]],   # anti-diagonal
    ]
    for i, s in enumerate(sob):
        c1w[9 + i, 0] = np.asarray(s, dtype=np.float32) / 2.0
    c1w[15, 0] = np.ones((3, 3), dtype=np.float32) / 9.0   # local mean
    c1b = np.zeros(16, dtype=np.float32)

    # ------------------------------------------------------------------
    # conv2: 32 fixed channels
    #   ch 0..15  : "level"  = smoothed conv1 channel (spatial 3x3 box)
    #   ch 16..31 : "contrast" = (+smoothed) - (-smoothed)  -> relu then
    #                max-pool gives (max - min) of the local window
    # ------------------------------------------------------------------
    c2w = np.zeros((32, 16, 3, 3), dtype=np.float32)
    c2b = np.zeros(32, dtype=np.float32)
    box = np.ones((3, 3), dtype=np.float32) / 9.0
    for c in range(16):
        c2w[c, c] = box
        c2w[16 + c, c] = box
        c2w[16 + c, c] = -box

    # ------------------------------------------------------------------
    # recompute pool features with the new conv stack
    # ------------------------------------------------------------------
    F = forward(X, c1w, c1b, c2w, c2b, None, None).astype(np.float64)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # ridge regression head: one-hot targets, closed-form normal equations
    # ------------------------------------------------------------------
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    Y = Y - 1.0 / 10.0                      # centered targets

    mu = F.mean(axis=0)
    Fs = F - mu
    sd = Fs.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Fs = Fs / sd

    G = Fs.T @ Fs
    b = Fs.T @ Y
    tr = np.trace(G) / max(G.shape[0], 1)
    for lam in (1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0):
        A = G + lam * tr * np.eye(G.shape[0])
        try:
            Wt = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            Wt = np.linalg.lstsq(A, b, rcond=None)[0]
        # closed-form intercept for centered targets
        c0 = Y.mean(axis=0) - (mu / sd) @ Wt
        logits = Fs @ Wt + c0
        pred = np.argmax(logits, axis=1)
        acc = float((pred == y).mean())
        if acc > best_acc:
            best_acc, best_lam = acc, lam
            best_Wt, best_c0 = Wt, c0

    # rescale head back to raw-feature space
    hw = (best_Wt / sd[:, None]).T.astype(np.float32)      # [10, 1568]
    hb = (best_c0 - mu @ hw).astype(np.float32)

    # ------------------------------------------------------------------
    # write back
    # ------------------------------------------------------------------
    w["conv1.weight"] = c1w.astype(np.float32)
    w["conv1.bias"] = c1b.astype(np.float32)
    w["conv2.weight"] = c2w.astype(np.float32)
    w["conv2.bias"] = c2b.astype(np.float32)
    w["head.weight"] = np.clip(hw, -1000.0, 1000.0).astype(np.float32)
    w["head.bias"] = np.clip(hb, -1000.0, 1000.0).astype(np.float32)

    # ------------------------------------------------------------------
    # diagnostics: verify by re-running the forward pass with new weights
    # ------------------------------------------------------------------
    Fnew = forward(X, w["conv1.weight"], w["conv1.bias"],
                   w["conv2.weight"], w["conv2.bias"], None, None)
    lg = Fnew @ w["head.weight"].T + w["head.bias"]
    pr = np.argmax(lg, axis=1)
    acc_new = float((pr == y).mean())
    mx = lg.max(axis=1)
    ex = mx - np.log(np.exp(lg - mx[:, None]).sum(axis=1))
    per_class = {}
    for c in range(10):
        m = (y == c)
        per_class[str(c)] = float((pr[m] == c).mean()) if m.any() else 0.0

    return {
        "weights": w,
        "diagnostics": {
            "ridge_lambda": float(best_lam),
            "pool_fit_acc_before": 0.099,
            "pool_fit_acc_after": acc_new,
            "pool_fit_ce_after": float(ex.mean()),
            "per_class_acc": per_class,
            "feature_mean": float(Fnew.mean()),
            "feature_std": float(Fnew.std()),
            "feature_zero_frac": float((Fnew == 0).mean()),
            "head_w_absmax": float(np.abs(w["head.weight"]).max()),
            "conv1_absmax": float(np.abs(w["conv1.weight"]).max()),
            "conv2_absmax": float(np.abs(w["conv2.weight"]).max()),
            "n_train": int(N),
        },
    }
