import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def edit(w, f):
    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def conv2d(x, W, b):
        """x: (N,C,H,W), W: (O,C,kh,kw) -> (N,O,H,W) cross-correlation, pad 1."""
        N, C, H, Wd = x.shape
        O, C2, kh, kw = W.shape
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        win = sliding_window_view(xp, (kh, kw), axis=(2, 3))      # (N,C,H,W,kh,kw)
        win = win.transpose(0, 2, 3, 1, 4).reshape(N * H * Wd, C * kh * kw)
        Wf = W.reshape(O, C * kh * kw).T
        out = win @ Wf + b[None, :]
        return out.reshape(N, H, Wd, O).transpose(0, 3, 1, 2)

    def ridge_solve(A, B, lam):
        """A:(n,p) B:(n,q) -> (p,q) ridge solution, plus intercept handled by caller."""
        p = A.shape[1]
        G = A.T @ A
        G.flat[:: p + 1] += lam
        try:
            return np.linalg.solve(G, A.T @ B)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(G, A.T @ B, rcond=None)[0]

    x = np.asarray(f["x"], dtype=np.float64)            # (N,1,16,16)
    t = np.asarray(f["t"]).astype(np.int64).ravel()     # (N,)
    y = np.asarray(f["y"], dtype=np.float64)            # (N,1,16,16)
    abar = np.asarray(f["alphabar"], dtype=np.float64)  # (32,)

    N = x.shape[0]
    H = Wd = 16
    P = H * Wd

    # ------------------------------------------------------------------
    # 1) empirical noise statistics from the training pairs
    # ------------------------------------------------------------------
    a = np.sqrt(np.clip(abar, 1e-8, 1.0))
    b = np.sqrt(np.clip(1.0 - abar, 1e-8, 1.0))
    # per-pixel variance of the clean images (proxy for sigma0^2)
    var0 = float(np.var(x.reshape(N, -1), axis=0, ddof=1).mean())
    var0 = max(var0, 1e-4)

    # ------------------------------------------------------------------
    # 2) build an affine (ReLU-inactive) feature extractor
    #    conv1: +-delta on x-channel + large bias  -> affine in x
    #    conv2/conv3: +-delta with large bias      -> affine pass-through
    #    out: per-timestep-group linear map of the conv3 features
    # ------------------------------------------------------------------
    M = 6.0          # keeps every pre-activation strictly positive
    EPS = 1e-3       # small positive slope for the t channel

    # ---- conv1 -------------------------------------------------------
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float64)
    c1b = np.full(32, M, dtype=np.float64)
    for k in range(16):
        c1w[k, 0, 1, 1] = 1.0        # +x  -> x + M
        c1w[16 + k, 0, 1, 1] = -1.0   # -x  -> -x + M
    for k in range(8):
        c1w[24 + k, 1, 1, 1] = EPS   # t channel (constant in space)
    c1b[24:] = M

    # ---- conv2 / conv3 : affine pass-through with sign flips ---------
    def passthrough(nout, nin):
        Wt = np.zeros((nout, nin, 3, 3), dtype=np.float64)
        bt = np.full(nout, M, dtype=np.float64)
        for k in range(nout):
            Wt[k, k % nin, 1, 1] = 1.0 if (k % 2 == 0) else -1.0
        return Wt, bt

    c2w, c2b = passthrough(64, 32)
    c3w, c3b = passthrough(64, 64)

    # forward pass (features for the whole training set)
    tt = (t.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x, np.broadcast_to(tt, (N, 1, H, Wd))], axis=1)
    h1 = np.maximum(conv2d(inp, c1w, c1b), 0.0)
    h2 = np.maximum(conv2d(h1, c2w, c2b), 0.0)
    h3 = np.maximum(conv2d(h2, c3w, c3b), 0.0)     # (N,64,16,16) = out-layer input

    # ------------------------------------------------------------------
    # 3) per-timestep-group closed-form ridge fit of the output layer
    # ------------------------------------------------------------------
    # design matrix rows: (n, i, j) ; columns: (c, dy, dx)
    feat = h3.transpose(0, 2, 3, 1).reshape(N, P, 64)              # (N,P,64)
    tgt = y.reshape(N, P, 1)

    groups = []
    for lo, hi in [(0, 8), (8, 16), (16, 24), (24, 32)]:
        idx = np.where((t >= lo) & (t < hi))[0]
        if idx.size >= 8:
            groups.append(idx)
    if len(groups) == 0:
        groups = [np.arange(N)]

    out_w = np.zeros((1, 64, 3, 3), dtype=np.float64)
    out_b = np.zeros(1, dtype=np.float64)

    # design: for each group build A (n*P, 576) via shifted feature maps
    def build_design(feat_g):
        # feat_g: (n,P,64) -> (n*P, 64*9) with 3x3 neighbourhood (zero pad)
        n = feat_g.shape[0]
        fg = feat_g.reshape(n, H, Wd, 64)
        fpad = np.pad(fg, ((0, 0), (1, 1), (1, 1), (0, 0)))
        cols = []
        for dy in range(3):
            for dx in range(3):
                cols.append(fpad[:, dy:dy + H, dx:dx + Wd, :])
        A = np.stack(cols, axis=2).reshape(n * P, 64 * 9)
        return A

    for gi, idx in enumerate(groups):
        fg = feat[idx]
        tg = tgt[idx]
        A = build_design(fg)
        B = tg.reshape(-1, 1)
        mu = A.mean(axis=0)
        ymu = B.mean(axis=0)
        Ac = A - mu
        Bc = B - ymu
        lam = 1e-6 * max(1.0, float(np.mean(np.sum(Ac * Ac, axis=0))))
        sol = ridge_solve(Ac, Bc, lam)
        bias = ymu - (mu @ sol)
        # scatter into the 64x3x3 output filter
        filt = sol.reshape(64, 3, 3)
        out_w[0] += filt
        out_b[0] += bias[0] / len(groups)

    # ------------------------------------------------------------------
    # 4) sanity check: forward pass with the new weights, MSE estimate
    # ------------------------------------------------------------------
    def forward(xin, tin, W1, B1, W2, B2, W3, B3, W4, B4):
        n = xin.shape[0]
        tt = (tin.astype(np.float64) / 31.0).reshape(n, 1, 1, 1)
        ii = np.concatenate([xin, np.broadcast_to(tt, (n, 1, H, Wd))], axis=1)
        h1 = np.maximum(conv2d(ii, W1, B1), 0.0)
        h2 = np.maximum(conv2d(h1, W2, B2), 0.0)
        h3 = np.maximum(conv2d(h2, W3, B3), 0.0)
        o = conv2d(h3, W4, B4)
        return o

    pred = forward(x, t, c1w, c1b, c2w, c2b, c3w, c3b, out_w, out_b)
    mse_new = float(np.mean((pred - y) ** 2))
    mse_old = float(np.mean((np.asarray(f["prediction"], dtype=np.float64) - y) ** 2))

    # ------------------------------------------------------------------
    # 5) write back, preserving keys/shapes/dtypes and value bounds
    # ------------------------------------------------------------------
    def put(key, arr):
        ref = w[key]
        a32 = np.asarray(arr, dtype=np.float32)
        a32 = np.clip(a32, -1000.0, 1000.0)
        a32 = np.nan_to_num(a32, nan=0.0, posinf=1000.0, neginf=-1000.0)
        w[key] = a32.reshape(ref.shape).astype(ref.dtype)

    put("conv1.weight", c1w)
    put("conv1.bias", c1b)
    put("conv2.weight", c2w)
    put("conv2.bias", c2b)
    put("conv3.weight", c3w)
    put("conv3.bias", c3b)
    put("out.weight", out_w)
    put("out.bias", out_b)

    diagnostics = {
        "mse_before": round(mse_old, 5),
        "mse_after_train": round(mse_new, 5),
        "var0_pixel": round(var0, 5),
        "n_groups": len(groups),
        "out_w_absmax": round(float(np.max(np.abs(out_w))), 5),
        "relu_inactive_frac": round(float(np.mean(h1 <= 1e-9) + np.mean(h2 <= 1e-9)
                                          + np.mean(h3 <= 1e-9)) / 3.0, 5),
    }
    return {"weights": w, "diagnostics": diagnostics}
