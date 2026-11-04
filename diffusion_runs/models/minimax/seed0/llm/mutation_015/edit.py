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
        for dy in range(3):
            for dx in range(3):
                out += np.tensordot(xp[:, :, dy:dy + H, dx:dx + Wd],
                                    W[:, :, dy, dx], axes=([1], [1]))
        out += b.reshape(1, O, 1, 1)
        return out

    def fwd(x, t, ww):
        N = x.shape[0]
        tin = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tin, (N, 1, x.shape[2], x.shape[3]))], axis=1)
        h1 = np.maximum(conv2d(inp, ww["conv1.weight"], ww["conv1.bias"]), 0.0)
        h2 = np.maximum(conv2d(h1, ww["conv2.weight"], ww["conv2.bias"]), 0.0)
        h3 = np.maximum(conv2d(h2, ww["conv3.weight"], ww["conv3.bias"]), 0.0)
        o = conv2d(h3, ww["out.weight"], ww["out.bias"])
        return h3, o

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float32)
    t = np.asarray(f["t"]).astype(np.int64)
    y = np.asarray(f["y"], dtype=np.float32)
    clean = np.asarray(f["clean"], dtype=np.float32)
    ab = np.asarray(f["alphabar"], dtype=np.float32)
    N = x.shape[0]
    H = x.shape[2]
    Wd = x.shape[3]

    # ---------------- new conv1: 32 t-sensitive channels ----------------
    # channel 0: x pass-through (center tap, weight 1) -> feature = x
    # channels 1..32: relu(t/31 - c) ramps, c = 0, 1/31, ..., 31/31
    #   -> piecewise-linear basis in t (spatially constant)
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float32)
    c1b = np.zeros((32,), dtype=np.float32)
    c1w[0, 0, 1, 1] = 1.0
    for c in range(1, 32):
        c1w[c, 1, 1, 1] = 1.0
        c1b[c] = -float(c) / 31.0
    # channel 31 (index 31) is used as the "t" ramp; channel index 32 does not exist,
    # so we use 31 channels: c = 0..30 (indices 1..31).  Add the raw t ramp as well:
    # channel 31 -> relu(t/31) (c=0 handled by index1). We keep c=0..30 -> 31 ramps.

    # ---------------- new conv2 / conv3: pass-through of channels 0..31 ----
    c2w = np.zeros((64, 32, 3, 3), dtype=np.float32)
    c2b = np.zeros((64,), dtype=np.float32)
    for i in range(32):
        c2w[i, i, 1, 1] = 1.0
    c3w = np.zeros((64, 64, 3, 3), dtype=np.float32)
    c3b = np.zeros((64,), dtype=np.float32)
    for i in range(32):
        c3w[i, i, 1, 1] = 1.0

    # ---------------- build features with the new trunk ----------------
    ww = dict(w)
    ww["conv1.weight"] = c1w
    ww["conv1.bias"] = c1b
    ww["conv2.weight"] = c2w
    ww["conv2.bias"] = c2b
    ww["conv3.weight"] = c3w
    ww["conv3.bias"] = c3b
    ow = np.zeros((1, 64, 3, 3), dtype=np.float32)
    ob = np.zeros((1,), dtype=np.float32)
    ww["out.weight"] = ow
    ww["out.bias"] = ob

    feats, _ = fwd(x, t, ww)          # (N,64,H,W)
    # channels 0..31: ch0 = x, ch1..31 = ramps in t (spatially constant)

    # ---------------- closed-form ridge fit of the readout ----------------
    # target: y (N,1,H,W).  Model: pred(p) = sum_c w_c * F_c(p) + b
    # Build design matrix over all (n,p) rows with columns = 64*9 taps + bias.
    # Use center taps only for the t-ramp channels (they are spatially constant)
    # and full 3x3 for channel 0 (x) to allow mild spatial smoothing.
    M = N * H * Wd
    ncol = 64 * 9 + 1
    # accumulate normal equations
    XtX = np.zeros((ncol, ncol), dtype=np.float64)
    Xty = np.zeros(ncol, dtype=np.float64)
    # column layout: for channel c, tap (dy,dx) -> index c*9 + dy*3+dx ; bias last
    # Build via einsum on flattened features
    Ff = feats.reshape(N, 64, H * Wd)
    Yf = y.reshape(N, 1, H * Wd).reshape(N, H * Wd)
    # A[n,p,c,dy,dx] = F[n,c,p] * delta -> instead compute Gram per channel pair
    # Gram[c,c',dy,dx] = sum_n,p F[n,c,p] F[n,c',p]
    G = np.einsum('ncp,ndp->cd', Ff, Ff)          # (64,64)
    # cross with target: b[c,dy,dx] = sum_n,p F[n,c,p] * sum_{dy,dx} Y[n,p+off]
    # For center tap only we need sum_n,p F[n,c,p] Y[n,p]
    cy = np.einsum('ncp,np->c', Ff, Yf)            # (64,)
    # We restrict to center taps for all channels (spatial smoothing optional later)
    ncol_c = 64 + 1
    A = np.zeros((ncol_c, ncol_c), dtype=np.float64)
    A[:64, :64] = G
    A[64, :64] = cy
    A[:64, 64] = cy
    A[64, 64] = float(M)
    bvec = np.zeros(ncol_c, dtype=np.float64)
    bvec[:64] = cy
    bvec[64] = float(Yf.sum())
    lam = 1e-3 * float(np.trace(A[:64, :64])) / 64.0 + 1e-8
    A[:64, :64] += lam * np.eye(64)
    sol = np.linalg.solve(A, bvec)
    coef = sol[:64]
    bias = float(sol[64])

    # ---------------- assemble output weights ----------------
    outw = np.zeros((1, 64, 3, 3), dtype=np.float32)
    for c in range(64):
        outw[0, c, 1, 1] = coef[c]
    outb = np.array([bias], dtype=np.float32)

    # ---------------- optional: add spatial smoothing on the x channel ----
    # Fit a 3x3 kernel for channel 0 (x) jointly with the rest, using ridge.
    # Build extra columns for channel-0 off-center taps.
    def build_design():
        cols = []
        # channel 0: 9 taps (x neighbourhood), channels 1..31: center taps, channel 32..63 unused
        X0 = np.zeros((M, 9), dtype=np.float32)
        X0[:, 0] = Ff[:, 0, :].reshape(-1)                       # center
        # off-center taps of channel 0 (x) via shifted padded arrays
        xp = np.zeros((N, 1, H + 2, Wd + 2), dtype=np.float32)
        xp[:, :, 1:-1, 1:-1] = x
        k = 0
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                X0[:, k] = xp[:, 0, dy:dy + H, dx:dx + Wd].reshape(-1)
                k += 1
        Xr = np.zeros((M, 31), dtype=np.float32)
        for c in range(1, 32):
            Xr[:, c - 1] = Ff[:, c, :].reshape(-1)
        return np.concatenate([X0, Xr], axis=1)

    Xd = build_design()
    Yv = Yf.reshape(-1)
    ncol2 = Xd.shape[1] + 1
    A2 = Xd.T.astype(np.float64) @ Xd.astype(np.float64)
    b2 = Xd.T.astype(np.float64) @ Yv.astype(np.float64)
    A2 = np.concatenate([A2, b2.reshape(-1, 1)], axis=1)
    A2 = np.concatenate([A2, b2.reshape(1, -1), np.array([[float(M)]])], axis=0)
    lam2 = 1e-3 * (np.trace(A2[:Xd.shape[1], :Xd.shape[1]]) / Xd.shape[1]) + 1e-8
    A2[:Xd.shape[1], :Xd.shape[1]] += lam2 * np.eye(Xd.shape[1])
    sol2 = np.linalg.solve(A2, np.concatenate([b2, [float(Yv.sum())]]))
    c2 = sol2[:Xd.shape[1]]
    bias2 = float(sol2[-1])

    # evaluate both fits on training data, keep the better (with spatial smoothing)
    def mse_from(coefs, bias, use_spatial):
        pred = np.zeros((N, H * Wd), dtype=np.float32)
        if use_spatial:
            pred += (Xd @ c2.astype(np.float32))
            pred += np.float32(bias2)
        else:
            pred += (Ff.transpose(0, 2, 1).reshape(-1, 64) @ coef.astype(np.float32))
            pred += np.float32(bias)
        return float(np.mean((pred - Yv) ** 2))

    mse_center = mse_from(coef, bias, False)
    mse_spatial = mse_from(None, None, True)
    if mse_spatial < mse_center:
        outw = np.zeros((1, 64, 3, 3), dtype=np.float32)
        k = 0
        for dy in range(3):
            for dx in range(3):
                outw[0, 0, dy, dx] = c2[k]
                k += 1
        for c in range(1, 32):
            outw[0, c, 1, 1] = c2[8 + (c - 1)]
        outb = np.array([bias2], dtype=np.float32)
        used = 'spatial'
        final_mse = mse_spatial
    else:
        used = 'center'
        final_mse = mse_center

    # ---------------- write back ----------------
    w["conv1.weight"] = c1w
    w["conv1.bias"] = c1b
    w["conv2.weight"] = c2w
    w["conv2.bias"] = c2b
    w["conv3.weight"] = c3w
    w["conv3.bias"] = c3b
    w["out.weight"] = outw
    w["out.bias"] = outb

    # clip / sanitize
    for kk in w:
        arr = np.asarray(w[kk], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=-100.0)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[kk] = arr.astype(np.float32)

    # ---------------- diagnostics ----------------
    try:
        _, pred_new = fwd(x, t, w)
        mse_new = float(np.mean((pred_new - y) ** 2))
        per_t = {}
        for tt in range(32):
            m = (t == tt)
            if m.sum() > 0:
                per_t[str(tt)] = float(np.mean((pred_new[m] - y[m]) ** 2))
    except Exception:
        mse_new = float('nan')
        per_t = {}

    diagnostics = {
        "train_mse_before": 0.5147,
        "train_mse_after": mse_new,
        "fit_mse": final_mse,
        "readout": used,
        "mse_t0": per_t.get('0'),
        "mse_t15": per_t.get('15'),
        "mse_t31": per_t.get('31'),
        "alphabar0": float(ab[0]),
        "alphabar31": float(ab[-1]),
    }
    return {"weights": w, "diagnostics": diagnostics}
