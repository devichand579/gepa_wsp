import numpy as np
import math

def edit(w, f):
    # ---- helpers -------------------------------------------------------
    def conv3x3(x, weight, bias, pad=1):
        # x: (N,C,H,W) ; weight: (O,C,3,3) ; bias: (O,)
        N, C, H, W = x.shape
        O = weight.shape[0]
        xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
        out = np.zeros((N, O, H, W), dtype=np.float32)
        for di in range(3):
            for dj in range(3):
                # weight[:, :, di, dj] : (O, C)
                wslice = weight[:, :, di, dj]  # (O,C)
                # xp[:, :, di:di+H, dj:dj+W] : (N,C,H,W)
                xs = xp[:, :, di:di+H, dj:dj+W]
                # out += einsum('oc,nchw->nohw', wslice, xs)
                out += np.einsum('oc,nchw->nohw', wslice, xs, optimize=True)
        if bias is not None:
            out += bias.reshape(1, O, 1, 1)
        return out

    def relu(x):
        return np.maximum(x, 0.0)

    def build_design(feat, pad=1):
        # feat: (N,C,H,W) -> X: (N*H*W, 9*C) with ordering (di,dj,c)
        N, C, H, W = feat.shape
        fp = np.pad(feat, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
        X = np.empty((N*H*W, 9*C), dtype=np.float32)
        k = 0
        for di in range(3):
            for dj in range(3):
                blk = fp[:, :, di:di+H, dj:dj+W]           # (N,C,H,W)
                X[:, k*C:(k+1)*C] = blk.transpose(0,2,3,1).reshape(N*H*W, C)
                k += 1
        return X

    def fit_readout(feat, y, lam=1e-4, chunk=200):
        # feat: (N,C,H,W) ; y: (N,1,H,W) -> out.weight (1,C,3,3), out.bias (1,)
        N, C, H, W = feat.shape
        D = 9*C
        XtX = np.zeros((D, D), dtype=np.float64)
        Xty = np.zeros(D, dtype=np.float64)
        ysum = 0.0
        yty = 0.0
        ntot = 0
        for s in range(0, N, chunk):
            e = min(N, s+chunk)
            Xc = build_design(feat[s:e])                  # (n*H*W, D)
            yc = y[s:e].transpose(0,2,3,1).reshape(-1).astype(np.float64)  # (n*H*W,)
            Xc64 = Xc.astype(np.float64)
            XtX += Xc64.T @ Xc64
            Xty += Xc64.T @ yc
            ysum += yc.sum()
            yty += float(yc @ yc)
            ntot += yc.shape[0]
        # Solve with intercept: augment with a constant column
        # We'll solve for w and b jointly via normal equations on [X, 1]
        Dp = D + 1
        A = np.zeros((Dp, Dp), dtype=np.float64)
        bvec = np.zeros(Dp, dtype=np.float64)
        A[:D, :D] = XtX
        A[:D, D] = Xty.sum() * 0  # placeholder, fill below
        # Need sum of X columns: recompute cheaply
        # (we already have Xty = X^T y; need X^T 1 = column sums)
        colsum = np.zeros(D, dtype=np.float64)
        for s in range(0, N, chunk):
            e = min(N, s+chunk)
            Xc = build_design(feat[s:e]).astype(np.float64)
            colsum += Xc.sum(axis=0)
        A[:D, D] = colsum
        A[D, :D] = colsum
        A[D, D] = float(ntot)
        bvec[:D] = Xty
        bvec[D] = ysum
        # ridge on weights only (not intercept)
        reg = np.zeros(Dp, dtype=np.float64)
        reg[:D] = lam * np.trace(XtX) / max(D, 1)
        A_reg = A.copy()
        A_reg[np.arange(D), np.arange(D)] += reg[:D]
        try:
            sol = np.linalg.solve(A_reg, bvec)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A_reg, bvec, rcond=None)[0]
        wflat = sol[:D]
        bias = sol[D]
        # reshape: ordering (di,dj,c) -> (3,3,C) -> (C,3,3)
        Wmat = wflat.reshape(3, 3, C).transpose(2, 0, 1).astype(np.float32)  # (C,3,3)
        out_w = Wmat.reshape(1, C, 3, 3)
        out_b = np.array([bias], dtype=np.float32)
        return out_w, out_b

    def mse_of(feat, out_w, out_b, y):
        pred = conv3x3(feat, out_w, out_b)
        return float(np.mean((pred - y) ** 2))

    # ---- current weights ----------------------------------------------
    conv1_w = np.asarray(w['conv1.weight'], dtype=np.float32)
    conv1_b = np.asarray(w['conv1.bias'], dtype=np.float32)
    conv2_w = np.asarray(w['conv2.weight'], dtype=np.float32)
    conv2_b = np.asarray(w['conv2.bias'], dtype=np.float32)
    conv3_w = np.asarray(w['conv3.weight'], dtype=np.float32)
    conv3_b = np.asarray(w['conv3.bias'], dtype=np.float32)
    out_w = np.asarray(w['out.weight'], dtype=np.float32)
    out_b = np.asarray(w['out.bias'], dtype=np.float32)

    features = np.asarray(f['features'], dtype=np.float32)   # (2000,64,16,16)
    y = np.asarray(f['y'], dtype=np.float32)                 # (2000,1,16,16)

    diag = {}

    # ---- baseline MSE with current out on cached features --------------
    base_mse = mse_of(features, out_w, out_b, y)
    diag['base_mse'] = base_mse

    # ---- variant A: refit out only (features unchanged) ----------------
    out_w_A, out_b_A = fit_readout(features, y, lam=1e-5)
    mse_A = mse_of(features, out_w_A, out_b_A, y)
    diag['mse_readout_refit'] = mse_A

    # ---- variant B: enrich conv2/conv3, recompute features, refit out --
    # Build structured non-identity conv2/conv3: spatial smoothing/diffusion
    # + channel mixing, deterministic (no search).  Keep identity path too.
    C1 = conv1_w.shape[0]   # 32
    C2 = conv2_w.shape[0]   # 64
    C3 = conv3_w.shape[0]   # 64

    def make_spatial_kernel(scale):
        # 3x3 kernel: center + small cross/edge terms
        k = np.zeros((3,3), dtype=np.float32)
        k[1,1] = 1.0
        k[0,1] = scale
        k[2,1] = scale
        k[1,0] = scale
        k[1,2] = scale
        return k

    rng = np.random.RandomState(12345)

    # conv2: 32 -> 64.  Keep identity on first 32 channels; add a second
    # "copy + spatial blur" bank on the remaining 32 output channels that
    # mixes channels via a fixed random orthogonal-ish projection.
    conv2_w_B = np.zeros_like(conv2_w)
    for o in range(min(C1, C2)):
        conv2_w_B[o, o, 1, 1] = 1.0
    # extra bank: output channels C1..C2-1 get blurred mixtures of inputs
    extra = C2 - C1
    if extra > 0:
        P = rng.randn(extra, C1).astype(np.float32)
        # normalize rows
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
        sk = make_spatial_kernel(0.25)
        for j in range(extra):
            o = C1 + j
            for c in range(C1):
                conv2_w_B[o, c, :, :] += P[j, c] * sk
    conv2_b_B = np.zeros_like(conv2_b)

    # conv3: 64 -> 64.  Identity plus a small spatial-diffusion mixing.
    conv3_w_B = np.zeros_like(conv3_w)
    for o in range(C3):
        conv3_w_B[o, o, 1, 1] = 1.0
    sk2 = make_spatial_kernel(0.15)
    Q = rng.randn(C3, C3).astype(np.float32) * 0.05
    for o in range(C3):
        for c in range(C3):
            conv3_w_B[o, c, :, :] += Q[o, c] * sk2
    conv3_b_B = np.zeros_like(conv3_b)

    # recompute features through conv1 -> ReLU -> conv2 -> ReLU -> conv3 -> ReLU
    x = np.asarray(f['x'], dtype=np.float32)      # (2000,1,16,16)
    t = np.asarray(f['t'], dtype=np.int64)        # (2000,)
    N = x.shape[0]
    tch = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x, np.broadcast_to(tch, (N,1,16,16))], axis=1)  # (N,2,16,16)

    # Use cached conv1 output if available for speed, else recompute.
    # Recompute in chunks to save memory.
    def recompute_features(chunk=250):
        outs = []
        for s in range(0, N, chunk):
            e = min(N, s+chunk)
            h1 = relu(conv3x3(inp[s:e], conv1_w, conv1_b))
            h2 = relu(conv3x3(h1, conv2_w_B, conv2_b_B))
            h3 = relu(conv3x3(h2, conv3_w_B, conv3_b_B))
            outs.append(h3.astype(np.float32))
        return np.concatenate(outs, axis=0)

    try:
        feat_B = recompute_features()
        out_w_B, out_b_B = fit_readout(feat_B, y, lam=1e-5)
        mse_B = mse_of(feat_B, out_w_B, out_b_B, y)
    except Exception:
        mse_B = float('inf')
        feat_B = None
    diag['mse_enriched_refit'] = mse_B

    # ---- choose best ---------------------------------------------------
    best = 'current'
    best_mse = base_mse
    if mse_A < best_mse - 1e-9:
        best, best_mse = 'readout', mse_A
    if feat_B is not None and mse_B < best_mse - 1e-9:
        best, best_mse = 'enriched', mse_B

    diag['chosen'] = best
    diag['final_mse'] = best_mse

    if best == 'readout':
        w['out.weight'] = out_w_A.astype(np.float32)
        w['out.bias'] = out_b_A.astype(np.float32)
    elif best == 'enriched':
        w['conv2.weight'] = conv2_w_B.astype(np.float32)
        w['conv2.bias'] = conv2_b_B.astype(np.float32)
        w['conv3.weight'] = conv3_w_B.astype(np.float32)
        w['conv3.bias'] = conv3_b_B.astype(np.float32)
        w['out.weight'] = out_w_B.astype(np.float32)
        w['out.bias'] = out_b_B.astype(np.float32)
    # else keep current weights

    # ---- safety: clamp / finite ----------------------------------------
    for k in list(w.keys()):
        arr = np.asarray(w[k])
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(arr, -1000.0, 1000.0).astype(np.float32)
            w[k] = arr

    return {"weights": w, "diagnostics": diag}
