import numpy as np

def edit(w, f):
    # ---------- helpers ----------
    def conv_forward(x, weight, bias, pad=1):
        # x: [N,Cin,H,W], weight: [Cout,Cin,3,3]
        N, Cin, H, W = x.shape
        Cout = weight.shape[0]
        xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
        out = np.zeros((N, Cout, H, W), dtype=np.float32)
        for di in range(3):
            for dj in range(3):
                patch = xp[:, :, di:di+H, dj:dj+W]  # [N,Cin,H,W]
                wk = weight[:, :, di, dj]           # [Cout,Cin]
                out += np.einsum('nchw,oc->nohw', patch, wk, optimize=True)
        out += bias.reshape(1, -1, 1, 1)
        return out

    def relu(x):
        return np.maximum(x, 0.0)

    # ---------- inputs ----------
    x_all = np.asarray(f['x'], dtype=np.float32)          # [2000,1,16,16]
    t_all = np.asarray(f['t'], dtype=np.float32)          # [2000]
    y_all = np.asarray(f['y'], dtype=np.float32)          # [2000,1,16,16]
    feats = np.asarray(f['features'], dtype=np.float32)   # [2000,64,16,16]
    N = x_all.shape[0]

    cw1 = np.asarray(w['conv1.weight'], dtype=np.float32)
    cb1 = np.asarray(w['conv1.bias'], dtype=np.float32)
    cw2 = np.asarray(w['conv2.weight'], dtype=np.float32)
    cb2 = np.asarray(w['conv2.bias'], dtype=np.float32)
    cw3 = np.asarray(w['conv3.weight'], dtype=np.float32)
    cb3 = np.asarray(w['conv3.bias'], dtype=np.float32)
    ow = np.asarray(w['out.weight'], dtype=np.float32)
    ob = np.asarray(w['out.bias'], dtype=np.float32)

    # ---------- build conv1 input ----------
    tch = (t_all / 31.0).reshape(N, 1, 1, 1).astype(np.float32)
    tch = np.broadcast_to(tch, (N, 1, 16, 16)).astype(np.float32)
    inp = np.concatenate([x_all, tch], axis=1)  # [N,2,16,16]

    # conv1 output (cached for reuse)
    a1 = relu(conv_forward(inp, cw1, cb1))  # [N,32,16,16]

    # ---------- readout refit on cached features ----------
    def fit_readout(feat_arr, y_arr, lam=1e-4):
        # feat_arr: [N,64,16,16], y_arr: [N,1,16,16]
        # design: for each pixel, 64*9 inputs (3x3 neighborhood, zero padded)
        Nf = feat_arr.shape[0]
        H = W = 16
        fp = np.pad(feat_arr, ((0,0),(0,0),(1,1),(1,1)))  # [N,64,18,18]
        # Build X: [N*H*W, 64*9]
        # We'll accumulate normal equations in chunks.
        D = 64 * 9
        XtX = np.zeros((D, D), dtype=np.float64)
        Xty = np.zeros((D, 1), dtype=np.float64)
        yv = y_arr.reshape(Nf, -1)  # [N,256]
        chunk = 200
        for s in range(0, Nf, chunk):
            e = min(s + chunk, Nf)
            fc = fp[s:e]  # [n,64,18,18]
            n = e - s
            # build design [n*256, 576]
            cols = []
            for di in range(3):
                for dj in range(3):
                    # [n,64,16,16] -> [n,256,64]
                    blk = fc[:, :, di:di+H, dj:dj+W]  # [n,64,16,16]
                    blk = np.transpose(blk, (0, 2, 3, 1)).reshape(n * H * W, 64)
                    cols.append(blk)
            X = np.concatenate(cols, axis=1).astype(np.float64)  # [n*256, 576]
            XtX += X.T @ X
            Xty += X.T @ yv[s:e].reshape(-1, 1).astype(np.float64)
        XtX += lam * np.eye(D)
        sol = np.linalg.solve(XtX, Xty)  # [576,1]
        Wr = sol.reshape(3, 3, 64).transpose(2, 0, 1)  # [64,3,3] -> need [1,64,3,3]
        # Actually sol ordering: di,dj,c -> index = (di*3+dj)*64 + c
        Wr = np.zeros((1, 64, 3, 3), dtype=np.float32)
        for di in range(3):
            for dj in range(3):
                Wr[0, :, di, dj] = sol[(di * 3 + dj) * 64:(di * 3 + dj) * 64 + 64, 0]
        br = np.zeros((1,), dtype=np.float32)
        return Wr, br

    def predict_with(feat_arr, Wr, br):
        # conv 3x3 pad1 with Wr [1,64,3,3]
        out = conv_forward(feat_arr, Wr, br)
        return out

    def mse(pred, y):
        return float(np.mean((pred - y) ** 2))

    # refit on cached features
    Wr_cached, br_cached = fit_readout(feats, y_all)
    pred_cached = predict_with(feats, Wr_cached, br_cached)
    mse_cached = mse(pred_cached, y_all)

    # current prediction MSE for reference
    pred_cur = np.asarray(f['prediction'], dtype=np.float32)
    mse_cur = mse(pred_cur, y_all)

    # ---------- enriched variant: structured conv2/conv3 ----------
    # Build conv2 as: identity passthrough for first 32 channels (1->1),
    # plus a spatial diffusion (center + 4-neighbor) and channel mixing.
    def make_conv2():
        W2 = np.zeros((64, 32, 3, 3), dtype=np.float32)
        # identity passthrough: out c <- in c for c<32
        for c in range(32):
            W2[c, c, 1, 1] = 1.0
        # diffusion + mixing for out channels 32..63
        rng = np.random.RandomState(1234)
        for o in range(32, 64):
            # random channel mixing over a few inputs
            idx = rng.randint(0, 32, size=6)
            for c in idx:
                W2[o, c, 1, 1] += rng.randn() * 0.35
            # spatial diffusion
            W2[o, (o - 32) % 32, 0, 1] += 0.25
            W2[o, (o - 32) % 32, 1, 0] += 0.25
            W2[o, (o - 32) % 32, 1, 2] += 0.25
            W2[o, (o - 32) % 32, 2, 1] += 0.25
            W2[o, (o - 32) % 32, 1, 1] += -0.2
        return W2

    def make_conv3():
        W3 = np.zeros((64, 64, 3, 3), dtype=np.float32)
        for c in range(64):
            W3[c, c, 1, 1] = 1.0
        rng = np.random.RandomState(5678)
        for o in range(64):
            idx = rng.randint(0, 64, size=4)
            for c in idx:
                W3[o, c, 1, 1] += rng.randn() * 0.25
            W3[o, o, 0, 1] += 0.15
            W3[o, o, 1, 0] += 0.15
            W3[o, o, 1, 2] += 0.15
            W3[o, o, 2, 1] += 0.15
        return W3

    W2n = make_conv2()
    W3n = make_conv3()

    # recompute features for all examples
    a2 = relu(conv_forward(a1, W2n, cb2))
    a3 = relu(conv_forward(a2, W3n, cb3))

    Wr_enr, br_enr = fit_readout(a3, y_all)
    pred_enr = predict_with(a3, Wr_enr, br_enr)
    mse_enr = mse(pred_enr, y_all)

    # choose best
    best = 'cached'
    best_mse = mse_cached
    if mse_enr < best_mse:
        best = 'enriched'
        best_mse = mse_enr

    # also compare to current out on cached features (sanity)
    if mse_cur < best_mse:
        best = 'current'
        best_mse = mse_cur

    if best == 'current':
        # keep everything as-is
        pass
    elif best == 'cached':
        w['out.weight'] = Wr_cached.astype(np.float32)
        w['out.bias'] = br_cached.astype(np.float32)
    else:
        w['conv2.weight'] = W2n.astype(np.float32)
        w['conv3.weight'] = W3n.astype(np.float32)
        w['out.weight'] = Wr_enr.astype(np.float32)
        w['out.bias'] = br_enr.astype(np.float32)

    # sanitize
    for k in list(w.keys()):
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[k] = arr.astype(np.float32)

    diag = {
        'mse_current': mse_cur,
        'mse_cached_refit': mse_cached,
        'mse_enriched_refit': mse_enr,
        'chosen': best,
        'final_mse_estimate': best_mse,
    }
    return {"weights": w, "diagnostics": diag}
