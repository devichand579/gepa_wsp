import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: (N,C,H,W), W: (O,C,kh,kw) cross-correlation, pad 1
        N, C, H, Wd = x.shape
        O, C2, kh, kw = W.shape
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        cols = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))
        cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * Wd, C * kh * kw)
        out = cols @ W.reshape(O, C * kh * kw).T
        out = out.reshape(N, H, Wd, O).transpose(0, 3, 1, 2)
        return out + b.reshape(1, O, 1, 1)

    def forward(x, t, W):
        tb = (t.astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (x.shape[0], 1, x.shape[1], x.shape[2]))], axis=1)
        h1 = np.maximum(conv2d(inp, W['conv1.weight'], W['conv1.bias']), 0)
        h2 = np.maximum(conv2d(h1, W['conv2.weight'], W['conv2.bias']), 0)
        h3 = np.maximum(conv2d(h2, W['conv3.weight'], W['conv3.bias']), 0)
        return conv2d(h3, W['out.weight'], W['out.bias'])

    def feats(x, t, W):
        tb = (t.astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (x.shape[0], 1, x.shape[1], x.shape[2]))], axis=1)
        h1 = np.maximum(conv2d(inp, W['conv1.weight'], W['conv1.bias']), 0)
        h2 = np.maximum(conv2d(h1, W['conv2.weight'], W['conv2.bias']), 0)
        return np.maximum(conv2d(h2, W['conv3.weight'], W['conv3.bias']), 0)

    W = {k: np.array(v, dtype=np.float64) for k, v in w.items()}
    diag = {}

    x = np.asarray(f['x'], dtype=np.float64)
    t = np.asarray(f['t'])
    y = np.asarray(f['y'], dtype=np.float64)
    N = x.shape[0]

    # ---------------- 1) verify forward implementation on the traced batch ----------------
    bidx = np.asarray(f['batch.ids'], dtype=np.int64)
    bx = np.asarray(f['batch.x'], dtype=np.float64)
    bt = np.asarray(f['batch.t'])
    bpred = np.asarray(f['prediction'][bidx], dtype=np.float64)
    myp = forward(bx, bt, W)
    diag['fwd_max_abs_err_batch'] = float(np.max(np.abs(myp - bpred)))
    ftr = np.asarray(f['features'], dtype=np.float64)
    myf = feats(bx, bt, W)
    diag['feat_max_abs_err_batch'] = float(np.max(np.abs(myf - ftr)))

    # ---------------- 2) closed-form optimal last layer on cached features ----------------
    # design matrix: for each example/pixel, the 64*3*3 = 576 receptive-field values of conv3 output
    def design_from_features(F):
        # F: (N,64,16,16) -> (N*256, 576)
        Nn = F.shape[0]
        Fp = np.pad(F, ((0, 0), (0, 0), (1, 1), (1, 1)))
        cols = np.lib.stride_tricks.sliding_window_view(Fp, (3, 3), axis=(2, 3))
        cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(Nn * 256, 64 * 9)
        return cols

    Ffull = ftr  # cached features for ALL training examples (parent state)
    A = design_from_features(Ffull)
    Yv = y.reshape(N * 256)
    # ridge-regularised normal equations (tiny ridge for conditioning)
    lam = 1e-3 * float(np.mean(np.sum(A * A, axis=1))) + 1e-8
    G = A.T @ A
    G.flat[:: G.shape[0] + 1] += lam
    b = A.T @ Yv
    sol = np.linalg.solve(G, b)
    Wo = sol[:576].reshape(1, 64, 3, 3)
    bo = np.array([sol[576]])
    W['out.weight'] = Wo
    W['out.bias'] = bo

    # diagnostics of the fit on training data
    resid = A @ sol - Yv
    diag['mse_lastlayer_fit_train'] = float(np.mean(resid ** 2))
    diag['mse_before_train'] = float(np.mean((bpred - np.asarray(f['prediction'][bidx], dtype=np.float64)) ** 2))

    # ---------------- 3) per-timestep optimal affine copy of x_t (reference) ----------------
    # c_t = <y,x>/<x,x> per timestep (pixelwise-shared), d_t = mean(y) - c_t mean(x)
    c_t = np.zeros(32); d_t = np.zeros(32); mse_aff = np.zeros(32)
    for tt in range(32):
        m = (t == tt)
        if m.sum() < 2:
            continue
        xt = x[m].reshape(m.sum(), -1); yt = y[m].reshape(m.sum(), -1)
        cc = float(np.sum(xt * yt) / max(np.sum(xt * xt), 1e-12))
        dd = float(yt.mean() - cc * xt.mean())
        c_t[tt] = cc; d_t[tt] = dd
        mse_aff[tt] = float(np.mean((yt - cc * xt - dd) ** 2))
    diag['c_t_first5'] = [round(float(v), 4) for v in c_t[:5]]
    diag['c_t_last5'] = [round(float(v), 4) for v in c_t[-5:]]
    diag['mse_affine_copy_mean'] = float(np.mean(mse_aff))
    diag['mse_zero_baseline'] = float(np.mean(y ** 2))

    # ---------------- 4) build an explicit epsilon-copy path in conv1/conv2/conv3 ----------
    # conv1 ch0: +x_t (identity), ch1: -x_t (identity)  -> after ReLU: ReLU(x), ReLU(-x)
    # conv2 ch0: +conv1ch0, ch1: -conv1ch1 -> pre-act = ReLU(x) - ReLU(-x) = x
    # conv3 ch0: +conv2ch0 -> ReLU(x); ch1: -conv2ch0 -> ReLU(-x)
    # out: +ch0 - ch1 -> x_t exactly (then fit residual on top)
    def ident():
        I = np.zeros((3, 3)); I[1, 1] = 1.0; return I

    c1w = np.zeros((32, 2, 3, 3))
    c1b = np.zeros(32)
    c1w[0, 0] = ident()
    c1w[1, 0] = -ident()
    # remaining conv1 channels: keep a small copy of the original random filters so that
    # extra (nonlinear) features remain available for the residual fit
    c1w[2:] = W['conv1.weight'][2:]
    c1b[:] = W['conv1.bias'][:]
    c1b[0] = 0.0; c1b[1] = 0.0

    c2w = np.zeros((64, 32, 3, 3))
    c2b = np.zeros(64)
    c2w[0, 0] = ident()
    c2w[1, 1] = -ident()
    c2w[2:] = W['conv2.weight'][2:]
    c2b[:] = W['conv2.bias'][:]
    c2b[0] = 0.0; c2b[1] = 0.0

    c3w = np.zeros((64, 64, 3, 3))
    c3b = np.zeros(64)
    c3w[0, 0] = ident()
    c3w[1, 0] = -ident()
    c3w[2:] = W['conv3.weight'][2:]
    c3b[:] = W['conv3.bias'][:]
    c3b[0] = 0.0; c3b[1] = 0.0

    W2 = dict(W)
    W2['conv1.weight'] = c1w; W2['conv1.bias'] = c1b
    W2['conv2.weight'] = c2w; W2['conv2.bias'] = c2b
    W2['conv3.weight'] = c3w; W2['conv3.bias'] = c3b

    # recompute features with the new conv1..3 (cached features are stale now)
    Fnew = feats(x, t, W2)
    A2 = design_from_features(Fnew)
    G2 = A2.T @ A2
    lam2 = 1e-4 * float(np.mean(np.sum(A2 * A2, axis=1))) + 1e-8
    G2.flat[:: G2.shape[0] + 1] += lam2
    b2 = A2.T @ Yv
    sol2 = np.linalg.solve(G2, b2)
    W2['out.weight'] = sol2[:576].reshape(1, 64, 3, 3)
    W2['out.bias'] = np.array([sol2[576]])
    r2 = A2 @ sol2 - Yv
    diag['mse_copy_path_fit_train'] = float(np.mean(r2 ** 2))

    # choose the better of the two constructions (exact closed-form comparisons)
    if diag['mse_copy_path_fit_train'] <= diag['mse_lastlayer_fit_train']:
        Wfinal = W2
        diag['chosen'] = 'copy_path'
    else:
        Wfinal = W
        diag['chosen'] = 'lastlayer_only'

    # ---------------- 5) sanity check final forward on the traced batch ----------------
    fp = forward(bx, bt, Wfinal)
    diag['final_batch_pred_std'] = float(fp.std())
    diag['final_batch_pred_mean'] = float(fp.mean())
    # training MSE of the final model (all 2000 examples) -- direct forward, no stale features
    fp_all = forward(x, t, Wfinal)
    diag['mse_final_train'] = float(np.mean((fp_all - y) ** 2))
    diag['mse_before_train'] = float(np.mean((np.asarray(f['prediction'], dtype=np.float64) - y) ** 2))

    # ---------------- 6) export, sanitised ----------------
    out = {}
    for k in w.keys():
        v = np.asarray(Wfinal[k], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=1000.0, neginf=-1000.0)
        v = np.clip(v, -1000.0, 1000.0)
        out[k] = v.astype(np.float32)
    return {"weights": out, "diagnostics": diag}
