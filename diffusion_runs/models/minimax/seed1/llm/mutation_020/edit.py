import numpy as np

def edit(w, f):
    rng = np.random.default_rng(0)

    def conv2d(x, W, b):
        # x: (N,Cin,H,W), W: (Cout,Cin,3,3) cross-correlation, pad 1
        N, Cin, H, Wd = x.shape
        Cout = W.shape[0]
        xp = np.zeros((N, Cin, H + 2, Wd + 2), dtype=np.float64)
        xp[:, :, 1:-1, 1:-1] = x
        out = np.zeros((N, Cout, H, Wd), dtype=np.float64)
        for ky in range(3):
            for kx in range(3):
                out += np.einsum('nihw,oi->nohw', xp[:, :, ky:ky + H, kx:kx + Wd], W[:, :, ky, kx])
        out += b[None, :, None, None]
        return out

    # ---- weights (float64 copies) ----
    W1 = np.asarray(w['conv1.weight'], dtype=np.float64)
    b1 = np.asarray(w['conv1.bias'], dtype=np.float64)
    W2 = np.asarray(w['conv2.weight'], dtype=np.float64)
    b2 = np.asarray(w['conv2.bias'], dtype=np.float64)
    W3 = np.asarray(w['conv3.weight'], dtype=np.float64)
    b3 = np.asarray(w['conv3.bias'], dtype=np.float64)
    Wo = np.asarray(w['out.weight'], dtype=np.float64)
    bo = np.asarray(w['out.bias'], dtype=np.float64)

    # ---- forward pass (validated against the provided layer traces) ----
    def forward(x, t, W1, b1, W2, b2, W3, b3, Wo, bo):
        v = (np.asarray(t, dtype=np.float64) / 31.0)[:, None, None, None]
        inp = np.concatenate([np.asarray(x, dtype=np.float64), np.broadcast_to(v, (x.shape[0], 1, x.shape[2], x.shape[3]))], axis=1)
        h1 = np.maximum(conv2d(inp, W1, b1), 0.0)
        h2 = np.maximum(conv2d(h1, W2, b2), 0.0)
        h3 = np.maximum(conv2d(h2, W3, b3), 0.0)
        o = conv2d(h3, Wo, bo)
        return inp, h1, h2, h3, o

    bx = np.asarray(f['batch.x'], dtype=np.float64)
    bt = np.asarray(f['batch.t'], dtype=np.int64)
    _, _, _, h3b, ob = forward(bx, bt, W1, b1, W2, b2, W3, b3, Wo, bo)

    tr_out_in = np.asarray(f['layer.out.input0'], dtype=np.float64)
    tr_out_out = np.asarray(f['layer.out.output'], dtype=np.float64)
    err_in = float(np.max(np.abs(h3b - tr_out_in)))
    err_out = float(np.max(np.abs(ob - tr_out_out)))

    # ---- check whether cached `features` are the out-layer input ----
    feats = np.asarray(f['features'], dtype=np.float64)
    ids = np.asarray(f['batch.ids'], dtype=np.int64)
    feat_b = feats[ids]
    feat_matches_outin = float(np.max(np.abs(feat_b - tr_out_in)))
    feat_matches_conv3 = float(np.max(np.abs(feat_b - np.asarray(f['layer.conv3.output'], dtype=np.float64))))

    # ---- recompute hidden map for ALL training examples (authoritative) ----
    x_all = np.asarray(f['x'], dtype=np.float64)
    t_all = np.asarray(f['t'], dtype=np.int64)
    y_all = np.asarray(f['y'], dtype=np.float64)

    N = x_all.shape[0]
    Hh = x_all.shape[2]
    Wd = x_all.shape[3]
    P = Hh * Wd

    def build_design(hmap):
        # (N,64,H,W) -> (N*P, 576) patch design + bias column
        hmap = np.ascontiguousarray(hmap.transpose(0, 2, 3, 1))  # N,H,W,64
        hp = np.zeros((N, Hh + 2, Wd + 2, 64), dtype=np.float64)
        hp[:, 1:-1, 1:-1, :] = hmap
        cols = []
        for ky in range(3):
            for kx in range(3):
                cols.append(hp[:, ky:ky + Hh, kx:kx + Wd, :].reshape(N * P, 64))
        A = np.concatenate(cols, axis=1)
        return A

    # hidden map for all training examples, computed with the (unchanged) parent weights
    h_all = np.empty((N, 64, Hh, Wd), dtype=np.float64)
    CH = 250
    for s in range(0, N, CH):
        e = min(N, s + CH)
        vv = (t_all[s:e].astype(np.float64) / 31.0)[:, None, None, None]
        inp = np.concatenate([x_all[s:e], np.broadcast_to(vv, (e - s, 1, Hh, Wd))], axis=1)
        h1 = np.maximum(conv2d(inp, W1, b1), 0.0)
        h2 = np.maximum(conv2d(h1, W2, b2), 0.0)
        h3 = np.maximum(conv2d(h2, W3, b3), 0.0)
        h_all[s:e] = h3
        del inp, h1, h2, h3

    A = build_design(h_all)
    Y = y_all.transpose(0, 2, 3, 1).reshape(N * P, 1)
    del h_all

    # ---- closed-form ridge fit of the last layer (out.weight, out.bias) ----
    # center both sides (bias absorbs the means)
    Am = A.mean(axis=0)
    Ym = float(Y.mean())
    Ac = A - Am
    Yc = Y - Ym

    G = Ac.T @ Ac
    B = Ac.T @ Yc
    tr = float(np.trace(G)) / G.shape[0]

    def fit(lam):
        Gl = G + lam * tr * np.eye(G.shape[0])
        try:
            theta = np.linalg.solve(Gl, B)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(Gl, B, rcond=None)[0]
        return theta

    # 5-fold CV over examples to pick the ridge parameter
    folds = 5
    fold_id = np.arange(N) % folds
    best_lam, best_cv = None, np.inf
    for lam in [1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]:
        err = 0.0
        cnt = 0
        for k in range(folds):
            tr_idx = np.where(fold_id != k)[0]
            va_idx = np.where(fold_id == k)[0]
            tr_rows = (tr_idx[:, None] * P + np.arange(P)[None, :]).reshape(-1)
            va_rows = (va_idx[:, None] * P + np.arange(P)[None, :]).reshape(-1)
            Atr = A[tr_rows]
            Ytr = Y[tr_rows]
            mu_a = Atr.mean(axis=0)
            mu_y = float(Ytr.mean())
            Acv = Atr - mu_a
            Ycv = Ytr - mu_y
            Gk = Acv.T @ Acv
            trk = float(np.trace(Gk)) / Gk.shape[0]
            Gl = Gk + lam * trk * np.eye(Gk.shape[0])
            try:
                th = np.linalg.solve(Gl, Acv.T @ Ycv)
            except np.linalg.LinAlgError:
                th = np.linalg.lstsq(Gl, Acv.T @ Ycv, rcond=None)[0]
            Ava = A[va_rows] - mu_a
            Yva = Y[va_rows] - mu_y
            pred = Ava @ th
            err += float(np.sum((pred - Yva) ** 2))
            cnt += Ava.shape[0]
        cv = err / cnt
        if cv < best_cv:
            best_cv = cv
            best_lam = lam

    theta = fit(best_lam)
    new_Wo = theta.reshape(3, 3, 64).transpose(2, 0, 1)  # (64,3,3)
    new_bo = np.array([Ym - float(Am @ theta)], dtype=np.float64)  # (1,)

    # ---- verify with an independent forward pass on the batch ----
    _, _, _, h3n, on = forward(bx, bt, W1, b1, W2, b2, W3, b3, new_Wo, new_bo)
    yb = np.asarray(f['batch.y'], dtype=np.float64)
    mse_batch_new = float(np.mean((on - yb) ** 2))
    mse_batch_old = float(np.mean((ob - yb) ** 2))

    # in-sample MSE over all training examples with the new readout
    pred_all = (A - Am) @ theta + Ym
    mse_all_new = float(np.mean((pred_all - Y) ** 2))
    del A, Y, Ac, Yc, G, B, pred_all

    # ---- write back (float32, finite, |.| <= 1000) ----
    out = {}
    for k, v in w.items():
        out[k] = np.array(v, dtype=np.float32, copy=True)
    out['out.weight'] = np.clip(new_Wo, -1000.0, 1000.0).astype(np.float32)
    out['out.bias'] = np.clip(new_bo, -1000.0, 1000.0).astype(np.float32)

    for k in out:
        arr = out[k]
        if not np.all(np.isfinite(arr)):
            out[k] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        out[k] = np.clip(out[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        'fwd_max_err_out_input': round(err_in, 8),
        'fwd_max_err_out_output': round(err_out, 8),
        'features_match_out_input_maxerr': round(feat_matches_outin, 8),
        'features_match_conv3_out_maxerr': round(feat_matches_conv3, 8),
        'ridge_lambda': float(best_lam),
        'cv_mse': round(float(best_cv), 6),
        'insample_mse_all_new': round(mse_all_new, 6),
        'batch_mse_old': round(mse_batch_old, 6),
        'batch_mse_new': round(mse_batch_new, 6),
        'out_w_absmean': round(float(np.mean(np.abs(new_Wo))), 6),
        'out_bias': round(float(new_bo[0]), 6),
    }
    return {'weights': out, 'diagnostics': diagnostics}
