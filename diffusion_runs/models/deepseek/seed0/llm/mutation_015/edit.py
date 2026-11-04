import numpy as np


def _out_forward(F, W, b):
    # F: (N,64,16,16) float32 ; W: (1,64,3,3) ; b: (1,)
    N, C, H, Wd = F.shape
    P = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float32)
    P[:, :, 1:-1, 1:-1] = F
    out = np.zeros((N, 1, H, Wd), dtype=np.float32)
    Wf = W.reshape(64, 3, 3)
    for ki in range(3):
        for kj in range(3):
            out[:, 0] += np.tensordot(P[:, :, ki:ki + H, kj:kj + Wd], Wf[:, ki, kj], axes=([1], [0]))
    out += b.reshape(1, 1, 1)
    return out


def edit(w, f):
    diag = {}
    F = np.asarray(f['features'], dtype=np.float32)      # (2000,64,16,16)
    Y = np.asarray(f['y'], dtype=np.float32)             # (2000,1,16,16)
    ids = np.asarray(f['batch.ids']).astype(np.int64)
    B_in = np.asarray(f['layer.out.input0'], dtype=np.float32)
    B_y = np.asarray(f['batch.y'], dtype=np.float32)

    W4 = np.array(w['out.weight'], dtype=np.float32).copy()   # (1,64,3,3)
    b4 = np.array(w['out.bias'], dtype=np.float32).copy()     # (1,)

    # ---- sanity: reproduce current MSE on the full set with our own forward ----
    cur = _out_forward(F, W4, b4)
    mse_cur_full = float(np.mean((cur - Y) ** 2))
    diag['mse_repro_full'] = mse_cur_full

    # ---- check feature/trace consistency on the paired batch ----
    if ids.size and B_in.shape[0] == ids.size:
        sub = F[ids]
        align_err = float(np.max(np.abs(sub - B_in)))
    else:
        align_err = np.inf
    diag['feature_trace_maxabs'] = align_err

    use_full = (align_err < 1e-3) and (mse_cur_full < 0.5)

    if use_full:
        Ffit, Yfit = F, Y
        diag['fit_source'] = 'full2000'
    else:
        Ffit, Yfit = B_in, B_y
        diag['fit_source'] = 'batch128'

    N, C, H, Wd = Ffit.shape
    ncol = C * 9 + 1

    # ---- accumulate normal equations in chunks (memory safe) ----
    XtX = np.zeros((ncol, ncol), dtype=np.float64)
    Xty = np.zeros(ncol, dtype=np.float64)

    pad = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float32)
    pad[:, :, 1:-1, 1:-1] = Ffit

    chunk = 200
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        p = pad[s:e]                                   # (n,C,18,18)
        win = np.lib.stride_tricks.sliding_window_view(p, (3, 3), axis=(2, 3))
        # win: (n,C,16,16,3,3) -> (n,16,16,C,3,3)
        win = np.transpose(win, (0, 2, 3, 1, 4, 5))
        X = win.reshape(-1, C * 9).astype(np.float64)  # (n*256, 576)
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
        t = Yfit[s:e, 0].reshape(-1).astype(np.float64)  # (n*256,)
        XtX += X.T @ X
        Xty += X.T @ t
        del X, win, p

    # tiny ridge for numerical stability (negligible shrinkage)
    lam = 1e-8 * max(1.0, float(np.trace(XtX)) / ncol)
    XtX[np.diag_indices(ncol)] += lam

    theta = np.linalg.solve(XtX, Xty)
    Wnew = theta[:C * 9].reshape(1, C, 3, 3).astype(np.float32)
    bnew = np.array([theta[C * 9]], dtype=np.float32)

    # ---- validate: accept only if full-set MSE improves ----
    new_pred = _out_forward(F, Wnew, bnew)
    mse_new_full = float(np.mean((new_pred - Y) ** 2))
    diag['mse_new_full'] = mse_new_full

    if not np.isfinite(mse_new_full) or mse_new_full >= mse_cur_full:
        diag['accepted'] = False
        return {"weights": w, "diagnostics": diag}

    diag['accepted'] = True
    diag['mse_before'] = mse_cur_full
    diag['mse_after'] = mse_new_full
    diag['out_weight_norm'] = float(np.linalg.norm(Wnew))
    diag['out_bias'] = float(bnew[0])

    w = {k: np.array(v, dtype=np.float32).copy() for k, v in w.items()}
    w['out.weight'] = Wnew
    w['out.bias'] = bnew

    # keep everything finite and bounded
    for k in w:
        a = np.asarray(w[k], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(a, -1000.0, 1000.0, out=a)
        w[k] = a
    return {"weights": w, "diagnostics": diag}
