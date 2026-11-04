import numpy as np
import math

def _conv2d(x, w, b, pad=1):
    # x: [N,C,H,W], w: [O,C,3,3], b: [O]
    N, C, H, W = x.shape
    O = w.shape[0]
    xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3,3), axis=(2,3))  # [N,C,H,W,3,3]
    cols = cols.transpose(0,2,3,1,4,5).reshape(N*H*W, C*9)
    Wm = w.reshape(O, C*9).T  # [C*9, O]
    out = cols @ Wm + b[None,:]
    out = out.reshape(N,H,W,O).transpose(0,3,1,2)
    return out

def _relu(x):
    return np.maximum(x, 0.0)

def _forward(w, x, t):
    N = x.shape[0]
    tc = (t.astype(np.float32) / 31.0).reshape(N,1,1,1)
    inp = np.concatenate([x, np.broadcast_to(tc, (N,1,x.shape[2],x.shape[3]))], axis=1)
    a1 = _relu(_conv2d(inp, w['conv1.weight'], w['conv1.bias']))
    a2 = _relu(_conv2d(a1, w['conv2.weight'], w['conv2.bias']))
    a3 = _relu(_conv2d(a2, w['conv3.weight'], w['conv3.bias']))
    out = _conv2d(a3, w['out.weight'], w['out.bias'])
    return inp, a1, a2, a3, out

def _fit_out(feats, y, lam):
    # feats: [N,64,16,16], y: [N,1,16,16]
    N, C, H, W = feats.shape
    # build design matrix of local 3x3 patches: [N*H*W, C*9]
    xp = np.pad(feats, ((0,0),(0,0),(1,1),(1,1)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3,3), axis=(2,3))  # [N,C,H,W,3,3]
    cols = cols.transpose(0,2,3,1,4,5).reshape(N*H*W, C*9)
    Y = y.reshape(N*H*W)
    # add bias column
    X = np.concatenate([cols, np.ones((cols.shape[0],1), dtype=np.float32)], axis=1)
    # normal equations with ridge (don't penalize bias)
    XtX = X.T @ X
    XtY = X.T @ Y
    d = XtX.shape[0]
    reg = np.eye(d, dtype=np.float64) * lam
    reg[-1,-1] = 0.0
    try:
        sol = np.linalg.solve(XtX.astype(np.float64) + reg, XtY.astype(np.float64))
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(XtX.astype(np.float64) + reg, XtY.astype(np.float64), rcond=None)[0]
    W = sol[:C*9].reshape(1, C, 3, 3).astype(np.float32)
    b = np.array([sol[-1]], dtype=np.float32)
    return W, b

def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}

    x = np.asarray(f['x'], dtype=np.float32)          # [2000,1,16,16]
    t = np.asarray(f['t']).astype(np.int64)           # [2000]
    y = np.asarray(f['y'], dtype=np.float32)          # [2000,1,16,16]
    feats = np.asarray(f['features'], dtype=np.float32)  # [2000,64,16,16] final hidden

    # ---- 1. Refit out layer on the full 2000-example feature bank ----
    # Try a small set of ridge strengths and pick the one with best
    # (in-sample, but heavily over-determined) MSE. 2000*256 = 512k rows
    # vs 577 params, so overfitting is negligible.
    best = None
    for lam in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        W, b = _fit_out(feats, y, lam)
        # evaluate
        xp = np.pad(feats, ((0,0),(0,0),(1,1),(1,1)))
        cols = np.lib.stride_tricks.sliding_window_view(xp, (3,3), axis=(2,3))
        cols = cols.transpose(0,2,3,1,4,5).reshape(-1, 64*9)
        pred = cols @ W.reshape(-1) + b[0]
        mse = float(np.mean((pred - y.reshape(-1))**2))
        if best is None or mse < best[0]:
            best = (mse, W, b, lam)
    _, W_out, b_out, lam_used = best

    # Guard against pathological blow-up
    W_out = np.clip(W_out, -50.0, 50.0).astype(np.float32)
    b_out = np.clip(b_out, -50.0, 50.0).astype(np.float32)

    w['out.weight'] = W_out
    w['out.bias'] = b_out

    # ---- 2. Recompute features with the new out layer (out doesn't affect
    #         upstream features, so this is a no-op for the head; we only
    #         recompute to report honest diagnostics). ----
    # ---- 3. Mild t-channel rescale in conv1 to improve conditioning. ----
    # The t channel currently enters as t/31 in [0,1].  We rescale the
    # conv1 weights on that channel by a factor that keeps the pre-activation
    # variance roughly matched to the x_t channel contribution.
    c1w = w['conv1.weight']  # [32,2,3,3]
    x_std = float(np.std(x)) + 1e-6
    t_std = float(np.std(t.astype(np.float32) / 31.0)) + 1e-6
    # current contribution ratio
    xw = c1w[:, 0:1, :, :]
    tw = c1w[:, 1:2, :, :]
    x_contrib = float(np.sqrt(np.mean(xw**2))) * x_std
    t_contrib = float(np.sqrt(np.mean(tw**2))) * t_std
    if t_contrib > 1e-8:
        scale = float(np.clip(x_contrib / t_contrib, 0.25, 4.0))
    else:
        scale = 1.0
    # Only apply a gentle correction (blend) to avoid destroying the parent.
    scale = 1.0 + 0.5 * (scale - 1.0)
    c1w = c1w.copy()
    c1w[:, 1:2, :, :] = c1w[:, 1:2, :, :] * scale
    w['conv1.weight'] = c1w.astype(np.float32)

    # ---- 4. Recompute the full forward pass on the 2000 training examples
    #         with the updated conv1, then refit the out layer on the
    #         *fresh* features so the head is consistent with the new
    #         upstream weights. ----
    # Process in chunks to respect memory limits.
    N = x.shape[0]
    chunk = 250
    fresh = np.empty((N, 64, 16, 16), dtype=np.float32)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        _, _, _, a3, _ = _forward(w, x[i:j], t[i:j])
        fresh[i:j] = a3
    W2, b2 = _fit_out(fresh, y, lam_used)
    W2 = np.clip(W2, -50.0, 50.0).astype(np.float32)
    b2 = np.clip(b2, -50.0, 50.0).astype(np.float32)
    w['out.weight'] = W2
    w['out.bias'] = b2

    # ---- 5. Final safety: ensure finite float32 and |.|<=1000 ----
    for k in list(w.keys()):
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[k] = arr.astype(np.float32)

    # Diagnostics
    diag = {
        "out_weight_max": float(np.max(np.abs(w['out.weight']))),
        "out_weight_mean": float(np.mean(w['out.weight'])),
        "out_bias": float(w['out.bias'][0]),
        "ridge": float(lam_used),
        "n_samples": int(N),
        "t_channel_scale": float(scale),
    }
    return {"weights": w, "diagnostics": diag}
