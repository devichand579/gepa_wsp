import numpy as np
import math

def _im2col(x, k=3, pad=1):
    # x: (N,C,H,W) -> (N, C*k*k, H*W)
    N, C, H, W = x.shape
    xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
    cols = np.empty((N, C*k*k, H*W), dtype=np.float32)
    idx = 0
    for i in range(k):
        for j in range(k):
            cols[:, idx*C:(idx+1)*C, :] = xp[:, :, i:i+H, j:j+W].reshape(N, C, H*W)
            idx += 1
    return cols

def _conv_forward(x, W, b):
    # x: (N,Cin,H,W), W: (Cout,Cin,k,k), b: (Cout,)
    N, Cin, H, Wd = x.shape
    Cout, _, k, _ = W.shape
    cols = _im2col(x, k, pad=1)  # (N, Cin*k*k, H*W)
    Wm = W.reshape(Cout, -1)     # (Cout, Cin*k*k)
    out = np.einsum('ok,nkp->nop', Wm, cols, optimize=True)
    out = out.reshape(N, Cout, H, Wd) + b.reshape(1, Cout, 1, 1)
    return out

def _relu(x):
    return np.maximum(x, 0.0)

def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    diag = {}

    x = np.asarray(f['x'], dtype=np.float32)          # (2000,1,16,16)
    t = np.asarray(f['t']).astype(np.float32)          # (2000,)
    y = np.asarray(f['y'], dtype=np.float32)           # (2000,1,16,16)
    feats = np.asarray(f['features'], dtype=np.float32)  # (2000,64,16,16)

    N = x.shape[0]
    # Build input: concat x with t/31 broadcast
    tch = (t / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x, np.broadcast_to(tch, (N,1,16,16))], axis=1)  # (N,2,16,16)

    # ---- 1) Exact closed-form refit of the output layer on cached features ----
    # out: (1,64,3,3), bias (1,).  yhat = conv_out(feats) + b_out
    # Solve least squares:  min || X_out @ theta - y ||^2
    # where theta = [Wout flattened (576), b_out (1)]
    # Use normal equations with accumulation to stay memory-safe.
    Cout, Cin, k, _ = w['out.weight'].shape
    P = Cout * Cin * k * k + Cout  # 577
    XtX = np.zeros((P, P), dtype=np.float64)
    Xty = np.zeros(P, dtype=np.float64)

    # process in chunks of samples
    chunk = 64
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        fb = feats[s:e]  # (n,64,16,16)
        n = fb.shape[0]
        cols = _im2col(fb, k=3, pad=1)  # (n, 64*9, 256)
        # design: for each output pixel, features are cols[:, :, p] (576) plus bias 1
        # X: (n*256, 577)
        Xc = np.concatenate([cols.transpose(0,2,1).reshape(n*256, Cin*k*k),
                             np.ones((n*256,1), dtype=np.float32)], axis=1).astype(np.float64)
        yc = y[s:e].reshape(n*256).astype(np.float64)
        XtX += Xc.T @ Xc
        Xty += Xc.T @ yc

    # tiny ridge for numerical stability
    ridge = 1e-8 * np.trace(XtX) / P
    XtX_reg = XtX + ridge * np.eye(P)
    try:
        theta = np.linalg.solve(XtX_reg, Xty)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]

    Wout = theta[:Cout*Cin*k*k].reshape(Cout, Cin, k, k).astype(np.float32)
    bout = theta[Cout*Cin*k*k:].astype(np.float32)
    w['out.weight'] = Wout
    w['out.bias'] = bout

    # ---- 2) Closed-form refit of conv3 bias with fixed ReLU mask ----
    # Recompute a2 = relu(conv2(relu(conv1(inp)))) for all samples (needed for conv3)
    a1 = _relu(_conv_forward(inp, w['conv1.weight'], w['conv1.bias']))
    a2 = _relu(_conv_forward(a1, w['conv2.weight'], w['conv2.bias']))
    z3 = _conv_forward(a2, w['conv3.weight'], w['conv3.bias'])
    M = (z3 > 0).astype(np.float32)  # fixed ReLU mask
    h3 = M * z3

    # yhat = conv_out(M * (conv3(a2) + b3)) + b_out
    # With fixed mask and fixed conv3 weights, yhat is affine in b3:
    #   yhat = const + conv_out(M * b3_broadcast)
    # Solve least squares for b3 (64 params).
    # const = conv_out(M * conv3(a2)) + b_out
    z3_nb = _conv_forward(a2, w['conv3.weight'], np.zeros_like(w['conv3.bias']))
    const = _conv_forward(M * z3_nb, w['out.weight'], w['out.bias'])  # (N,1,16,16)

    # Build design for b3: for each sample, yhat_pixel = const + sum_c Wout_eff[c, pixel] * b3[c]
    # where Wout_eff[c, pixel] = sum_{di,dj} Wout[0,c,di,dj] * M[n,c, i+di-1, j+dj-1]
    # This is a per-sample (256 x 64) matrix. Accumulate normal equations.
    P3 = 64
    XtX3 = np.zeros((P3, P3), dtype=np.float64)
    Xty3 = np.zeros(P3, dtype=np.float64)
    Wout0 = w['out.weight'][0]  # (64,3,3)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Mb = M[s:e]  # (n,64,16,16)
        n = Mb.shape[0]
        # For each output pixel (i,j), gather M[n,c,i+di-1,j+dj-1] for di,dj in 0..2
        Mp = np.pad(Mb, ((0,0),(0,0),(1,1),(1,1)))
        # design matrix D: (n, 256, 64)
        D = np.zeros((n, 256, 64), dtype=np.float32)
        for di in range(3):
            for dj in range(3):
                # M[n, c, i+di-1, j+dj-1] -> slice
                sl = Mp[:, :, di:di+16, dj:dj+16]  # (n,64,16,16)
                # weight Wout0[c,di,dj]
                D += sl.transpose(0,2,3,1).reshape(n, 256, 64) * Wout0[:, di, dj].reshape(1,1,64)
        D = D.astype(np.float64)
        rb = (y[s:e].reshape(n,256).astype(np.float64) - const[s:e].reshape(n,256).astype(np.float64))
        XtX3 += np.einsum('npc,npd->cd', D, D, optimize=True)
        Xty3 += np.einsum('npc,np->c', D, rb, optimize=True)

    ridge3 = 1e-8 * np.trace(XtX3) / P3
    XtX3_reg = XtX3 + ridge3 * np.eye(P3)
    try:
        b3 = np.linalg.solve(XtX3_reg, Xty3)
    except np.linalg.LinAlgError:
        b3 = np.linalg.lstsq(XtX3_reg, Xty3, rcond=None)[0]
    w['conv3.bias'] = b3.astype(np.float32)

    # ---- diagnostics ----
    # recompute prediction with updated weights
    a1b = _relu(_conv_forward(inp, w['conv1.weight'], w['conv1.bias']))
    a2b = _relu(_conv_forward(a1b, w['conv2.weight'], w['conv2.bias']))
    z3b = _conv_forward(a2b, w['conv3.weight'], w['conv3.bias'])
    h3b = _relu(z3b)
    pred = _conv_forward(h3b, w['out.weight'], w['out.bias'])
    mse_after = float(np.mean((pred - y) ** 2))
    diag['mse_after_estimate'] = mse_after
    diag['out_weight_norm'] = float(np.linalg.norm(w['out.weight']))
    diag['out_bias'] = float(w['out.bias'][0])
    diag['conv3_bias_norm'] = float(np.linalg.norm(w['conv3.bias']))
    diag['ridge_out'] = float(ridge)
    diag['ridge_conv3b'] = float(ridge3)

    # clip to finite float32 within bounds
    for kk in w:
        w[kk] = np.nan_to_num(w[kk], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        w[kk] = np.clip(w[kk], -1000.0, 1000.0)

    return {"weights": w, "diagnostics": diag}
