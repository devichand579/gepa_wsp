import numpy as np

def _conv_forward(x, weight, bias, pad=1):
    # x: (N, Cin, H, W); weight: (Cout, Cin, 3, 3); returns (N, Cout, H, W)
    N, Cin, H, W = x.shape
    Cout = weight.shape[0]
    xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    out = np.zeros((N, Cout, H, W), dtype=np.float32)
    for a in range(3):
        for b in range(3):
            patch = xp[:, :, a:a+H, b:b+W]  # (N, Cin, H, W)
            w = weight[:, :, a, b]          # (Cout, Cin)
            out += np.einsum('oc,nchw->nohw', w, patch, optimize=True)
    out += bias.reshape(1, Cout, 1, 1)
    return out

def _relu(z):
    return np.maximum(z, 0.0)

def _im2col_features(feat):
    # feat: (N, C, H, W) -> design (N*H*W, C*9) with 3x3 zero-padded neighborhood
    N, C, H, W = feat.shape
    fp = np.pad(feat, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    cols = []
    for a in range(3):
        for b in range(3):
            cols.append(fp[:, :, a:a+H, b:b+W])
    # cols: list of (N,C,H,W) -> stack to (N,C,9,H,W) then transpose
    X = np.stack(cols, axis=2)          # (N, C, 9, H, W)
    X = X.transpose(0, 3, 4, 1, 2)      # (N, H, W, C, 9)
    X = X.reshape(N*H*W, C*9)
    return X

def _fit_readout(features, y, ridge=1e-3):
    # features: (N, C, H, W); y: (N, 1, H, W)
    N, C, H, W = features.shape
    X = _im2col_features(features)                 # (M, C*9)
    Y = y.transpose(0, 2, 3, 1).reshape(N*H*W, 1)  # (M, 1)
    M = X.shape[0]
    XtX = X.T @ X
    XtY = X.T @ Y
    reg = ridge * np.trace(XtX) / XtX.shape[0]
    XtX = XtX + reg * np.eye(XtX.shape[0], dtype=np.float64)
    sol = np.linalg.solve(XtX.astype(np.float64), XtY.astype(np.float64))  # (C*9, 1)
    # reshape to (1, C, 3, 3)
    w_out = sol.reshape(C, 3, 3).astype(np.float32)[None, ...]
    return w_out

def _predict(features, w_out, b_out):
    N, C, H, W = features.shape
    return _conv_forward(features, w_out, b_out)

def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    diag = {}

    features = np.asarray(f['features'], dtype=np.float32)   # (2000,64,16,16)
    y = np.asarray(f['y'], dtype=np.float32)                 # (2000,1,16,16)

    # ---- Baseline: refit readout on given features ----
    w_out = _fit_readout(features, y, ridge=1e-3)
    b_out = np.array([0.0], dtype=np.float32)

    # compute train MSE of refit readout (no bias)
    pred = _predict(features, w_out, b_out)
    mse_base = float(np.mean((pred - y) ** 2))
    diag['mse_readout_refit'] = mse_base

    # ---- Enriched features: make conv2/conv3 non-identity ----
    # Build conv2 (32->64) and conv3 (64->64) as structured spatial-diffusion
    # + channel-mixing operators that expand receptive field and add ReLU capacity.
    rng = np.random.RandomState(1234)

    # conv2: 32 -> 64. Each output channel = a 3x3 spatial smoothing of a
    # (possibly mixed) input channel. Use a small set of spatial kernels.
    Cin2, Cout2 = 32, 64
    w2 = np.zeros((Cout2, Cin2, 3, 3), dtype=np.float32)
    # spatial kernels: identity, box blur, and horizontal/vertical gradients
    k_id = np.zeros((3,3), np.float32); k_id[1,1] = 1.0
    k_box = np.ones((3,3), np.float32) / 9.0
    k_h = np.array([[0,0,0],[-1,0,1],[0,0,0]], np.float32)
    k_v = np.array([[0,-1,0],[0,0,0],[0,1,0]], np.float32)
    kernels = [k_id, k_box, k_h, k_v]
    for o in range(Cout2):
        c = o % Cin2
        k = kernels[(o // Cin2) % len(kernels)]
        w2[o, c] = k
    b2 = np.zeros(Cout2, dtype=np.float32)

    # conv3: 64 -> 64. Mix channels lightly + spatial identity to preserve info.
    Cin3, Cout3 = 64, 64
    w3 = np.zeros((Cout3, Cin3, 3, 3), dtype=np.float32)
    for o in range(Cout3):
        # keep a strong identity-ish center tap on same channel
        w3[o, o, 1, 1] = 1.0
        # add a small cross-channel mixing tap
        c2 = (o + 7) % Cin3
        w3[o, c2, 1, 1] = 0.15
    b3 = np.zeros(Cout3, dtype=np.float32)

    # Recompute features through conv1 -> ReLU -> conv2 -> ReLU -> conv3 -> ReLU
    x_in = np.asarray(f['layer.conv1.input0'], dtype=np.float32)  # (128,2,16,16) batch only
    # We only have conv1.input0 for the 128-batch; for full 2000 we must rebuild
    # concat(x, t/31) from x and t.
    x_all = np.asarray(f['x'], dtype=np.float32)      # (2000,1,16,16)
    t_all = np.asarray(f['t'], dtype=np.float32)      # (2000,)
    tch = (t_all / 31.0).reshape(-1, 1, 1, 1).astype(np.float32)
    tch = np.broadcast_to(tch, (x_all.shape[0], 1, 16, 16)).copy()
    inp_all = np.concatenate([x_all, tch], axis=1)    # (2000,2,16,16)

    h1 = _relu(_conv_forward(inp_all, w['conv1.weight'], w['conv1.bias']))
    h2 = _relu(_conv_forward(h1, w2, b2))
    h3 = _relu(_conv_forward(h2, w3, b3))

    # Refit readout on enriched features
    w_out2 = _fit_readout(h3, y, ridge=1e-3)
    pred2 = _predict(h3, w_out2, np.array([0.0], np.float32))
    mse_enriched = float(np.mean((pred2 - y) ** 2))
    diag['mse_enriched_refit'] = mse_enriched

    # ---- Decide: use enriched if clearly better, else baseline ----
    if mse_enriched < mse_base * 0.999:
        w['conv2.weight'] = w2
        w['conv2.bias'] = b2
        w['conv3.weight'] = w3
        w['conv3.bias'] = b3
        w['out.weight'] = w_out2
        w['out.bias'] = np.array([0.0], np.float32)
        diag['mode'] = 'enriched'
        diag['final_mse_estimate'] = mse_enriched
    else:
        w['out.weight'] = w_out
        w['out.bias'] = np.array([0.0], np.float32)
        diag['mode'] = 'readout_only'
        diag['final_mse_estimate'] = mse_base

    # sanitize
    for k in w:
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[k] = arr

    return {"weights": w, "diagnostics": diag}
