import numpy as np

def _conv_forward(inp, weight, bias, pad=1, relu=True):
    """PyTorch-style cross-correlation conv, stride 1."""
    N, C, H, W = inp.shape
    O = weight.shape[0]
    xp = np.pad(inp, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((N, O, H, W), dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            patch = xp[:, :, di:di + H, dj:dj + W]          # (N,C,H,W)
            wk = weight[:, :, di, dj]                        # (O,C)
            # (N,H,W,C) x (C,O) -> (N,H,W,O)
            out += np.einsum('nhwc,oc->nohw', patch.transpose(0, 2, 3, 1), wk,
                             optimize=True).astype(np.float32)
    out += bias.reshape(1, O, 1, 1)
    if relu:
        np.maximum(out, 0, out=out)
    return out


def _refit_readout(F, Y, lam=1e-4):
    """Closed-form ridge least squares for out.weight (1,64,3,3) + out.bias.
    F: (N,64,H,W) features; Y: (N,1,H,W) targets."""
    N, C, H, W = F.shape
    Fp = np.pad(F.astype(np.float32), ((0, 0), (0, 0), (1, 1), (1, 1)))
    shifts = []
    for di in range(3):
        for dj in range(3):
            sh = Fp[:, :, di:di + H, dj:dj + W].transpose(0, 2, 3, 1).reshape(-1, C)
            shifts.append(np.ascontiguousarray(sh))
    ones = np.ones((N * H * W, 1), dtype=np.float32)
    shifts.append(ones)
    P = 9 * C + 1
    M = np.zeros((P, P), dtype=np.float64)
    rhs = np.zeros(P, dtype=np.float64)
    yv = Y.transpose(0, 2, 3, 1).reshape(-1).astype(np.float64)
    for i in range(10):
        si = shifts[i].astype(np.float64)
        rhs[i * C:(i + 1) * C] = si.T @ yv if i < 9 else si.T @ yv
        for j in range(i, 10):
            sj = shifts[j].astype(np.float64)
            blk = si.T @ sj
            if i < 9 and j < 9:
                M[i * C:(i + 1) * C, j * C:(j + 1) * C] = blk
                if i != j:
                    M[j * C:(j + 1) * C, i * C:(i + 1) * C] = blk.T
            else:
                # bias row/col
                if i < 9:
                    M[i * C:(i + 1) * C, 9 * C] = blk.ravel()
                    M[9 * C, i * C:(i + 1) * C] = blk.ravel()
                else:
                    M[9 * C, 9 * C] = blk.ravel()[0]
    M[np.arange(P), np.arange(P)] += lam
    sol = np.linalg.solve(M, rhs).astype(np.float32)
    Wout = sol[:9 * C].reshape(3, 3, C).transpose(2, 0, 1).reshape(1, C, 3, 3)
    bout = sol[9 * C:9 * C + 1]
    return Wout, bout


def _mse(pred, y):
    return float(np.mean((pred.astype(np.float64) - y.astype(np.float64)) ** 2))


def edit(w, f):
    diag = {}
    x = f['x']; t = f['t']; y = f['y']
    feats = f['features'].astype(np.float32)   # (2000,64,16,16) out-layer input
    pred = f['prediction'].astype(np.float32)

    base_mse = _mse(pred, y)
    diag['base_mse'] = base_mse

    # ---- Variant A: refit readout on cached features ----
    WoutA, bA = _refit_readout(feats, y, lam=1e-4)
    # evaluate
    N, C, H, W = feats.shape
    Fp = np.pad(feats, ((0, 0), (0, 0), (1, 1), (1, 1)))
    predA = np.zeros((N, 1, H, W), dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            patch = Fp[:, :, di:di + H, dj:dj + W].transpose(0, 2, 3, 1)
            predA[:, 0] += np.einsum('nhwc,oc->nhw', patch, WoutA[:, :, di, dj], optimize=True)
    predA += bA.reshape(1, 1, 1)
    mseA = _mse(predA, y)
    diag['mse_readout_refit'] = mseA

    best = ('A', mseA, WoutA, bA, None)

    # ---- Variant B: exact-identity conv2/conv3, recompute features, refit ----
    try:
        c1w = w['conv1.weight'].astype(np.float32); c1b = w['conv1.bias'].astype(np.float32)
        # build input: concat x with t/31
        u = (t.astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
        u = np.broadcast_to(u, (x.shape[0], 1, x.shape[2], x.shape[3])).astype(np.float32)
        inp = np.concatenate([x.astype(np.float32), u], axis=1)   # (N,2,16,16)
        h1 = _conv_forward(inp, c1w, c1b, relu=True)              # (N,32,16,16)
        # identity conv2: 32->64, channel c<32 -> c, else 0
        c2w = np.zeros((64, 32, 3, 3), dtype=np.float32)
        for c in range(32):
            c2w[c, c, 1, 1] = 1.0
        c2b = np.zeros(64, dtype=np.float32)
        h2 = _conv_forward(h1, c2w, c2b, relu=True)               # (N,64,16,16)
        c3w = np.zeros((64, 64, 3, 3), dtype=np.float32)
        for c in range(64):
            c3w[c, c, 1, 1] = 1.0
        c3b = np.zeros(64, dtype=np.float32)
        h3 = _conv_forward(h2, c3w, c3b, relu=True)               # (N,64,16,16)
        WoutB, bB = _refit_readout(h3, y, lam=1e-4)
        predB = np.zeros((N, 1, H, W), dtype=np.float32)
        FpB = np.pad(h3, ((0, 0), (0, 0), (1, 1), (1, 1)))
        for di in range(3):
            for dj in range(3):
                patch = FpB[:, :, di:di + H, dj:dj + W].transpose(0, 2, 3, 1)
                predB[:, 0] += np.einsum('nhwc,oc->nhw', patch, WoutB[:, :, di, dj], optimize=True)
        predB += bB.reshape(1, 1, 1)
        mseB = _mse(predB, y)
        diag['mse_identity_refit'] = mseB
        if mseB < best[1]:
            best = ('B', mseB, WoutB, bB, (c2w, c2b, c3w, c3b))
    except Exception as e:
        diag['variantB_error'] = str(e)

    # ---- commit best ----
    tag, mse_best, Wout, bout, extra = best
    diag['chosen'] = tag
    diag['final_mse_estimate'] = mse_best

    w = dict(w)
    w['out.weight'] = np.asarray(Wout, dtype=np.float32)
    w['out.bias'] = np.asarray(bout, dtype=np.float32)
    if extra is not None:
        c2w, c2b, c3w, c3b = extra
        w['conv2.weight'] = c2w
        w['conv2.bias'] = c2b
        w['conv3.weight'] = c3w
        w['conv3.bias'] = c3b

    # sanitize
    for k, v in list(w.items()):
        v = np.asarray(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.clip(v, -1000.0, 1000.0)
        w[k] = v

    return {"weights": w, "diagnostics": diag}
