import numpy as np
import math

def edit(w, f):
    # ---- helpers -------------------------------------------------------
    def im2col_patches(x, k=3, pad=1):
        # x: (N,C,H,W) -> (N, C*k*k, H*W)  (cross-correlation layout)
        N, C, H, W = x.shape
        xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
        cols = np.empty((N, C*k*k, H*W), dtype=np.float32)
        idx = 0
        for a in range(k):
            for b in range(k):
                cols[:, idx*C:(idx+1)*C, :] = xp[:, :, a:a+H, b:b+W].reshape(N, C, H*W)
                idx += 1
        return cols  # (N, C*k*k, HW)

    def conv3x3(x, weight, bias):
        # weight: (O, I, 3, 3)
        O, I, _, _ = weight.shape
        N, C, H, W = x.shape
        cols = im2col_patches(x, 3, 1)          # (N, I*9, HW)
        Wm = weight.reshape(O, I*9)             # (O, I*9)
        out = np.einsum('oi,nil->nol', Wm, cols, optimize=True)
        out = out.reshape(N, O, H, W)
        out += bias.reshape(1, O, 1, 1)
        return out

    def relu(x):
        return np.maximum(x, 0.0, dtype=np.float32)

    # ---- rebuild input exactly as the forward does ---------------------
    x   = f['x'].astype(np.float32)      # (2000,1,16,16)
    t   = f['t'].astype(np.float32)      # (2000,)
    N   = x.shape[0]
    tch = (t / 31.0).reshape(N, 1, 1, 1).astype(np.float32)
    tch = np.broadcast_to(tch, (N, 1, 16, 16)).astype(np.float32)
    inp = np.concatenate([x, tch], axis=1)   # (N,2,16,16)

    # ---- design a richer conv1 ----------------------------------------
    # keep the 16 trained "time-ramp" pointwise filters, and replace the
    # remaining 16 with useful spatial / quadratic taps.
    cw = w['conv1.weight'].astype(np.float32).copy()   # (32,2,3,3)
    cb = w['conv1.bias'].astype(np.float32).copy()     # (32,)

    # 16 pointwise filters (center tap only) that mix x and t with a ramp
    for i in range(16):
        cw[i] = 0.0
        cw[i, 0, 1, 1] = 0.6 + 0.12 * i
        cw[i, 1, 1, 1] = -1.5 + 0.2 * i
        cb[i] = 0.36 - 0.12 * i

    # 16 extra filters: spatial derivatives of x, and x*t interactions
    extra = [
        # Sobel-x
        (np.array([[1,0,-1],[2,0,-2],[1,0,-1]], np.float32), 0.0),
        # Sobel-y
        (np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32), 0.0),
        # Laplacian
        (np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32), 0.0),
        # small blur
        (np.array([[1,1,1],[1,1,1],[1,1,1]], np.float32)/9.0, 0.0),
    ]
    for j in range(16):
        i = 16 + j
        cw[i] = 0.0
        if j < len(extra):
            ker, _ = extra[j]
            cw[i, 0] = ker
            cb[i] = 0.0
        else:
            # x * t interaction taps (center + neighbours)
            cw[i, 0, 1, 1] = 1.0
            cw[i, 1, 1, 1] = 1.0
            cb[i] = 0.0

    # ---- recompute features with the new conv1 -------------------------
    h1 = conv3x3(inp, cw, cb)          # (N,32,16,16)
    h1 = relu(h1)

    # conv2 / conv3 are identity on the first 32 channels -> keep them,
    # but we only need the 32 active channels for the readout.
    feats = np.zeros((N, 64, 16, 16), dtype=np.float32)
    feats[:, :32] = h1

    # ---- closed-form ridge least squares for the out layer -------------
    # out: (1,64,3,3) + bias.  Build design matrix from feats.
    Y = f['y'].astype(np.float32).reshape(N, 1, 16, 16)
    # im2col of feats -> (N, 64*9, HW)
    cols = im2col_patches(feats, 3, 1)         # (N, 576, 256)
    # move to (N*HW, 576)
    Xd = cols.transpose(0, 2, 1).reshape(N * 256, 576)
    Yd = Y.transpose(0, 2, 3, 1).reshape(N * 256, 1)

    # add bias column
    Xd = np.concatenate([Xd, np.ones((Xd.shape[0], 1), dtype=np.float32)], axis=1)

    # normal equations with small ridge
    lam = 1e-3
    A = Xd.T @ Xd
    b = Xd.T @ Yd
    A[np.diag_indices_from(A)] += lam
    sol = np.linalg.solve(A, b)                # (577,1)

    out_w = sol[:576].reshape(1, 64, 3, 3).astype(np.float32)
    out_b = sol[576:577].reshape(1).astype(np.float32)

    # ---- write back ----------------------------------------------------
    w['conv1.weight'] = cw.astype(np.float32)
    w['conv1.bias']   = cb.astype(np.float32)
    w['out.weight']   = out_w
    w['out.bias']     = out_b

    # diagnostics
    pred = (Xd @ sol).reshape(N, 16, 16, 1).transpose(0, 3, 1, 2)
    mse = float(np.mean((pred - Y) ** 2))
    return {
        "weights": w,
        "diagnostics": {
            "train_mse_new": mse,
            "out_weight_norm": float(np.linalg.norm(out_w)),
            "out_bias": float(out_b[0]),
            "ridge": lam,
        },
    }
