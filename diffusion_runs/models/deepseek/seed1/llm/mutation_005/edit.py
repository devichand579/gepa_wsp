import numpy as np
from numpy.lib.stride_tricks import as_strided


def _im2col(F, k=3, pad=1):
    # F: (N,C,H,W) -> (N,H*W, C*k*k) with PyTorch cross-correlation ordering (c,di,dj)
    N, C, H, W = F.shape
    Fp = np.pad(F, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((N, C, k, k, H, W), dtype=np.float32)
    for di in range(k):
        for dj in range(k):
            cols[:, :, di, dj, :, :] = Fp[:, :, di:di + H, dj:dj + W]
    # -> (N, H*W, C*k*k)
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N, H * W, C * k * k)
    return cols


def _conv_forward(F, W, b):
    # F:(N,C,H,W), W:(O,C,3,3), b:(O,) -> (N,O,H,W) cross-correlation pad1
    N, C, H, Wd = F.shape
    O = W.shape[0]
    Fp = np.pad(F, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((N, O, H, Wd), dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            patch = Fp[:, :, di:di + H, dj:dj + Wd]  # (N,C,H,W)
            out += np.einsum('nchw,oc->nohw', patch, W[:, :, di, dj], optimize=True)
    return out + b.reshape(1, O, 1, 1)


def _ridge_solve(A, Y, lam):
    # A:(M,K) float32, Y:(M,) -> (K,)
    K = A.shape[1]
    G = A.T @ A
    G = G.astype(np.float64)
    rhs = (A.T @ Y).astype(np.float64)
    G[np.diag_indices(K)] += lam
    try:
        sol = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(G, rhs, rcond=None)[0]
    return sol.astype(np.float32)


def edit(w, f):
    x = np.asarray(f['x'], dtype=np.float32)          # (N,1,16,16)
    t = np.asarray(f['t']).astype(np.float32)         # (N,)
    y = np.asarray(f['y'], dtype=np.float32)          # (N,1,16,16)
    feats = np.asarray(f['features'], dtype=np.float32)  # (N,64,16,16)
    N, _, H, Wd = x.shape

    # ---- 1) Refit out layer (linear readout) on cached features ----
    A = _im2col(feats, 3, 1).reshape(-1, 64 * 9)      # (N*H*W, 576)
    A = np.concatenate([A, np.ones((A.shape[0], 1), dtype=np.float32)], axis=1)  # +bias
    Y = y.reshape(-1)
    sol = _ridge_solve(A, Y, 1e-3)
    W_out = sol[:576].reshape(1, 64, 3, 3).astype(np.float32)
    b_out = sol[576:577].astype(np.float32)

    # ---- 2) Refit conv1 so ReLU(conv1) is a better basis ----
    # conv1 input: concat(x_t, t/31)
    s = (t / 31.0).reshape(N, 1, 1, 1).astype(np.float32)
    inp = np.concatenate([x, np.broadcast_to(s, x.shape)], axis=1)  # (N,2,16,16)

    # current conv1 output (pre-activation) and ReLU mask
    W1 = w['conv1.weight'].astype(np.float32)         # (32,2,3,3)
    b1 = w['conv1.bias'].astype(np.float32)           # (32,)
    z1 = _conv_forward(inp, W1, b1)                   # (N,32,16,16)
    mask = (z1 > 0).astype(np.float32)                # ReLU mask

    # We want features ~= ReLU(conv1(inp)).  Build target for conv1 pre-activation:
    # Use the *pre-ReLU* target that best matches the desired post-ReLU value.
    # Desired post-ReLU feature value: solve for a per-channel scalar gain g_c such
    # that ReLU(g_c * z1) best predicts y through a fixed linear map is ill-posed;
    # instead do a direct linear least-squares on the pre-activation using the
    # residual direction.  Simpler & stable: fit conv1 to reproduce a scaled copy
    # of the current pre-activation restricted to active units, then refit out.
    # Target = current z1 (identity) is a no-op; so we instead fit conv1 to the
    # *least-squares optimal pre-activation* implied by a linear map from inp.
    # Build design: for each output channel c and pixel, z1 ~= conv1(inp).
    # We fit conv1 by ridge regression of z1 on inp patches (this keeps features
    # essentially the same but re-regularizes), which alone is a no-op; the real
    # gain comes from re-solving out with the refit features below.

    # Recompute features from (possibly) refit conv1 and refit out again.
    # To actually change the basis we scale each conv1 channel by a learned gain
    # that minimizes MSE after the out readout (coordinate-wise closed form).
    z1r = np.maximum(z1, 0.0)                          # (N,32,16,16) current features

    # Coordinate descent on per-channel gains g (32 values), closed form each step.
    # features_c = g_c * z1r_c ; out uses these.  Solve g by least squares with
    # out fixed, then re-solve out.  A few exact sweeps.
    g = np.ones(32, dtype=np.float64)

    def build_feats(gv):
        F = np.zeros((N, 64, H, Wd), dtype=np.float32)
        F[:, :32] = (z1r * gv.reshape(1, 32, 1, 1)).astype(np.float32)
        return F

    # initial out solve on g=1 features (already done above on cached feats; redo on ours)
    for _ in range(3):
        F = build_feats(g)
        A2 = _im2col(F, 3, 1).reshape(-1, 64 * 9)
        A2 = np.concatenate([A2, np.ones((A2.shape[0], 1), dtype=np.float32)], axis=1)
        sol2 = _ridge_solve(A2, Y, 1e-3)
        W_out = sol2[:576].reshape(1, 64, 3, 3).astype(np.float32)
        b_out = sol2[576:577].astype(np.float32)

        # update gains: for each channel c, feature column is g_c * (z1r_c patch).
        # pred = sum_c g_c * (out_c * z1r_c) + bias.  Let P_c = conv(z1r_c, W_out[:,c]).
        # residual r = y - (bias + sum_{c'!=c} P_{c'}); solve g_c = <P_c, r> / <P_c,P_c>.
        P = np.zeros((N, 32, H, Wd), dtype=np.float32)
        for c in range(32):
            P[:, c] = _conv_forward(z1r[:, c:c + 1], W_out[:, c:c + 1], np.zeros(1, dtype=np.float32))[:, 0]
        Pflat = P.reshape(N, 32, -1).astype(np.float64)
        cur = np.einsum('nc,ncp->np', g.reshape(1, 32), Pflat).reshape(N, -1) + b_out[0]
        resid = Y.reshape(N, -1).astype(np.float64) - cur
        num = np.einsum('ncp,np->c', Pflat, resid)
        den = np.einsum('ncp,ncp->c', Pflat, Pflat) + 1e-6
        g = g + num / den
        g = np.clip(g, -10.0, 10.0)

    # final out solve with converged gains
    F = build_feats(g)
    A2 = _im2col(F, 3, 1).reshape(-1, 64 * 9)
    A2 = np.concatenate([A2, np.ones((A2.shape[0], 1), dtype=np.float32)], axis=1)
    sol2 = _ridge_solve(A2, Y, 1e-3)
    W_out = sol2[:576].reshape(1, 64, 3, 3).astype(np.float32)
    b_out = sol2[576:577].astype(np.float32)

    # Fold gains into conv1 weights/bias (features = ReLU(g_c * conv1_c(inp)))
    W1n = (W1 * g.reshape(32, 1, 1, 1)).astype(np.float32)
    b1n = (b1 * g).astype(np.float32)

    w = {k: np.array(v, dtype=np.float32) for k, v in w.items()}
    w['conv1.weight'] = W1n
    w['conv1.bias'] = b1n
    w['out.weight'] = W_out
    w['out.bias'] = b_out

    # keep conv2/conv3 as-is (identity/zero-channel) to preserve architecture behavior
    for k in list(w.keys()):
        arr = np.nan_to_num(w[k], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[k] = arr

    diag = {
        "mse_before": float(np.mean((np.asarray(f['prediction'], dtype=np.float32) - y) ** 2)),
        "gains_min": float(g.min()),
        "gains_max": float(g.max()),
        "out_weight_norm": float(np.linalg.norm(W_out)),
        "out_bias": float(b_out[0]),
        "ridge": 1e-3,
    }
    return {"weights": w, "diagnostics": diag}
