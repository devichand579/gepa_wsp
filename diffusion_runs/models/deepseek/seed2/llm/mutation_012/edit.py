import numpy as np

def edit(w, f):
    # ---------- helpers ----------
    def im2col(x, k=3, pad=1):
        # x: (N,C,H,W) -> (N, C*k*k, H*W) with zero padding
        N, C, H, W = x.shape
        xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
        cols = np.empty((N, C, k, k, H, W), dtype=np.float32)
        for i in range(k):
            for j in range(k):
                cols[:, :, i, j, :, :] = xp[:, :, i:i+H, j:j+W]
        return cols.reshape(N, C*k*k, H*W)

    def conv_forward(x, Wt, b):
        # x: (N,Cin,H,W), Wt: (Cout,Cin,3,3)
        N, Cin, H, W = x.shape
        Cout = Wt.shape[0]
        cols = im2col(x)                      # (N, Cin*9, H*W)
        Wm = Wt.reshape(Cout, Cin*9)          # (Cout, Cin*9)
        out = np.einsum('oc,ncp->nop', Wm, cols)  # (N,Cout,H*W)
        out = out + b.reshape(1, Cout, 1)
        return out.reshape(N, Cout, H, W)

    # ---------- recover arrays ----------
    x   = np.asarray(f['x'], dtype=np.float32)          # (2000,1,16,16)
    t   = np.asarray(f['t'], dtype=np.float32)          # (2000,)
    y   = np.asarray(f['y'], dtype=np.float32)          # (2000,1,16,16)
    feat= np.asarray(f['features'], dtype=np.float32)   # (2000,64,16,16) conv3 out (post-ReLU)

    W1 = np.asarray(w['conv1.weight'], dtype=np.float32)
    b1 = np.asarray(w['conv1.bias'],   dtype=np.float32)
    W2 = np.asarray(w['conv2.weight'], dtype=np.float32)
    b2 = np.asarray(w['conv2.bias'],   dtype=np.float32)
    W3 = np.asarray(w['conv3.weight'], dtype=np.float32)
    b3 = np.asarray(w['conv3.bias'],   dtype=np.float32)
    Wo = np.asarray(w['out.weight'],   dtype=np.float32)
    bo = np.asarray(w['out.bias'],     dtype=np.float32)

    N = x.shape[0]
    # ---------- 1) exact ridge LS readout on cached features ----------
    # design: for each example & pixel, 64*9 features from 3x3 neighbourhood of feat
    A = im2col(feat)                       # (N, 64*9, 256)
    A = A.transpose(0, 2, 1).reshape(-1, 64*9)   # (N*256, 576)
    Y = y.reshape(-1, 1)                          # (N*256, 1)

    # normal equations with tiny ridge for numerical stability
    G = A.T @ A                                # (576,576)
    scale = np.trace(G) / G.shape[0] + 1e-12
    lam = 1e-6 * scale
    G[np.diag_indices_from(G)] += lam
    rhs = A.T @ Y
    sol = np.linalg.solve(G, rhs)              # (576,1)
    Wo_new = sol.reshape(1, 64, 3, 3).astype(np.float32)
    # refit bias given new Wo (mean residual)
    pred_lin = (A @ sol).reshape(N, 256)
    bo_new = np.float32(np.mean(Y.reshape(N, 256) - pred_lin))

    # ---------- 2) target propagation: desired conv3 output ----------
    # want Wo_new * a3 + bo_new ~= y  =>  a3 ~= pinv(Wo_new) (y - bo_new)
    Wom = Wo_new.reshape(1, 64 * 9)            # (1,576)
    # minimum-norm per-pixel target for the 3x3 conv output (ignore ReLU sign)
    # pinv of a row vector: v / (v v^T)
    denom = float(np.dot(Wom.ravel(), Wom.ravel())) + 1e-12
    # target for conv3 *pre-activation* (before ReLU) at each pixel
    # We build the target on the 3x3 neighbourhood space then invert the conv.
    # Simpler: use the current conv3 output as the target basis and only
    # re-scale it so that the new readout matches y as closely as possible.
    # (Keeps ReLU validity and avoids unstable pinv blow-up.)
    cur_a3 = feat.reshape(N, 64, 256)
    cur_pred = (Wom @ im2col(feat).reshape(N, 576, 256)).reshape(N, 256) + bo_new
    # per-pixel scalar correction so that readout(cur_a3 * s) ~= y
    # (scale the whole 64-channel feature vector per pixel)
    num = np.sum(cur_a3 * cur_a3, axis=1)      # (N,256)
    # We solve for a single global scale alpha minimising ||alpha*cur_pred + bo - y||
    resid = Y.reshape(N, 256) - bo_new
    alpha = float(np.sum(cur_pred * resid) / (np.sum(cur_pred * cur_pred) + 1e-12))
    alpha = float(np.clip(alpha, 0.5, 2.0))

    # ---------- 3) refit conv3 to the scaled target (closed form, per out-channel) ----------
    # input to conv3 = conv2 output; recompute it exactly with current weights
    xin = np.concatenate([x, np.broadcast_to((t / 31.0).reshape(N, 1, 1, 1), (N, 1, 16, 16))], axis=1)
    a1 = np.maximum(conv_forward(xin, W1, b1), 0.0)
    a2 = np.maximum(conv_forward(a1,  W2, b2), 0.0)

    A2 = im2col(a2)                            # (N, 64*9, 256)
    A2 = A2.transpose(0, 2, 1).reshape(-1, 64 * 9)   # (N*256, 576)
    T3 = (alpha * cur_a3).transpose(0, 2, 1).reshape(-1, 64)  # (N*256, 64)

    # solve  A2 @ W3c^T ~= T3  with per-output-channel ridge LS
    G2 = A2.T @ A2                             # (576,576)
    scale2 = np.trace(G2) / G2.shape[0] + 1e-12
    G2[np.diag_indices_from(G2)] += 1e-5 * scale2
    RHS = A2.T @ T3                            # (576,64)
    W3_sol = np.linalg.solve(G2, RHS)          # (576,64)
    W3_new = W3_sol.T.reshape(64, 64, 3, 3).astype(np.float32)
    # bias: mean residual per channel
    pred3 = (A2 @ W3_sol)                      # (N*256,64)
    b3_new = np.mean(T3 - pred3, axis=0).astype(np.float32)

    # ---------- 4) refit out layer on the *new* conv3 features ----------
    a3_new = np.maximum(conv_forward(a2, W3_new, b3_new), 0.0)
    A3 = im2col(a3_new).transpose(0, 2, 1).reshape(-1, 64 * 9)
    G3 = A3.T @ A3
    scale3 = np.trace(G3) / G3.shape[0] + 1e-12
    G3[np.diag_indices_from(G3)] += 1e-6 * scale3
    sol3 = np.linalg.solve(G3, A3.T @ Y)
    Wo_final = sol3.reshape(1, 64, 3, 3).astype(np.float32)
    bo_final = np.float32(np.mean(Y - (A3 @ sol3)))

    # ---------- write back ----------
    w['conv3.weight'] = W3_new
    w['conv3.bias']   = b3_new
    w['out.weight']   = Wo_final
    w['out.bias']     = np.array([bo_final], dtype=np.float32)

    # ---------- diagnostics ----------
    mse_before = float(np.mean((np.asarray(f['prediction'], dtype=np.float32) - y) ** 2))
    # evaluate new pipeline on the 2000 training examples
    a1n = np.maximum(conv_forward(xin, W1, b1), 0.0)
    a2n = np.maximum(conv_forward(a1n, W2, b2), 0.0)
    a3n = np.maximum(conv_forward(a2n, W3_new, b3_new), 0.0)
    predn = conv_forward(a3n, Wo_final, bo_final)
    mse_after = float(np.mean((predn - y) ** 2))

    diag = {
        "mse_before": mse_before,
        "mse_after_train": mse_after,
        "alpha": alpha,
        "ridge_out": float(lam),
        "ridge_conv3": float(1e-5 * scale2),
        "out_weight_max": float(np.max(np.abs(Wo_final))),
        "conv3_weight_max": float(np.max(np.abs(W3_new))),
        "finite": bool(np.isfinite(Wo_final).all() and np.isfinite(W3_new).all()),
    }
    return {"weights": w, "diagnostics": diag}
