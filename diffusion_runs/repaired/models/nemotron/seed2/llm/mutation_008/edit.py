import numpy as np

def edit(w, f):
    """
    Closed‑form update of the output conv layer (out.weight + out.bias)
    using the provided training feedback arrays. No autograd, no iterative optimiser.
    """
    # ----- extract permitted feedback arrays -----
    features = f['features']            # shape (2000, 64, 16, 16)
    y        = f['y']                   # shape (2000, 1, 16, 16)
    # prediction is not needed for the fit, but kept for possible diagnostics

    N = features.shape[0]               # 2000 training examples
    C = 64                              # input channels to out Conv
    H = W = 16                          # spatial size (output of previous layers)
    K = 3                               # kernel size of out Conv
    M = N * H * W                       # total rows = 512 000

    # ----- build Gram matrix AᵀA and cross‑term Aᵀy (including bias) -----
    # we augment each row with a constant 1 so the solved bias is returned
    G = np.zeros((577, 577), dtype=np.float64)   # 576 weights + 1 bias
    h = np.zeros(577, dtype=np.float64)

    for i in range(N):
        feat = features[i]                     # (64,16,16)
        for hp in range(H):
            for wp in range(W):
                # ----- extract the 576‑element patch that the conv uses -----
                r = np.zeros(576, dtype=np.float64)
                idx = 0
                for c in range(C):
                    for kh in range(K):
                        for kw in range(K):
                            # PyTorch conv with padding=1 uses padded input;
                            # the effective input index is (hp+kh-1, wp+kw-1)
                            fh = hp + kh - 1
                            fw = wp + kw - 1
                            if 0 <= fh < H and 0 <= fw < W:
                                val = float(feat[c, fh, fw])
                            else:
                                val = 0.0          # zero padding
                            r[idx] = val
                            idx += 1
                # ----- augment with bias term (value 1) -----
                r_ext = np.zeros(577, dtype=np.float64)
                r_ext[:576] = r
                r_ext[576] = 1.0

                # ----- accumulate G += r_ext ⊗ r_ext  and  h += r_ext * y_val -----
                y_val = float(y[i, 0, hp, wp])   # target epsilon at this spatial pos
                G += np.outer(r_ext, r_ext)
                h += r_ext * y_val

    # ----- solve the normal equations (ridge‑free, but lstsq for safety) -----
    # G is 577×577, h is length 577
    w_ext, residuals, rank, s = np.linalg.lstsq(G, h, rcond=None)
    # w_ext[:576] = optimal weights, w_ext[576] = optimal bias

    # ----- reshape and clip -----
    new_weight = w_ext[:576].astype(np.float32).reshape(1, C, K, K)   # (1,64,3,3)
    new_bias   = float(w_ext[576]).astype(np.float32)                 # scalar

    # keep values in a sane range (the task limits |v| <= 1000)
    new_weight = np.clip(new_weight, -1000, 1000)
    new_bias   = np.clip(new_bias, -1000, 1000)

    # ----- update the weight dict, leave everything else untouched -----
    w = dict(w)                              # shallow copy to avoid mutating input
    w['out.weight'] = new_weight
    w['out.bias'] = np.array([new_bias], dtype=np.float32)   # shape (1,)

    # ----- diagnostics (small JSON‑compatible dict) -----
    # compute closed‑form training MSE from the normal equations
    y_flat = y.reshape(-1).astype(np.float64)          # length M
    y_sum_sq = np.sum(y_flat ** 2)
    # residual sum of squares = yᵀy – w_extᵀ h
    rss = y_sum_sq - float(np.dot(w_ext, h))
    after_mse = rss / M

    # MSE of the current prediction (before edit) for reference
    pred_flat = w['prediction'].reshape(-1).astype(np.float64) if 'prediction' in f else np.zeros(M)
    before_mse = np.mean((pred_flat - y_flat) ** 2) if len(pred_flat) == M else float('nan')

    diag = {
        "before_mse": before_mse,
        "after_mse": float(after_mse),
        "weight_change_frobenius": float(np.linalg.norm(
            new_weight.astype(np.float64) -
            w['out.weight'].astype(np.float64)
        )),
        "solved_bias": float(new_bias),
        "rank": int(rank)
    }

    # The problem permits returning just w or {"weights": w, "diagnostics": …}
    return {"weights": w, "diagnostics": diag}
