import numpy as np

def edit(w, f):
    """
    Edit the fixed DDPM epsilon‑prediction network using a single closed‑form
    least‑squares update of the output layer.

    Parameters
    ----------
    w : dict
        Current network parameters (all weight tensors). Keys include
        'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
        'conv3.weight', 'conv3.bias', 'out.weight', 'out.bias'.
    f : dict
        Feedback arrays supplied by the task:
            - y          : [2000,1,16,16] target Gaussian ε
            - features   : [2000,64,16,16] final hidden feature maps
            - prediction : [2000,1,16,16] current ε prediction
            - (other arrays may be present but are not needed)

    Returns
    -------
    w_new : dict
        Updated weights (only out.weight and out.bias may change).
    diagnostics : dict
        Small statistics dictionary.
    """
    # ------------------------------------------------------------------
    # 1. Pull the required feedback arrays
    # ------------------------------------------------------------------
    y       = f['y']                # shape (2000, 1, 16, 16)
    features = f['features']        # shape (2000, 64, 16, 16)
    prediction = f['prediction']    # shape (2000, 1, 16, 16)

    N, C, H, W = features.shape      # N=2000, C=64, H=W=16
    error = y - prediction           # (2000,1,16,16)

    # ------------------------------------------------------------------
    # 2. Build the normal‑equation accumulators for the output layer.
    #    The output conv has weight shape (1,64,3,3) → 576 parameters.
    #    For every spatial position we extract a 3×3 patch from the
    #    64‑channel feature map (zero‑padded “same” convolution) and
    #    flatten it to a 576‑vector  v.  The scalar error at that pixel
    #    is e.  We accumulate  XᵀX  and  Xᵀe  across all examples and
    #    positions without ever materialising the full design matrix.
    # ------------------------------------------------------------------
    K = 3 * 3 * C          # 576
    XTX = np.zeros((K, K), dtype=np.float64)
    Xty = np.zeros(K,      dtype=np.float64)

    error_sum = 0.0
    pixel_cnt = 0

    for i in range(N):
        feat = features[i]                # (64,16,16)
        err  = error[i][0]                # (16,16)

        # zero‑pad once per example (same‑padding convention)
        pad = np.zeros((C, H + 2, W + 2), dtype=np.float32)
        pad[:, 1:-1, 1:-1] = feat

        for h in range(H):
            for w in range(W):
                patch = pad[:, h:h + 3, w:w + 3].reshape(-1)   # (576,)
                ev    = err[h, w]

                XTX += np.outer(patch.astype(np.float64),
                                patch.astype(np.float64))
                Xty += ev * patch.astype(np.float64)

                error_sum += ev
                pixel_cnt += 1

    # ------------------------------------------------------------------
    # 3. Solve the normal equations  (XTX) w = Xty  in closed form.
    # ------------------------------------------------------------------
    w_vec, residuals, rank, s = np.linalg.lstsq(XTX, Xty, rcond=None)
    # w_vec has length 576
    new_out_weight = w_vec.reshape(1, 64, 3, 3).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Compute the bias term that minimises MSE given the new weights.
    #    We reuse the same patch extraction to obtain  e – v·w_vec  for
    #    every pixel and average it.
    # ------------------------------------------------------------------
    bias_sum = 0.0
    for i in range(N):
        feat = features[i]
        err  = error[i][0]
        pad  = np.zeros((C, H + 2, W + 2), dtype=np.float32)
        pad[:, 1:-1, 1:-1] = feat
        for h in range(H):
            for w in range(W):
                patch = pad[:, h:h + 3, w:w + 3].reshape(-1)
                ev    = err[h, w]
                pred  = np.dot(patch, w_vec)          # v·w_vec
                bias_sum += (ev - pred)

    new_out_bias = np.float32(bias_sum / pixel_cnt if pixel_cnt > 0 else 0.0)

    # ------------------------------------------------------------------
    # 5. Clip to finite float32 and keep values ≤ 1000 (as required).
    # ------------------------------------------------------------------
    new_out_weight = np.clip(new_out_weight, -1000, 1000).astype(np.float32)
    new_out_bias   = np.clip(new_out_bias,   -1000, 1000).astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Copy the original weight dict and replace only the output layer.
    # ------------------------------------------------------------------
    w_new = {k: v.copy() for k, v in w.items()}
    w_new['out.weight'] = new_out_weight
    w_new['out.bias']   = new_out_bias

    # ------------------------------------------------------------------
    # 7. Diagnostics: MSE before the edit (using the original prediction)
    # ------------------------------------------------------------------
    mse_before = float(np.mean((y - prediction) ** 2))

    diagnostics = {
        "mse_before": mse_before,
        "num_pixels": int(pixel_cnt),
        "out_weight_norm": float(np.linalg.norm(new_out_weight)),
        "out_bias": float(new_out_bias)
    }

    return w_new, diagnostics
