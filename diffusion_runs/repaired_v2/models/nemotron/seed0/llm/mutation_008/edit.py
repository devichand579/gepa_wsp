import numpy as np

def edit(w, f):
    """
    Closed‑form bias‑and‑weight update for the DDPM epsilon‑prediction network.
    Only the output convolution (out.weight, out.bias) is modified;
    all earlier layers and cached features stay untouched.

    Parameters
    ----------
    w : dict
        Current network parameters (full precision).  Keys include
        'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
        'conv3.weight', 'conv3.bias', 'out.weight', 'out.bias'.
    f : dict
        Feedback arrays (the "feedback_arrays" dict from the task).  Must contain
        'features'   – shape (2000, 64, 16, 16)   – final hidden maps,
        'prediction' – shape (2000, 1, 16, 16)   – current ε‑prediction,
        'y'          – shape (2000, 1, 16, 16)   – target Gaussian ε.

    Returns
    -------
    w : dict
        Updated weights (only out.weight / out.bias change).
    {"weights": w, "diagnostics": {...}}   # JSON‑compatible stats
    """
    # ------------------------------------------------------------------
    # 1. Pull the permitted feedback arrays
    # ------------------------------------------------------------------
    features = f['features']          # (N, 64, 16, 16), N=2000
    prediction = f['prediction']      # (N, 1, 16, 16)
    y = f['y']                       # (N, 1, 16, 16)

    e = y - prediction                # error map, shape (N, 1, 16, 16)
    N, C, H, W = features.shape       # C=64, H=W=16

    # ------------------------------------------------------------------
    # 2. Pad features so that a 3×3 receptive field can be extracted
    #    for every output pixel (same‑padding behaviour).
    # ------------------------------------------------------------------
    features_pad = np.pad(features, ((0,0),(0,0),(1,1),(1,1)),
                        mode='constant', constant_values=0.)
    # now features_pad shape (N,64,18,18)

    # ------------------------------------------------------------------
    # 3. Build the design matrix X (rows = N * H * W = 512 000)
    #    and the target vector y_vec (same length).
    #    For each example n and each output pixel (i,j) we flatten the
    #    3×3×64 receptive‑field patch into a 576‑dim vector.
    # ------------------------------------------------------------------
    total_pixels = N * H * W               # 512 000
    X = np.empty((total_pixels, C * 3 * 3))   # 576 columns
    y_vec = np.empty(total_pixels)

    idx = 0
    for n in range(N):
        # sliding window gives (64, 16, 16, 3, 3)
        patches = np.lib.stride_tricks.sliding_window_view(
                    features_pad[n], (3, 3), axis=(1, 2))
        # transpose to (16, 16, 64, 3, 3) then reshape each patch
        patches = np.transpose(patches, (1, 2, 0, 3, 4))   # (16,16,64,3,3)
        patches_flat = patches.reshape(H * W, C * 3 * 3)   # (256,576)

        e_flat = e[n].reshape(H * W)   # (256,)

        X[idx:idx + H*W, :] = patches_flat
        y_vec[idx:idx + H*W] = e_flat
        idx += H * W

    # ------------------------------------------------------------------
    # 4. Add a bias column (ones) and solve the normal equations
    #    min || X_b @ [w; b] – y_vec ||²   using np.linalg.lstsq.
    # ------------------------------------------------------------------
    X_b = np.hstack([X, np.ones((total_pixels, 1))])   # (512k, 577)
    coeff, _, _, _ = np.linalg.lstsq(X_b, y_vec, rcond=None)

    weight_flat = coeff[: C * 3 * 3]          # 576 values
    bias = float(coeff[-1])                    # scalar

    # ------------------------------------------------------------------
    # 5. Reshape the updated output weights and bias back to the
    #    network’s expected shapes and cast to float32.
    # ------------------------------------------------------------------
    new_weight = weight_flat.reshape(1, C, 3, 3).astype(np.float32)
    new_bias   = np.array([bias], dtype=np.float32)

    w['out.weight'] = new_weight
    w['out.bias']   = new_bias

    # ------------------------------------------------------------------
    # 6. Diagnostics: training MSE after the update, bias shift, etc.
    # ------------------------------------------------------------------
    rss = np.sum((X_b @ coeff - y_vec) ** 2)          # residual sum‑of‑squares
    mse_after = rss / total_pixels

    diag = {
        "examples_used": int(N),
        "pixels_per_example": H * W,
        "total_pixels": total_pixels,
        "mse_after_update": float(mse_after),
        "bias_shift": bias,
        "weight_norm": float(np.linalg.norm(weight_flat)),
    }

    return w, {"weights": w, "diagnostics": diag}
