import numpy as np

def edit(w, f):
    """
    Closed‑form least‑squares update of the output convolutional layer
    to minimise per‑example MSE between the epsilon prediction and the
    target Gaussian epsilon, using the provided feedback arrays.

    Parameters
    ----------
    w : dict
        Current network parameters.  Keys include
        ``conv1.weight``, ``conv1.bias``, ``conv2.weight``, ``conv2.bias``,
        ``conv3.weight``, ``conv3.bias``, ``out.weight``, ``out.bias``.
    f : dict
        Feedback arrays:
        ``features``   – final hidden feature maps [N,64,16,16]
        ``prediction`` – current epsilon prediction      [N,1,16,16]
        ``y``          – target Gaussian epsilon         [N,1,16,16]
        (other keys are ignored)

    Returns
    -------
    w : dict
        Updated weights (only ``out.weight`` and ``out.bias`` may change).
    diagnostics : dict
        ``mse_before``, ``mse_after`` and ``weight_change`` for the
        output layer.
    """
    # ------------------------------------------------------------------
    # 1.  Pull the relevant feedback arrays (all are guaranteed to have
    #     the shapes described in the task).
    # ------------------------------------------------------------------
    features = f['features']          # shape [N, 64, 16, 16], float32
    pred     = f['prediction']        # shape [N, 1, 16, 16], float32
    y        = f['y']                 # shape [N, 1, 16, 16], float32

    N = features.shape[0]             # number of training examples
    # ------------------------------------------------------------------
    # 2.  Build the design matrix X.
    #     For each spatial output position (i,j) we extract the 3×3×64
    #     patch from the feature map (zero‑padded “same” convolution).
    #     X will have shape (N*256, 576)  (576 = 64 * 3 * 3).
    # ------------------------------------------------------------------
    n_patches = N * 16 * 16           # 256 per example
    X = np.empty((n_patches, 64 * 3 * 3), dtype=np.float32)
    y_vec = y.reshape(-1).astype(np.float32)   # target, length n_patches

    idx = 0
    for n in range(N):
        feat = features[n]            # (64,16,16)
        # zero‑pad the feature map so that a 3×3 kernel with stride 1
        # and padding=1 can be applied without boundary artefacts.
        pad = np.zeros((64, 18, 18), dtype=np.float32)
        pad[:, 1:17, 1:17] = feat

        for i in range(16):
            for j in range(16):
                patch = pad[:, i:i+3, j:j+3].ravel()   # 576 elements
                X[idx] = patch
                idx += 1

    # ------------------------------------------------------------------
    # 3.  Closed‑form least‑squares solution for the output weight vector.
    # ------------------------------------------------------------------
    # Current output weight (flattened to 576‑dim vector)
    w_out_cur = w['out.weight'].astype(np.float32).reshape(-1)   # shape (576,)

    # Solve  X @ w ≈ y_vec   (no bias term yet)
    coeff, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)   # coeff shape (576,)

    # Reshape to the original weight tensor [1,64,3,3]
    new_w_out = coeff.reshape(w['out.weight'].shape).astype(np.float32)

    # ------------------------------------------------------------------
    # 4.  Compute the optimal bias term.
    # ------------------------------------------------------------------
    # Predicted values (without bias) = X @ coeff
    y_pred_no_bias = X @ coeff                                 # length n_patches
    # Bias = mean( target - prediction_no_bias )
    bias_new = np.mean(y_vec - y_pred_no_bias).astype(np.float32)

    # ------------------------------------------------------------------
    # 5.  Update the weight dictionary.
    # ------------------------------------------------------------------
    w = {k: v.copy() for k, v in w.items()}   # shallow copy to avoid mutating input
    w['out.weight'] = new_w_out
    w['out.bias']   = np.float32(bias_new)

    # ------------------------------------------------------------------
    # 6.  Diagnostics – MSE before/after the edit and weight change.
    # ------------------------------------------------------------------
    # MSE before the edit (using the original prediction)
    mse_before = float(np.mean((pred - y) ** 2))

    # MSE after the edit – reconstruct the predicted epsilon using the
    # new linear map (patches → dot‑product + bias) and reshape back.
    y_pred_vec = y_pred_no_bias + bias_new               # length n_patches
    y_pred_new = y_pred_vec.reshape(pred.shape).astype(np.float32)
    mse_after  = float(np.mean((y_pred_new - y) ** 2))

    weight_change = float(np.max(np.abs(new_w_out - w_out_cur)))

    diagnostics = {
        'mse_before': mse_before,
        'mse_after':  mse_after,
        'weight_change': weight_change,
    }

    return w, diagnostics
