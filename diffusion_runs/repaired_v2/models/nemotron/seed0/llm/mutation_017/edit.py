import numpy as np
from numpy.lib.stride_tricks import as_strided

def edit(w, f):
    """
    Closed‑form update of the output convolutional layer (out.weight,
    out.bias) by minimising the mean‑squared error between the current
    epsilon prediction and the target Gaussian epsilon, using only the
    permitted feedback arrays (features, prediction, y).

    Parameters
    ----------
    w : dict
        Full current parameter set (all layers, biases, etc.).
    f : dict
        Feedback arrays:
            - f['features']   shape (N, 64, 16, 16)   – final hidden
              feature maps for N training examples.
            - f['prediction'] shape (N, 1, 16, 16)   – current ε‑prediction.
            - f['y']          shape (N, 1, 16, 16)   – target Gaussian ε.

    Returns
    -------
    w : dict
        Updated weight dictionary (only out.weight / out.bias change).
    diagnostics : dict
        Small statistics (pre‑ and post‑edit MSE, condition number, etc.).
    """
    # ------------------------------------------------------------------
    # 1. Pull the permitted arrays (all are full‑precision float32)
    # ------------------------------------------------------------------
    features = f['features']            # (N, 64, 16, 16)
    pred     = f['prediction']          # (N, 1, 16, 16)
    target   = f['y']                   # (N, 1, 16, 16)

    N, C, H, W = features.shape        # N=2000, C=64, H=W=16
    K = 3                                # kernel size
    out_ch = 1
    in_ch  = C

    # ------------------------------------------------------------------
    # 2. Build the im2col design matrix X  (shape (N*H*W, in_ch*K*K))
    # ------------------------------------------------------------------
    # Pad feature maps with zeros so that a 3×3 kernel with padding=1
    # preserves the 16×16 output size and out‑of‑bounds entries are zero.
    pad = 1
    feat_pad = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=np.float32)
    feat_pad[:, :, pad:pad+H, pad:pad+W] = features.astype(np.float32)

    # Number of output pixels per sample
    n_out = H * W                     # 256
    X_rows = N * n_out
    D = in_ch * K * K                 # 64 * 3 * 3 = 576
    X = np.empty((X_rows, D), dtype=np.float32)
    y = np.empty(X_rows, dtype=np.float32)

    # Fill X and y by iterating over samples and spatial positions
    idx = 0
    for n in range(N):
        fp = feat_pad[n]               # (C, H+2, W+2)
        # For each output pixel (i,j) extract the 3×3 patch, flatten
        for i in range(H):
            for j in range(W):
                # patch = fp[:, i:i+3, j:j+3]   # shape (C,3,3)
                patch = fp[:, i:i+K, j:j+K].ravel()   # (576,)
                X[idx, :] = patch
                y[idx] = target[n, 0, i, j]          # scalar ε
                idx += 1

    # ------------------------------------------------------------------
    # 3. Closed‑form least‑squares solution (one normal‑equation step)
    # ------------------------------------------------------------------
    # ridge term to guarantee invertibility (tiny regularisation)
    lam = 1e-6
    Gram = X.T @ X + lam * np.eye(D, dtype=np.float32)
    # solve Gram * w = X.T @ y
    w_out_flat = np.linalg.solve(Gram, X.T @ y)   # (576,)

    # ------------------------------------------------------------------
    # 4. Bias from the residual
    # ------------------------------------------------------------------
    pred_flat = X @ w_out_flat                     # (X_rows,)
    b_out = np.mean(target.astype(np.float32).ravel() - pred_flat)   # scalar

    # ------------------------------------------------------------------
    # 5. Reshape and clip to finite float32, |value| ≤ 1000
    # ------------------------------------------------------------------
    w_out = w_out_flat.reshape((out_ch, in_ch, K, K)).astype(np.float32)
    b_out = np.float32(b_out)

    # Clip just in case (the problem limits |value| ≤ 1000)
    w_out = np.clip(w_out, -1000.0, 1000.0)
    b_out = np.clip(b_out, -1000.0, 1000.0)

    # ------------------------------------------------------------------
    # 6. Update the weight dictionary (preserve all other keys/shapes)
    # ------------------------------------------------------------------
    w = {k: v.copy() for k, v in w.items()}   # shallow copy to avoid mutating input
    w['out.weight'] = w_out
    w['out.bias']   = np.array([b_out], dtype=np.float32)   # shape (1,)

    # ------------------------------------------------------------------
    # 7. Diagnostics – MSE before/after on the training batch
    # ------------------------------------------------------------------
    # Compute MSE using the *original* prediction (for reference) and
    # the new prediction obtained by a forward pass with the updated weights.
    # Here we just evaluate the new MSE analytically via the least‑squares
    # solution (since X and y are known).  The “before” MSE is the supplied
    # full‑batch MSE from the feedback metrics.
    mse_before = 1.0037   # from context_representation.full_feedback_metrics.mse

    # New prediction = X @ w_out + b_out (broadcast back to (N,1,16,16))
    new_pred_flat = X @ w_out_flat + b_out
    # reshape to (N,1,16,16) – we only need the mean squared error
    new_pred = new_pred_flat.reshape((N, 1, H, W))
    mse_after = np.mean((new_pred - target.astype(np.float32))**2)

    cond_np = np.linalg.cond(Gram)   # condition number of the regularised Gram matrix

    diagnostics = {
        "mse_before": float(mse_before),
        "mse_after":  float(mse_after),
        "reduction":  float(mse_before - mse_after),
        "gram_cond":  float(cond_np),
        "n_samples":  int(N),
        "n_patches":  int(n_out),
        "weight_norm": float(np.linalg.norm(w_out)),
        "bias":        float(b_out),
    }

    return w, diagnostics
