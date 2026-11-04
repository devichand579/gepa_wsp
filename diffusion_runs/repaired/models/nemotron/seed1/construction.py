import numpy as np

def edit(w, f):
    """
    One‑shot, closed‑form weight update for the output convolutional layer
    of a DDPM epsilon‑prediction network.

    Parameters
    ----------
    w : dict
        Current network parameters.  Must contain at least the keys
        ``'out.weight'`` (shape (1,64,3,3)) and ``'out.bias'`` (shape (1,)).
    f : dict
        Permitted feedback arrays.  Required keys:
        - ``'y'``          : target Gaussian ε, shape (2000,1,16,16)
        - ``'features'``   : final hidden feature maps, shape (2000,64,16,16)
        - ``'prediction'`` : current ε prediction, shape (2000,1,16,16)

    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        Updated weight dict and a tiny statistics dict with before/after MSE.
    """
    # ------------------------------------------------------------------
    # 1. Pull the permitted arrays (all are float32, already guaranteed)
    # ------------------------------------------------------------------
    y       = f["y"]                # (N,1,16,16)   N = 2000
    features = f["features"]        # (N,64,16,16)
    pred    = f["prediction"]       # (N,1,16,16)

    N = features.shape[0]           # 2000
    H, W = 16, 16
    K = H * W                         # 256 spatial positions

    # ------------------------------------------------------------------
    # 2. Build the design matrix X  (N*K rows, 64 columns)
    # ------------------------------------------------------------------
    # features_flat: (N,64,K)  ->  transpose to (K,N,64) then reshape
    X = features.reshape(N, 64, K).transpose(2, 0, 1).reshape(-1, 64)
    # X shape: (N*K, 64)  == (512000, 64)

    # Target ε flattened over examples and spatial positions
    y_target = y.reshape(N, -1).reshape(-1).astype(np.float32)  # (N*K,)

    # ------------------------------------------------------------------
    # 3. Ordinary least‑squares:  beta = (X^T X)^{-1} X^T y
    # ------------------------------------------------------------------
    A = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])  # add bias column
    beta, _, _, _ = np.linalg.lstsq(A, y_target, rcond=None)

    w_opt = beta[:64].astype(np.float32)      # 64 channel‑wise weights
    b_opt = beta[64].astype(np.float32)       # scalar bias

    # ------------------------------------------------------------------
    # 4. Map the optimal 64‑dim vector into the required out.weight shape
    #    (uniform 3×3 kernel, same weight per channel across all positions)
    # ------------------------------------------------------------------
    new_weight = np.zeros((1, 64, 3, 3), dtype=np.float32)
    # broadcast w_opt[c] to the whole 3×3 slice for channel c
    new_weight[0, :, :, :] = w_opt.reshape(1, 64, 1, 1)   # NumPy broadcasting

    new_bias = np.array(b_opt, dtype=np.float32)   # shape ()

    # ------------------------------------------------------------------
    # 5. Update the weight dict (copy to avoid mutating the input)
    # ------------------------------------------------------------------
    w_out = dict(w)                     # shallow copy
    w_out["out.weight"] = new_weight
    w_out["out.bias"]   = new_bias

    # ------------------------------------------------------------------
    # 6. Diagnostics: MSE before and after the edit on the training set
    # ------------------------------------------------------------------
    # Linear prediction using the new weights applied to the same features:
    #   y_pred_flat = X @ w_opt + b_opt   (size N*K)
    y_pred_flat = (X @ w_opt) + b_opt
    y_pred      = y_pred_flat.reshape(N, 1, H, W).astype(np.float32)

    mse_before = float(np.mean((pred - y) ** 2))
    mse_after  = float(np.mean((y_pred - y) ** 2))

    diagnostics = {
        "mse_before": mse_before,
        "mse_after":  mse_after,
        "reduction":  mse_before - mse_after
    }

    return {"weights": w_out, "diagnostics": diagnostics}
