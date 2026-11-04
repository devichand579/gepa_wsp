import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    """
    Edit the DDPM epsilon‑prediction network using a closed‑form least‑squares fit.
    
    Parameters
    ----------
    w : dict
        Current full set of network parameters (contains at least
        ``out.weight`` of shape (1,64,3,3) and ``out.bias`` of shape (1,)).
    f : dict
        Feedback arrays; must contain the keys:
        - ``features``   : shape (N,64,16,16)  – final hidden feature maps
        - ``prediction`` : shape (N,1,16,16)  – current epsilon prediction
        - ``y``          : shape (N,1,16,16)  – target Gaussian epsilon
    
    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        Updated weight dict and a small statistics dictionary.
    """
    # ------------------------------------------------------------------
    # 1. Pull the permitted feedback arrays
    # ------------------------------------------------------------------
    features = f["features"]                # (N, 64, 16, 16), float32
    prediction = f["prediction"]            # (N, 1, 16, 16), float32
    y = f["y"]                              # (N, 1, 16, 16), float32

    N = features.shape[0]                   # number of training examples

    # ------------------------------------------------------------------
    # 2. Build the design matrix A that maps out.weight → prediction
    # ------------------------------------------------------------------
    # Pad feature maps with zeros so that 'same' convolution (pad=1) is
    # equivalent to extracting 3×3 patches centred at every output pixel.
    padded = np.pad(features, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant", constant_values=0.0)   # (N,64,18,18)

    # Extract all 3×3 patches for every spatial position (same 'valid' after padding)
    # sliding_window_view returns (N, 64, 16, 16, 3, 3)
    patches = sliding_window_view(padded, (3, 3), axis=(2, 3))

    # Reorder dimensions so that channels come first for each patch:
    # (N, 16, 16, 64, 3, 3) → (N, 256, 576) after flattening
    patches = patches.transpose(0, 2, 3, 1, 4, 5)          # (N,16,16,64,3,3)
    A = patches.reshape(N, -1, 576)                       # (N, 256, 576)
    A = A.reshape(-1, 576)                                # (N*256, 576)

    # Target vector: every output pixel of every example
    y_vec = prediction.reshape(-1)                        # (N*256,)

    # ------------------------------------------------------------------
    # 3. Closed‑form least‑squares solution for out.weight
    # ------------------------------------------------------------------
    w_vec, sumsq, rank, s = np.linalg.lstsq(A, y_vec, rcond=None)
    # w_vec has length 576; reshape to (1,64,3,3)
    new_weight = w_vec.reshape(1, 64, 3, 3).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Compute the new bias (out.bias) that minimises the residual
    # ------------------------------------------------------------------
    pred_resid = A @ w_vec                                 # (N*256,)
    bias_val = float(np.mean(y_vec - pred_resid))
    new_bias = np.array([bias_val], dtype=np.float32)

    # ------------------------------------------------------------------
    # 5. Update the weight dict (only the final layer changes)
    # ------------------------------------------------------------------
    w["out.weight"] = new_weight
    w["out.bias"] = new_bias

    # ------------------------------------------------------------------
    # 6. Diagnostics
    # ------------------------------------------------------------------
    mse_before = float(np.mean((prediction - y) ** 2))
    mse_after  = float(sumsq / (N * 256))                # RSS → MSE

    diagnostics = {
        "mse_before": mse_before,
        "mse_after":  mse_after,
        "weight_change_mean_abs": float(np.mean(np.abs(
            new_weight - w["out.weight"].astype(np.float32)))),
        "bias_change": float(new_bias - w["out.bias"].astype(np.float32))
    }

    return {"weights": w, "diagnostics": diagnostics}
