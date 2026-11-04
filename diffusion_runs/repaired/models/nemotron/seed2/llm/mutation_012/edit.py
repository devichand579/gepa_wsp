import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    """
    Closed‑form update of the output layer of a DDPM epsilon‑prediction network.
    Only the output convolution (out.weight, out.bias) is modified; all other
    parameters stay untouched.  The update minimises the per‑pixel MSE between
    the current epsilon prediction and the supplied target epsilon using a
    ridge‑regression normal‑equation solution – no autograd, back‑prop or
    iterative optimiser is used.

    Parameters
    ----------
    w : dict
        Current network parameters.  Expected keys: "out.weight" (shape
        (1,64,3,3)), "out.bias" (shape (1,)).
    f : dict
        Permitted feedback arrays.  Required keys:
        - "features": final hidden feature maps, shape (N,64,16,16)
        - "y": target Gaussian epsilon, shape (N,1,16,16)
        - "prediction": current epsilon prediction, shape (N,1,16,16)

    Returns
    -------
    {"weights": w_updated, "diagnostics": small_stats_dict}
    """
    # ------------------------------------------------------------------
    # 1. Pull the arrays we need
    # ------------------------------------------------------------------
    out_weight = w["out.weight"]            # (1,64,3,3)
    out_bias   = w["out.bias"]              # (1,)
    features   = f["features"]              # (N,64,16,16)   N = 2000
    y_target   = f["y"]                     # (N,1,16,16)
    pred       = f["prediction"]            # (N,1,16,16)

    N, C, H, W = features.shape             # C=64, H=W=16

    # ------------------------------------------------------------------
    # 2. Build the im2col design matrix X from the final features
    # ------------------------------------------------------------------
    # Zero‑pad spatially (same padding as the original conv)
    padded = np.pad(features, ((0,0),(0,0),(1,1),(1,1)),
                    mode="constant")               # (N,64,18,18)

    # sliding_window_view gives a view of shape (N,C,H,W,3,3) covering
    # every output‑pixel patch (the original output size H×W).
    patches = sliding_window_view(padded, (3,3), axis=(2,3))
    # Reshape to (N*H*W, C*3*3) i.e. one row per spatial location per example
    X = patches.reshape(-1, C * 3 * 3)       # (N*256, 576)

    # ------------------------------------------------------------------
    # 3. Flatten targets and current predictions
    # ------------------------------------------------------------------
    y_flat = y_target.reshape(-1).astype(np.float32)          # (N*256,)
    p_flat = pred.reshape(-1).astype(np.float32)            # (N*256,)

    # ------------------------------------------------------------------
    # 4. Ridge‑regression normal equation (closed‑form)
    # ------------------------------------------------------------------
    lam = 1e-6                     # small ridge constant for stability
    I   = np.eye(C * 3 * 3, dtype=np.float32)
    XtX = X.T @ X + lam * I
    Xty = X.T @ y_flat

    w_out_flat = np.linalg.solve(XtX, Xty).astype(np.float32)   # (576,)
    new_out_weight = w_out_flat.reshape(out_weight.shape)       # (1,64,3,3)

    # Bias = mean residual after the weight fit
    resid = y_flat - X @ w_out_flat
    new_out_bias = np.mean(resid).astype(np.float32)

    # Clip to the allowed finite‑float32 range |value| <= 1000
    new_out_weight = np.clip(new_out_weight, -1000, 1000)
    new_out_bias   = np.clip(new_out_bias,   -1000, 1000)

    # ------------------------------------------------------------------
    # 5. Assemble the updated weight dictionary
    # ------------------------------------------------------------------
    w_new = {k: v for k, v in w.items()}          # shallow copy of keys/values
    w_new["out.weight"] = new_out_weight
    w_new["out.bias"]   = new_out_bias

    # ------------------------------------------------------------------
    # 6. Diagnostics: MSE before/after and weight‑change magnitude
    # ------------------------------------------------------------------
    y_hat = X @ w_out_flat + new_out_bias          # predicted epsilon with new weights
    mse_before = float(np.mean((p_flat - y_flat) ** 2))
    mse_after  = float(np.mean((y_hat   - y_flat) ** 2))
    w_change   = float(np.linalg.norm(
        new_out_weight.astype(np.float64) - out_weight.astype(np.float64)))

    diagnostics = {
        "mse_before": mse_before,
        "mse_after":  mse_after,
        "weight_change": w_change
    }

    return {"weights": w_new, "diagnostics": diagnostics}
