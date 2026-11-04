import numpy as np

def edit(w, f):
    """
    One‑shot feedback‑assisted weight update.
    
    Parameters
    ----------
    w : dict
        Current full‑precision parameters (keys: conv1.weight, conv1.bias,
        conv2.weight, conv2.bias, conv3.weight, conv3.bias, out.weight,
        out.bias). Shapes are preserved exactly as in the original network.
    f : dict
        Permitted feedback arrays:
        - features: final hidden feature maps, shape (N=2000, 64, 16, 16)
        - y: target Gaussian epsilon, shape (N=2000, 1, 16, 16)
        - alphabar: 32‑step schedule coefficients (unused here)
        - plus other arrays that are kept untouched.
    
    Returns
    -------
    w : dict
        Updated weights (only out.weight and out.bias change). All other
        tensors are left unchanged, shapes and dtypes preserved.
    diagnostics : dict
        Small statistics from the closed‑form fit.
    """
    # ------------------------------------------------------------------
    # 1. Pull the permitted arrays
    # ------------------------------------------------------------------
    features = f["features"]                # (2000, 64, 16, 16)
    y        = f["y"]                       # (2000, 1, 16, 16)

    # ------------------------------------------------------------------
    # 2. Global average pooling over spatial dimensions
    # ------------------------------------------------------------------
    # features_flat: (2000, 64)   – mean over H×W for each channel
    # y_flat       : (2000,)       – mean over H×W for each example
    features_flat = features.mean(axis=(2, 3))   # (2000, 64)
    y_flat        = y.mean(axis=(1, 2))        # (2000,)

    # ------------------------------------------------------------------
    # 3. Closed‑form ridge regression: w = (XᵀX + λI)⁻¹ Xᵀ y
    # ------------------------------------------------------------------
    n_examples, n_channels = features_flat.shape   # 2000, 64
    lam = 1e-6                                     # tiny ridge constant
    XtX = features_flat.T @ features_flat + lam * np.eye(n_channels)   # (64,64)
    Xty = features_flat.T @ y_flat                # (64,)

    # Solve the normal equations directly (no iterative optimizer)
    w_out = np.linalg.solve(XtX, Xty)            # (64,) – optimal linear weights

    # ------------------------------------------------------------------
    # 4. Compute the optimal bias (mean residual after applying the weights)
    # ------------------------------------------------------------------
    pred_y = features_flat @ w_out                # (2000,)
    out_bias = float(np.mean(y_flat - pred_y))

    # ------------------------------------------------------------------
    # 5. Map the 64‑dim weight vector into the required out.weight shape
    # ------------------------------------------------------------------
    # Original out.weight shape: (1, 64, 3, 3). We place the learned vector
    # into the central element of the 3×3 kernel (effective 1×1 convolution)
    # and initialise the rest of the kernel to zero. This respects the exact
    # tensor shape and dtype.
    out_weight_new = np.zeros_like(w["out.weight"], dtype=np.float32)
    # centre index for a 3×3 kernel is (1,1)
    out_weight_new[0, :, 1, 1] = w_out.astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Assemble the updated weight dictionary
    # ------------------------------------------------------------------
    w_new = dict(w)                     # shallow copy of all original keys
    w_new["out.weight"] = out_weight_new
    w_new["out.bias"] = np.float32(out_bias)

    # ------------------------------------------------------------------
    # 7. Diagnostics (all values are finite float32)
    # ------------------------------------------------------------------
    reg_mse = float(np.mean((pred_y - y_flat) ** 2))   # training MSE of the fit
    diag = {
        "out_weight_center_norm": float(np.linalg.norm(w_out)),
        "out_bias": out_bias,
        "regr_mse": reg_mse,
        "n_train": n_examples,
        "n_features": n_channels,
    }

    return {"weights": w_new, "diagnostics": diag}
