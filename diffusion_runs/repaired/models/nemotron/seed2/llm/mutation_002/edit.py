import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    """
    Closed‑form weight update for the output layer of a DDPM epsilon predictor.
    
    Uses a single normal‑equation least‑squares step on the final convolutional layer,
    keeping all earlier layers (conv1‑conv3) unchanged so that cached features remain valid.
    
    Parameters
    ----------
    w : dict
        Full current parameter arrays (keys/weights/shapes as supplied).
    f : dict
        Permitted feedback arrays only: features, y, prediction, etc.
    
    Returns
    -------
    {"weights": w, "diagnostics": {...}}
    """
    # ----- extract feedback arrays -----
    features = f["features"]            # [N, 64, 16, 16]   float32
    y        = f["y"]                   # [N, 1, 16, 16]   float32  (target epsilon)
    prediction = f["prediction"]       # [N, 1, 16, 16]   float32 (current epsilon pred)
    
    N = features.shape[0]              # number of training examples (2000)
    
    # ----- output layer parameters -----
    out_w = w["out.weight"]            # [1, 64, 3, 3]    float32
    out_b = w["out.bias"]              # [1]            float32
    
    # ----- reshape targets / predictions to pixel‑wise vectors -----
    y_flat = y.reshape(N, -1)          # [N, 256]
    # prediction already matches y spatial layout
    
    # ----- build design matrix A [N*256, 576] and target vector t -----
    # The output layer performs zero‑padded conv3x3 (pad1) on [N, 64, 16, 16] → [N, 1, 16, 16].
    # We embed this operation by extracting 3×3 patches from zero‑padded features.
    
    # pad feature maps so that a 3×3 extractor at positions 0..15 yields the same as pad1 conv
    pad_w = 1
    features_padded = np.pad(features, 
                             ((0, 0), (0, 0), (pad_w, pad_w), (pad_w, pad_w)), 
                             mode='constant', constant_values=0.)  # [N, 64, 18, 18]
    
    # sliding_window_view gives [N, 64, 16, 16, 3, 3] where
    #   patches[n, c, i, j, k, l] = features_padded[n, c, i+k, j+l]
    patches = sliding_window_view(features_padded, (3, 3), axis=(2, 3))
    
    # Reorder axes so that the last two dimensions are the kernel indices (k,l)
    # and the first three are (n, i, j) → rows = (n, i, j) = N*256 rows, cols = 64*9 = 576
    A = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * 256, out_w.size).astype(np.float64)
    
    t = y_flat.reshape(-1).astype(np.float64)   # [N*256] target values
    
    # ----- closed‑form normal‑equation solve (ridge‑regularized) -----
    lam = 1e-6                     # tiny ridge for numerical stability (permitted closed‑form fitting)
    Gram = A.T @ A + lam * np.eye(out_w.size, dtype=np.float64)   # 576×576
    rhs  = A.T @ t                 # [576]
    w_vec = np.linalg.solve(Gram, rhs).astype(np.float32)       # [576]
    
    # ----- reshape updated weights and compute new bias -----
    new_out_w = w_vec.reshape(1, 64, 3, 3)      # [1, 64, 3, 3]  float32
    
    # bias: mean residual after weight solve (since we did not augment A with a column of ones)
    residual = t - A @ w_vec
    new_out_b = np.mean(residual).astype(np.float32)
    
    # ----- update weights dict, preserving all other keys/shapes -----
    w_out = w.copy()
    w_out["out.weight"] = new_out_w
    w_out["out.bias"]   = new_out_b
    # all other weight keys remain exactly as in the input `w`
    
    # ----- diagnostics -----
    # compute MSE on the training batch before and after (using the same forward logic via A)
    new_pred_flat = (A @ w_vec).reshape(N, 256) + new_out_b
    old_mse = np.mean((prediction.reshape(N, 256) - y_flat) ** 2)
    new_mse = np.mean((new_pred_flat - y_flat) ** 2)
    
    diagnostics = {
        "old_mse":          float(old_mse),
        "new_mse":          float(new_mse),
        "weight_delta_norm": float(np.linalg.norm(new_out_w - out_w)),
        "bias_update":      float(new_out_b - out_b),
        "ridge_lambda":     float(lam)
    }
    
    return {"weights": w_out, "diagnostics": diagnostics}
