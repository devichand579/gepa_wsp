import numpy as np

def edit(w, f):
    """
    Closed‑form bias correction for the DDPM epsilon‑prediction network.
    
    Parameters
    ----------
    w : dict
        Current network parameters (conv1/2/3 weights/biases, out.weight, out.bias).
    f : dict
        Feedback arrays: x, t, y, clean, features, prediction, alphabar, etc.
    
    Returns
    -------
    w : dict
        Updated weights (only `out.bias` changed).
    diagnostics : dict
        MSE before and after the edit, mean error, etc.
    """
    # ------------------------------------------------------------------
    # 1. Pull the relevant arrays from the feedback dict
    # ------------------------------------------------------------------
    prediction = f['prediction']          # shape (N, 1, 16, 16), float32
    target     = f['y']                   # shape (N, 1, 16, 16), float32
    # squeeze the channel dimension for easier handling
    pred_sq   = prediction.squeeze(axis=1)   # (N, 16, 16)
    tgt_sq    = target.squeeze(axis=1)       # (N, 16, 16)

    # ------------------------------------------------------------------
    # 2. Compute the per‑pixel mean signed error
    # ------------------------------------------------------------------
    error = pred_sq - tgt_sq                 # (N, 16, 16)
    mean_error = np.mean(error)              # scalar float32

    # ------------------------------------------------------------------
    # 3. Closed‑form bias update
    #    New bias = old_bias - mean_error
    #    (subtracting the mean error shifts every prediction toward the target)
    # ------------------------------------------------------------------
    old_bias = float(w['out.bias'].item())   # current bias (scalar)
    new_bias = old_bias - mean_error
    w = {k: v.copy() for k, v in w.items()}  # preserve all other weights
    w['out.bias'] = np.array([new_bias], dtype=np.float32)

    # ------------------------------------------------------------------
    # 4. Evaluate the effect of the edit (still using the *original*
    #    predictions, but with the bias shifted).  This is just a
    #    diagnostic; the edit itself never runs during sampling.
    # ------------------------------------------------------------------
    # Shifted prediction = original prediction - mean_error
    pred_shifted = pred_sq - mean_error     # (N, 16, 16)
    mse_before = np.mean((pred_sq - tgt_sq) ** 2)
    mse_after  = np.mean((pred_shifted - tgt_sq) ** 2)

    diagnostics = {
        "mse_before": float(mse_before),
        "mse_after":  float(mse_after),
        "mean_error": float(mean_error),
        "bias_shift": float(new_bias - old_bias),   # = -mean_error
    }

    return w, diagnostics
