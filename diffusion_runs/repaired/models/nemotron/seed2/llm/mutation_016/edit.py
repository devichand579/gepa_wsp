import numpy as np

def edit(w, f):
    """
    Closed‑form bias correction for the DDPM epsilon‑prediction network.
    
    Parameters
    ----------
    w : dict
        Full current parameter arrays (contains ``out.bias`` of shape (1,)).
    f : dict
        Permitted training feedback arrays:
            - ``y``     : target Gaussian epsilon, shape (2000, 1, 16, 16)
            - ``prediction`` : current epsilon prediction, shape (2000, 1, 16, 16)
    
    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        Updated weights (only ``out.bias`` changed) and a tiny statistics dict.
    """
    # --- pull the permitted feedback arrays ---
    y       = f["feedback_arrays"]["y"]          # shape (2000, 1, 16, 16), float32
    pred    = f["feedback_arrays"]["prediction"]# shape (2000, 1, 16, 16), float32

    # --- compute the mean signed error over the whole batch ---
    e       = pred - y                          # element‑wise error
    mean_err = np.mean(e).astype(np.float32)   # scalar, float32

    # --- closed‑form bias update (no gradient, no iterative optimizer) ---
    w["out.bias"] = (w["out.bias"] - np.float32(mean_err)).astype(np.float32)

    # --- diagnostics (MSE before/after the bias shift) ---
    mse_before = np.mean(e ** 2).astype(np.float32)

    # new prediction after the bias shift (same forward pass, just bias changed)
    new_pred = pred - np.float32(mean_err)
    mse_after  = np.mean((new_pred - y) ** 2).astype(np.float32)

    diagnostics = {
        "mse_before": float(mse_before),
        "mse_after" : float(mse_after),
        "mean_error": float(mean_err)
    }

    return {"weights": w, "diagnostics": diagnostics}
