import numpy as np

def edit(w, f):
    """
    Closed‑form edit that shifts the output bias by the mean residual
    between the target epsilon (y) and the current prediction (prediction).
    No autograd, back‑prop, or iterative optimiser is used.
    
    Parameters
    ----------
    w : dict
        Current network parameters (full‑precision).  Keys include
        'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
        'conv3.weight', 'conv3.bias', 'out.weight', 'out.bias'.
    f : dict
        Feedback arrays under the key ``feedback_arrays``.  Required
        arrays: ``y`` (target epsilon), ``prediction`` (current epsilon
        prediction), and any others may be ignored.
    
    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        Updated weight dictionary and a small statistics dictionary.
    """
    fb = f["feedback_arrays"]                     # permitted feedback only
    y   = fb["y"].astype(np.float32)              # shape (2000,1,16,16)
    pred= fb["prediction"].astype(np.float32)    # shape (2000,1,16,16)

    # ------- closed‑form bias shift -------
    res   = y - pred                               # residual, same shape
    bias_shift = float(np.mean(res))               # scalar, may be positive or negative

    # update the output bias (single scalar addition)
    w["out.bias"] = np.clip(
        w["out.bias"].astype(np.float32) + bias_shift,
        -1000.0, 1000.0
    ).astype(np.float32)

    # ------- diagnostics (MSE before / after) -------
    mse_before = float(np.mean((y - pred) ** 2))
    new_pred   = pred + bias_shift                 # bias‑shifted prediction
    mse_after  = float(np.mean((y - new_pred) ** 2))

    diagnostics = {
        "bias_shift": bias_shift,
        "mse_before": mse_before,
        "mse_after":  mse_after
    }

    return {"weights": w, "diagnostics": diagnostics}
