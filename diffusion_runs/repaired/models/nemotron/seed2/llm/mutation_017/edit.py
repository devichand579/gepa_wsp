import numpy as np

def edit(w, f):
    """
    Closed‑form bias correction for the DDPM epsilon‑prediction network.
    
    Parameters
    ----------
    w : dict
        Current full‑precision weights.  Keys include all convolution
        kernels, biases, etc.  In particular `w['out.bias']` has shape (1,).
    f : dict
        Permitted feedback arrays for the current batch.  Must contain
        at least:
            - `prediction`   : float32 [N, 1, 16, 16]   current epsilon output
            - `y`            : float32 [N, 1, 16, 16]   target Gaussian epsilon
    
    Returns
    -------
    w : dict
        Modified weights with `out.bias` adjusted.
    diagnostics : dict
        Small statistics dictionary (MSE before/after, mean error, bias change).
    """
    # ------------------------------------------------------------------
    # 1. Pull the required arrays; they are guaranteed to exist per spec.
    # ------------------------------------------------------------------
    pred = f['prediction'].astype(np.float32)      # [N,1,16,16]
    targ = f['y'].astype(np.float32)               # [N,1,16,16]

    # ------------------------------------------------------------------
    # 2. Compute the mean signed error across all examples and pixels.
    # ------------------------------------------------------------------
    error = pred - targ                              # [N,1,16,16]
    mean_err = float(np.mean(error))                 # scalar, may be positive/negative

    # ------------------------------------------------------------------
    # 3. Closed‑form bias update.
    #    Subtracting the mean error from the output bias shifts every
    #    prediction by –mean_err, which is the exact minimiser of the
    #    per‑pixel MSE for a constant offset.
    # ------------------------------------------------------------------
    old_bias = w['out.bias'].astype(np.float32)      # shape (1,)
    new_bias = old_bias - mean_err                     # still a scalar bias
    # Clip to avoid runaway values (spec: |value| <= 1000)
    new_bias = np.clip(new_bias, -1000.0, 1000.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Apply the edit, preserving every other parameter unchanged.
    # ------------------------------------------------------------------
    w = {k: (v.copy() if k == 'out.bias' else v) for k, v in w.items()}
    w['out.bias'] = new_bias

    # ------------------------------------------------------------------
    # 5. Diagnostics: MSE before and after the (theoretical) bias shift.
    # ------------------------------------------------------------------
    mse_before = float(np.mean(error ** 2))

    # New prediction after bias correction:
    pred_adj = pred - mean_err
    mse_after  = float(np.mean((pred_adj - targ) ** 2))

    diag = {
        "mse_before": mse_before,
        "mse_after":  mse_after,
        "mean_error": mean_err,
        "bias_change": float(new_bias - old_bias),   # = -mean_err
    }

    return w, diag
