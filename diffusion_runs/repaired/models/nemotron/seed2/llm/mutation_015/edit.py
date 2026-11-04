import numpy as np

def edit(w, f):
    """
    Closed‑form bias correction for the DDPM epsilon‑prediction head.
    
    Parameters
    ----------
    w : dict
        Current network parameters (all conv weights + biases). Keys and shapes
        must be preserved exactly.
    f : dict
        Permitted feedback arrays only. Guaranteed keys:
            - f['prediction']   : float32 [N,1,16,16] current epsilon output
            - f['y']            : float32 [N,1,16,16] target Gaussian epsilon
    
    Returns
    -------
    w : dict
        Updated weights with the output bias shifted by the negative mean
        error. All other tensors are unchanged.
    diagnostics : dict
        Small statistics about the edit (mean error before/after, MSE change).
    """
    # ----- copy weights preserving keys/shapes -----
    w_new = {k: v.copy() for k, v in w.items()}

    # ----- permitted feedback arrays -----
    pred = f['prediction']          # [N,1,16,16]
    target = f['y']                 # [N,1,16,16]

    # ----- compute mean signed error over the whole batch -----
    err = pred - target             # [N,1,16,16]
    mean_err = np.mean(err)         # scalar float32

    # ----- update only the output bias (closed‑form fit) -----
    # out.bias has shape [1]; broadcasting adds it to every pixel.
    w_new['out.bias'] = w['out.bias'] - mean_err

    # ----- diagnostics -----
    # MSE before and after (optional, computed on the same batch)
    mse_before = np.mean((pred - target) ** 2)
    mse_after  = np.mean((pred - mean_err - target) ** 2)

    diag = {
        "mean_error": float(mean_err),
        "mse_before": float(mse_before),
        "mse_after":  float(mse_after),
        "mse_reduction": float(mse_before - mse_after),
    }

    return {"weights": w_new, "diagnostics": diag}
