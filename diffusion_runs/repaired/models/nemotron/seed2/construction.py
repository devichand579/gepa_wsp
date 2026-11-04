import numpy as np

def edit(w, f):
    """
    One‑shot feedback‑assisted weight update.
    Only the output bias is adjusted by the mean signed error
    (target epsilon – current prediction) over the entire training batch.
    All other parameters are left unchanged.
    
    Parameters
    ----------
    w : dict
        Current network parameters (full precision float32).
    f : dict
        Permitted feedback arrays (features, prediction, y, etc.).
    
    Returns
    -------
    {"weights": w_updated, "diagnostics": {...}}
    """
    # ------------------------------------------------------------------
    # 1. Pull the relevant arrays (shapes are fixed by the task)
    # ------------------------------------------------------------------
    out_bias = w["out.bias"]                # shape (1,)
    features = f["features"]                # (N, 64, 16, 16), N = 2000
    prediction = f["prediction"]            # (N, 1, 16, 16)
    target = f["y"]                         # (N, 1, 16, 16)

    N = features.shape[0]                    # number of training examples

    # ------------------------------------------------------------------
    # 2. Compute the per‑pixel error and flatten to a 1‑D vector
    # ------------------------------------------------------------------
    error = target - prediction               # (N, 1, 16, 16)
    error_flat = error.reshape(-1)            # (N * 256,)  → 512 000 entries

    # ------------------------------------------------------------------
    # 3. Closed‑form bias update: new_bias = old_bias + mean(error)
    #    (ridge with λ≈0 is equivalent to the ordinary least‑squares solution
    #     for a constant bias term.)
    # ------------------------------------------------------------------
    mean_err = np.mean(error_flat).astype(np.float32)

    # Preserve finite float32 and the [-1000, 1000] bound
    new_bias = np.clip(out_bias.astype(np.float32) + mean_err, -1000, 1000)

    # ------------------------------------------------------------------
    # 4. Build the updated weight dictionary (deep‑copy only the changed key)
    # ------------------------------------------------------------------
    w_new = dict(w)                # shallow copy of the outer dict
    w_new["out.bias"] = new_bias   # bias shape stays (1,)

    # ------------------------------------------------------------------
    # 5. Diagnostics – small JSON‑compatible statistics
    # ------------------------------------------------------------------
    diagnostics = {
        "bias_update_mean_error": float(mean_err),
        "bias_before": float(out_bias.flatten()[0]),
        "bias_after": float(new_bias),
        "mse_before": float(np.mean(error_flat ** 2)),
        # mse_after could be computed but is not required for the return
    }

    return {"weights": w_new, "diagnostics": diagnostics}
