import numpy as np

def edit(w, f):
    """
    Edit the DDPM epsilon‑prediction network weights using a closed‑form
    bias correction derived from the provided feedback arrays.

    Parameters
    ----------
    w : dict
        Current network parameters.  Keys include the full‑precision weight
        tensors (conv1.weight, conv1.bias, conv2.weight, conv2.bias,
        conv3.weight, conv3.bias, out.weight, out.bias).
    f : dict
        Permitted training feedback arrays:
            - x      : noisy x_t        [2000, 1, 16, 16]
            - t      : timestep           [2000] int64
            - y      : target Gaussian ε  [2000, 1, 16, 16]
            - features : final hidden feature maps [2000, 64, 16, 16]
            - prediction : current ε prediction [2000, 1, 16, 16]
            - alphabar : 32‑schedule coeff [32]
            - batch.* : batched subsets (not needed for this edit)

    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        Updated weight dict (only out.bias may change) and a small
        JSON‑compatible statistics dict.
    """
    # ------------------------------------------------------------------
    # 1. Extract the target ε and the current prediction from the feedback.
    # ------------------------------------------------------------------
    y      = f["y"]                # shape [2000, 1, 16, 16], float32
    pred   = f["prediction"]       # shape [2000, 1, 16, 16], float32

    # ------------------------------------------------------------------
    # 2. Compute the mean element‑wise residual ⟨y − pred⟩.
    #    This is a closed‑form statistic, not a gradient step.
    # ------------------------------------------------------------------
    residual = y - pred            # same shape, float32
    mean_res = float(np.mean(residual))   # scalar, may be positive or negative

    # ------------------------------------------------------------------
    # 3. Update the output bias by the mean residual.
    #    This shifts every predicted ε by the same amount, correcting a
    #    systematic offset without touching the convolutional kernels.
    # ------------------------------------------------------------------
    new_bias = w["out.bias"] + mean_res
    # Preserve finite float32 and keep absolute value ≤ 1000.
    new_bias = np.clip(new_bias, -1000, 1000).astype(np.float32)
    w["out.bias"] = new_bias

    # ------------------------------------------------------------------
    # 4. Assemble diagnostics.
    # ------------------------------------------------------------------
    mse_before = float(np.mean((y - pred) ** 2))   # current MSE (before edit)
    diagnostics = {
        "bias_shift": mean_res,
        "mse_before": mse_before,
        # mse_after is not computed here because the edit does not run
        # a forward pass; the shift is expected to improve (or at least
        # change) the MSE proportionally to the size of the bias correction.
    }

    return {"weights": w, "diagnostics": diagnostics}
