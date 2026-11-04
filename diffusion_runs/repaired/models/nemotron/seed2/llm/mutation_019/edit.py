import numpy as np

def edit(w, f):
    """
    Closed‑form bias correction for the output layer of a DDPM epsilon‑prediction network.
    The function subtracts the batch‑wise mean error (prediction - y) from out.bias,
    leaving all convolution kernels unchanged.  This minimises per‑pixel MSE in a single
    analytic step and does not use autograd, back‑prop, or any iterative optimizer.
    
    Parameters
    ----------
    w : dict
        Current network parameters.  Keys: conv1.weight, conv1.bias, conv2.weight,
        conv2.bias, conv3.weight, conv3.bias, out.weight, out.bias.  All arrays are
        float32 with the shapes shown in the network description.
    f : dict
        Permitted feedback arrays only.  Keys present: x, t, y, clean, features,
        prediction, alphabar, batch.ids, batch.x/t/y, generated.samples, layer traces.
    
    Returns
    -------
    w : dict
        Updated weights (out.bias only).  All other keys are unchanged.
    {"weights": w, "diagnostics": {...}}   # as a single dict (the function returns w;
                                            # the caller may wrap it).
    """
    # ------------------------------------------------------------------
    # 1. Extract the current prediction and the target epsilon.
    # ------------------------------------------------------------------
    pred = f["prediction"]          # shape (N, 1, 16, 16), float32
    y    = f["y"]                   # shape (N, 1, 16, 16), float32
    N    = pred.shape[0]            # number of training examples (2000)

    # ------------------------------------------------------------------
    # 2. Compute the mean signed error over the whole batch.
    # ------------------------------------------------------------------
    err = pred - y                  # (N, 1, 16, 16)
    mean_err = np.mean(err)         # scalar float32
    
    # ------------------------------------------------------------------
    # 3. Update only the output bias.
    # ------------------------------------------------------------------
    # out.bias has shape (1,).  Subtract the mean error so that the new bias
    # shifts every prediction by exactly –mean_err, which is the optimal constant
    # offset for minimizing MSE given fixed convolution kernels.
    new_bias = w["out.bias"].astype(np.float32) - mean_err
    # Clip to the allowed finite float32 range |value| <= 1000 (defensive).
    new_bias = np.clip(new_bias, -1000.0, 1000.0)
    
    # ------------------------------------------------------------------
    # 4. Preserve every other parameter unchanged.
    # ------------------------------------------------------------------
    w_out = {
        "conv1.weight": w["conv1.weight"].copy(),
        "conv1.bias":   w["conv1.bias"].copy(),
        "conv2.weight": w["conv2.weight"].copy(),
        "conv2.bias":   w["conv2.bias"].copy(),
        "conv3.weight": w["conv3.weight"].copy(),
        "conv3.bias":   w["conv3.bias"].copy(),
        "out.weight":   w["out.weight"].copy(),
        "out.bias":     new_bias,
    }
    
    # ------------------------------------------------------------------
    # 5. Diagnostics (tiny JSON‑compatible dictionary).
    # ------------------------------------------------------------------
    mse_before = np.mean((pred - y) ** 2)          # from feedback metrics
    mse_after  = np.mean((pred - y) ** 2)         # will be recomputed after edit
    # We can quickly evaluate the new MSE by applying the updated bias only:
    #   new_pred = prediction - mean_err  (since only bias changed)
    # But the spec only requires returning w; the caller may compute MSE.
    # Here we report the before‑edit MSE and the amount of bias shift.
    diagnostics = {
        "mse_before": float(mse_before),
        "mean_error": float(mean_err),
        "bias_shift": float(-mean_err),          # new_bias - old_bias
        "update_norm": float(np.linalg.norm(new_bias - w["out.bias"].astype(np.float32))),
        "dtype": "float32",
    }
    
    # The function is expected to return the weight dict; the wrapper in the
    # evaluation harness will pair it with the diagnostics dict.
    return w_out, {"weights": w_out, "diagnostics": diagnostics}
