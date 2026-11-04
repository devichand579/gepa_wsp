import numpy as np

def edit(w, f):
    """
    Closed‑form bias‑only edit using the provided feedback arrays.
    No autograd, back‑prop, or iterative optimisation is used.
    Only a single manual normal‑equation / mean‑shift computation.
    """
    # ----- unpack permitted feedback -----
    pred = f['prediction']          # shape (2000, 1, 16, 16), float32
    y    = f['y']                   # shape (2000, 1, 16, 16), float32
    # current weights (we keep everything else untouched)
    out_bias_old = w['out.bias'].astype(np.float32)   # scalar, currently 0.0184

    # ----- compute mean residual -----
    err = pred - y                  # element‑wise error
    mean_err = np.mean(err).astype(np.float32)        # scalar

    # ----- update output bias -----
    # new bias = old bias + mean residual (shifts all predictions by the same amount)
    new_out_bias = out_bias_old + mean_err
    # enforce finite float32 and |value| <= 1000
    new_out_bias = np.clip(new_out_bias, -1000.0, 1000.0).astype(np.float32)

    # ----- assemble updated weight dict -----
    new_weights = {}
    for k, v in w.items():
        if k == 'out.bias':
            new_weights[k] = new_out_bias
        else:
            new_weights[k] = v.astype(np.float32)

    # ----- diagnostics -----
    # MSE before the edit (using the original prediction)
    mse_before = np.mean((pred - y).astype(np.float32) ** 2)

    # MSE after the edit: we shift the prediction by the same bias change
    pred_shifted = pred + (new_out_bias - out_bias_old)
    mse_after = np.mean((pred_shifted - y).astype(np.float32) ** 2)

    diagnostics = {
        "mse_before": float(mse_before),
        "mse_after":  float(mse_after),
        "bias_shift": float(new_out_bias - out_bias_old)
    }

    # return the updated weights together with a tiny statistics dict
    return {"weights": new_weights, "diagnostics": diagnostics}
