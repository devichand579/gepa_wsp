import numpy as np

def edit(w, f):
    """
    Edit the DDPM epsilon‑prediction network by applying a global
    gain‑and‑bias factor to its output layer.  The factors are computed
    in closed‑form from the current prediction and the target epsilon
    (both provided in the feedback arrays).  No autograd, back‑prop,
    or iterative optimiser is used.

    Parameters
    ----------
    w : dict
        Full current parameter set (all convolutional kernels, biases,
        etc.).  Keys follow the convention shown in the task description.
    f : dict
        Feedback arrays only.  Permitted keys include ``prediction``,
        ``y`` (target Gaussian ε), ``x``, ``t``, ``clean``, ``features``,
        ``alphabar``, etc.

    Returns
    -------
    {"weights": w_new, "diagnostics": diag}
        *w_new* has the same keys and shapes as *w*; only the output‑layer
        weights and bias are altered.
        *diag* contains a few statistics for inspection.
    """
    # ------------------------------------------------------------------
    # 1. Pull the quantities we need from the feedback dict.
    # ------------------------------------------------------------------
    pred = f["prediction"]          # shape (N, 1, 16, 16)
    y    = f["y"]                   # shape (N, 1, 16, 16), target ε
    N    = pred.shape[0]            # number of training examples (2000)
    H    = pred.shape[2]            # 16
    W    = pred.shape[3]            # 16
    # ------------------------------------------------------------------
    # 2. Flatten spatial dimensions → (N, H*W) = (N, 256)
    # ------------------------------------------------------------------
    pred_flat = pred.reshape(N, -1)      # (N, 256)
    y_flat    = y.reshape(N, -1)         # (N, 256)

    # ------------------------------------------------------------------
    # 3. Compute global gain (s) and bias (b) via ordinary least squares.
    #    We minimise  Σ_{pixels} ( s * pred_pix + b - y_pix )²  over all
    #    N*256 scalar samples.
    # ------------------------------------------------------------------
    pred_all = pred_flat.ravel()          # length N*256
    y_all    = y_flat.ravel()

    mean_pred = pred_all.mean()
    mean_y    = y_all.mean()
    var_pred  = pred_all.var()            # population variance (ddof=0)
    cov_xy    = np.mean((pred_all - mean_pred) * (y_all - mean_y))

    if var_pred == 0.0:
        s = 0.0
    else:
        s = cov_xy / var_pred

    b = mean_y - s * mean_pred

    # Keep values finite and inside the allowed range [-1000, 1000].
    s = float(np.clip(s, -10.0, 10.0))
    b = float(np.clip(b, -10.0, 10.0))

    # ------------------------------------------------------------------
    # 4. Apply the gain to the output‑layer weights and add the bias.
    #    The output conv has weight shape (1, 64, 3, 3) and bias (1,).
    #    Multiplying the weights by *s* scales the convolution output;
    #    adding *b* to the existing bias shifts the whole activation.
    # ------------------------------------------------------------------
    new_out_weight = (s * w["out.weight"]).astype(np.float32)
    new_out_bias   = (w["out.bias"].astype(np.float32) + b)

    # ------------------------------------------------------------------
    # 5. Copy the original weight dict and overwrite only the output layer.
    # ------------------------------------------------------------------
    w_new = {k: v.copy() for k, v in w.items()}
    w_new["out.weight"] = new_out_weight
    w_new["out.bias"]   = new_out_bias

    # ------------------------------------------------------------------
    # 6. Diagnostics – MSE before and after the (approximate) edit.
    # ------------------------------------------------------------------
    mse_before = np.mean((pred - y) ** 2)
    # Approximate MSE after applying s*pred + b (uses the *current* pred,
    # not the re‑forwarded network, but illustrates the effect).
    mse_after  = np.mean((s * pred + b - y) ** 2)

    diagnostics = {
        "s": s,
        "b": b,
        "global_mse_before": float(mse_before),
        "global_mse_after":  float(mse_after),
        "n_examples": N,
        "pixels_per_example": H * W
    }

    return {"weights": w_new, "diagnostics": diagnostics}
