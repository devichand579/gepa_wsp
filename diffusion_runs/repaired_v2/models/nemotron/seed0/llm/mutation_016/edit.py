import numpy as np

def edit(w, f):
    """
    Closed‑form adjustment of the output layer to minimise MSE
    between the current epsilon prediction and the target epsilon.

    Parameters
    ----------
    w : dict
        Full current parameter set (contains every weight tensor).
    f : dict
        Permitted feedback arrays (only y, prediction, etc. are needed).

    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        Updated weight dict and a small statistics dictionary.
    """
    # ------------------------------------------------------------------
    # 1. Pull the relevant feedback arrays
    # ------------------------------------------------------------------
    y      = f['y']                # shape (N, 1, 16, 16)   target epsilon
    pred   = f['prediction']       # shape (N, 1, 16, 16)   current epsilon prediction

    # ------------------------------------------------------------------
    # 2. Flatten spatial & channel dimensions for a global linear fit
    # ------------------------------------------------------------------
    y_flat = y.ravel()             # (N*256,)
    pred_flat = pred.ravel()       # (N*256,)

    # ------------------------------------------------------------------
    # 3. Closed‑form scale (a) and bias (b) that minimise
    #    MSE = E[(a * pred + b - y)^2]
    # ------------------------------------------------------------------
    # np.cov returns a 2x2 matrix [[var(pred), cov(pred,y)],
    #                            [cov(y,pred), var(y)]]
    cov = np.cov(pred_flat, y_flat, bias=False)   # sample cov (ddof=1)
    a = cov[0, 1] / cov[0, 0]                     # var(pred) in denominator
    b = np.mean(y_flat) - a * np.mean(pred_flat)

    # keep everything in float32 and clip to reasonable magnitude
    a = np.float32(np.clip(a, -1000, 1000))
    b = np.float32(np.clip(b, -1000, 1000))

    # ------------------------------------------------------------------
    # 4. Diagnostics: MSE before and after the hypothetical adjustment
    # ------------------------------------------------------------------
    mse_before = np.mean((pred - y) ** 2).astype(np.float32)
    mse_after  = np.mean((a * pred_flat + b - y_flat) ** 2).astype(np.float32)

    diag = {
        "a": float(a),
        "b": float(b),
        "mse_before": float(mse_before),
        "mse_after":  float(mse_after)
    }

    # ------------------------------------------------------------------
    # 5. Apply the update *only* to the output layer parameters
    # ------------------------------------------------------------------
    # out.weight shape: (1, 64, 3, 3) – scale every element by a
    w['out.weight'] = (w['out.weight'] * a).astype(np.float32)

    # out.bias shape: (1,) – set to the computed offset b
    w['out.bias'] = np.array([b], dtype=np.float32)

    # ------------------------------------------------------------------
    # 6. Return the modified weight dict together with diagnostics
    # ------------------------------------------------------------------
    return {"weights": w, "diagnostics": diag}
