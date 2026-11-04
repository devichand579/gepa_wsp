import numpy as np

def edit(w, f):
    """
    Update the output convolutional layer of a DDPM epsilon‑prediction network
    by ordinary least‑squares fitting on a 128‑example batch.

    Parameters
    ----------
    w : dict
        Current network parameters (keys/ shapes as supplied).
    f : dict
        Permitted feedback arrays (x, t, y, features, prediction,
        alphabar, batch.ids, layer traces, generated.samples).

    Returns
    -------
    w : dict
        Updated weights (only ``out.weight`` and ``out.bias`` change).
    {"weights": w, "diagnostics": {...}}   # small statistics dict
    """
    # ------------------------------------------------------------------
    # 1. Pick the 128‑example batch that the framework provides.
    # ------------------------------------------------------------------
    batch_ids = f["batch.ids"].astype(np.int64)          # shape (128,)
    N = batch_ids.size                                 # 128

    pred = f["prediction"][batch_ids]                  # (128,1,16,16)
    targ = f["y"][batch_ids]                           # (128,1,16,16)
    err  = pred - targ                                 # (128,1,16,16)

    # Final‑hidden features for the same examples (used as the “input” to `out`).
    feat = f["features"][batch_ids]                    # (128,64,16,16)
    # The network also stores the explicit out‑layer input trace; use it if present.
    out_in = f["layer.out.input0"][batch_ids]          # (128,64,16,16)

    # ------------------------------------------------------------------
    # 2. Build the design matrix X and target vector t from the output layer.
    #    Each spatial position (h,w) sees a 3×3 patch from the 64‑channel feature map.
    # ------------------------------------------------------------------
    C = 64
    H = W = 16
    n_patches = N * H * W                               # 128 * 256 = 32768

    rows = []
    targets = []

    # Iterate over the batch and spatial locations – 32768 iterations is trivial.
    for i in range(N):
        # Choose either the explicit trace or the features dict; both have same shape.
        feats_i = out_in[i] if out_in is not None else feat[i]   # (64,16,16)
        for h in range(H):
            for w in range(W):
                # Extract the 3×3 patch over all 64 channels → flatten to 576 values.
                patch = feats_i[:, h:h+3, w:w+3]          # (64,3,3)
                rows.append(patch.ravel())                # 576‑dim vector
                targets.append(err[i, 0, h, w])          # scalar ε‑error

    X = np.array(rows, dtype=np.float32)          # (32768, 576)
    t = np.array(targets, dtype=np.float32)       # (32768,)

    # ------------------------------------------------------------------
    # 3. Closed‑form least‑squares solution for the output filter.
    # ------------------------------------------------------------------
    w_opt, residuals, rank, s = np.linalg.lstsq(X, t, rcond=None)
    # w_opt shape (576,) → reshape to (1,64,3,3) as required by the architecture.
    out_weight_new = w_opt.reshape(1, 64, 3, 3).astype(np.float32)

    # Bias: the mean of the residuals after the optimal linear map.
    t_pred = X @ w_opt                               # (32768,) predicted ε values
    out_bias_new = np.float32(np.mean(t - t_pred))

    # ------------------------------------------------------------------
    # 4. Apply the update, keeping every other weight untouched.
    # ------------------------------------------------------------------
    w = {k: v.copy() for k, v in w.items()}        # shallow copy of all keys
    w["out.weight"] = out_weight_new
    w["out.bias"]   = out_bias_new

    # ------------------------------------------------------------------
    # 5. Diagnostics: MSE before/after the linear fit on the batch.
    # ------------------------------------------------------------------
    mse_before = float(np.mean(err ** 2))          # current MSE on the batch
    mse_after  = float(np.mean((t - t_pred) ** 2))  # MSE of the fitted linear model

    diagnostics = {
        "mse_before": mse_before,
        "mse_after":  mse_after,
        "n_patches":  n_patches,
        "rank":       int(rank),
        "residual_norm": float(np.sqrt(np.sum(residuals))) if residuals.size else 0.0,
    }

    return {"weights": w, "diagnostics": diagnostics}
