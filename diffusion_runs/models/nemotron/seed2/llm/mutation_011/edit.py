import numpy as np

def edit(w, f):
    """
    Closed‑form weight update for the output layer of a DDPM epsilon‑prediction net.
    Uses only the permitted feedback arrays (x, t, y, features, prediction) and
    NumPy linear algebra – no autograd, no iterative optimiser.

    Parameters
    ----------
    w : dict
        Full current parameter set (weights & biases of all layers).  Shapes are
        preserved exactly.
    f : dict
        Permitted feedback arrays:
            - x   : noisy x_t      [N,1,16,16] float32
            - t   : timestep         [N]      int64
            - y   : target ε        [N,1,16,16] float32
            - features : final hidden feature maps [N,64,16,16] float32
            - prediction : current ε prediction [N,1,16,16] float32
            - (other keys are ignored)

    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        Updated weight dict and a small statistics dictionary.
    """
    # ------------------------------------------------------------------
    # 1. Pull the feedback arrays we need
    # ------------------------------------------------------------------
    y        = f['y']                # target ε, shape (N,1,16,16)
    pred     = f['prediction']      # current ε prediction, shape (N,1,16,16)
    feats    = f['features']        # final hidden feature maps, shape (N,64,16,16)

    # ------------------------------------------------------------------
    # 2. Error signal e = prediction – target
    # ------------------------------------------------------------------
    e = pred - y                     # (N,1,16,16), float32

    # ------------------------------------------------------------------
    # 3. Bias update: subtract the batch‑averaged error
    # ------------------------------------------------------------------
    mean_err = float(np.mean(e))                 # scalar
    w['out.bias'] = w['out.bias'].astype(np.float32) - mean_err
    w['out.bias'] = np.clip(w['out.bias'], -1000, 1000)

    # ------------------------------------------------------------------
    # 4. Weight correction for the output kernel (rank‑one nudge)
    # ------------------------------------------------------------------
    # Flatten spatial dimensions: (N, 256) for error, (N, 64, 256) for features
    e_flat = e.reshape(e.shape[0], -1)            # (N, 256)
    feats_flat = feats.reshape(feats.shape[0],
                               feats.shape[1], -1)   # (N, 64, 256)

    # Broadcast error over the channel dimension → (N, 64, 256)
    product = e_flat[:, np.newaxis, :] * feats_flat

    # Average over batch (axis 0) and spatial positions (axis 2) → (64,) vector
    avg_grad = np.mean(product, axis=(0, 2)).astype(np.float32)   # shape (64,)

    # Mean square of all feature activations (used for closed‑form step size)
    mean_feat_sq = float(np.mean(feats_flat ** 2))
    eps = 1e-6
    factor = 1.0 / (mean_feat_sq + eps)          # scalar step size

    # Build a 3×3 kernel of ones (same weight across kernel positions)
    kernel = np.ones((3, 3), dtype=np.float32)    # (3,3)

    # Reshape avg_grad to (1,64,1,1) and broadcast across the 3×3 kernel
    delta_w = factor * (avg_grad[:, np.newaxis, np.newaxis] * kernel[np.newaxis, :, :])
    # delta_w shape: (1, 64, 3, 3) – matches out.weight

    w['out.weight'] = w['out.weight'].astype(np.float32) - delta_w
    w['out.weight'] = np.clip(w['out.weight'], -1000, 1000)

    # ------------------------------------------------------------------
    # 5. Diagnostics (small JSON‑compatible dict)
    # ------------------------------------------------------------------
    diagnostics = {
        "bias_shift": mean_err,
        "weight_correction_norm": float(np.linalg.norm(delta_w)),
        "mse_before": float(np.mean(e ** 2)),    # current per‑example MSE
        # mse_after would require a forward pass; omitted per spec.
    }

    return {"weights": w, "diagnostics": diagnostics}
