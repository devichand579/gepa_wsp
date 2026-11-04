import numpy as np

def edit(w, f):
    """
    Update the network's output‑layer weights (and bias) with a single
    closed‑form least‑squares step that minimises the per‑pixel MSE between
    the current ε‑prediction and the target Gaussian ε, using the provided
    training feedback arrays.

    Parameters
    ----------
    w : dict
        Full current parameter set (all conv‑weights, biases, etc.).
    f : dict
        Permitted feedback arrays: x, t, y, clean, features, prediction,
        alphabar, batch.ids, batch.x/t/y, generated.samples, layer traces.

    Returns
    -------
    {"weights": w, "diagnostics": small stats dict}
        The updated weight dictionary and a tiny diagnostics dictionary.
    """
    # ------------------------------------------------------------------
    # 1.  Pull the relevant feedback arrays for the 128‑example batch.
    # ------------------------------------------------------------------
    y_full      = f["y"]                # shape (2000, 1, 16, 16)
    pred_full   = f["prediction"]       # shape (2000, 1, 16, 16)
    feat_full   = f["features"]         # shape (2000, 64, 16, 16)
    batch_ids   = f["batch.ids"]        # shape (128,), int64

    y_batch     = y_full[batch_ids]     # (128, 1, 16, 16)
    pred_batch  = pred_full[batch_ids]  # (128, 1, 16, 16)
    feat_batch  = feat_full[batch_ids]  # (128, 64, 16, 16)

    N = y_batch.shape[0]                # 128
    H = W = 16
    total_pixels = N * H * W             # 32 768

    # ------------------------------------------------------------------
    # 2.  Form the residual vector r = target – current prediction.
    # ------------------------------------------------------------------
    y_flat = y_batch.reshape(-1).astype(np.float32)      # (total_pixels,)
    pred_flat = pred_batch.reshape(-1).astype(np.float32)
    r = y_flat - pred_flat                               # (total_pixels,)

    # ------------------------------------------------------------------
    # 3.  Build the design matrix A2.
    #    Each row corresponds to one spatial position (h,w) of one example
    #    and contains the flattened 3×3 receptive‑field patch from the
    #    final‑hidden feature map (64 channels) plus a constant 1 for the bias.
    # ------------------------------------------------------------------
    A2_rows = []
    for i in range(N):
        feat = feat_batch[i]                     # (64, 16, 16)
        # Pad with one zero pixel on each side so that a 3×3 window
        # centred at every original (h,w) position can be extracted.
        feat_pad = np.pad(feat, ((0, 0), (1, 1), (1, 1)),
                          mode='constant', constant_values=0.0)  # (64, 18, 18)

        for h in range(H):
            for w in range(W):
                patch = feat_pad[:, h:h+3, w:w+3].reshape(-1)   # (64*3*3 = 576,)
                row = np.concatenate([patch, np.float32(1.0)])  # (577,)
                A2_rows.append(row)

    A2 = np.array(A2_rows, dtype=np.float32)   # (32 768, 577)

    # ------------------------------------------------------------------
    # 4.  Closed‑form least‑squares solution for weight + bias update.
    # ------------------------------------------------------------------
    delta, _, _, _ = np.linalg.lstsq(A2, r, rcond=None)
    # delta shape (577,)

    delta_w_flat = delta[:576]          # (576,)
    delta_bias   = delta[576]           # scalar

    # Reshape weight delta to the output‑layer shape (1, 64, 3, 3)
    delta_w = delta_w_flat.reshape(w["out.weight"].shape).astype(np.float32)

    # ------------------------------------------------------------------
    # 5.  Apply the update, clip to finite values ≤ 1000, and return.
    # ------------------------------------------------------------------
    w["out.weight"] = (w["out.weight"].astype(np.float32) + delta_w)
    w["out.bias"]   = (w["out.bias"].astype(np.float32) + delta_bias)

    for k in ("out.weight", "out.bias"):
        np.clip(w[k], -1000, 1000, out=w[k])
        w[k] = np.where(np.isfinite(w[k]), w[k], 0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 6.  Diagnostics (tiny statistics, no extra forward pass).
    # ------------------------------------------------------------------
    diagnostics = {
        "delta_weight_norm": float(np.linalg.norm(delta_w_flat)),
        "delta_bias": float(delta_bias),
        "residual_norm_before": float(np.linalg.norm(r)),
    }

    return {"weights": w, "diagnostics": diagnostics}
