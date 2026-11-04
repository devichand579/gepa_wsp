def edit(w, f):
    """
    Closed‑form edit of the DDPM epsilon‑prediction network.
    Only the final (output) convolutional layer is re‑fit by ordinary
    least‑squares on the (feature, target) pairs supplied in f.
    All other weights are left unchanged.
    Returns {"weights": w_updated, "diagnostics": {...}}.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # 1. Pick the 128 examples that belong to the paired batch.
    # ------------------------------------------------------------------
    ids = f["batch.ids"]                     # shape (128,)
    features = f["features"][ids]           # (128, 64, 16, 16)
    y      = f["y"][ids]                    # (128, 1, 16, 16)   target ε

    N = features.shape[0]                    # 128
    H, W = 16, 16
    n_pixels = N * H * W                     # 32768

    # ------------------------------------------------------------------
    # 2. Build design matrix A (n_pixels × 576) and target vector t.
    #    Each row corresponds to one spatial location (h,w) and contains
    #    the 3×3 receptive‑field values from the 64 channels (flattened
    #    to length 64·3·3 = 576).  Weight sharing is automatically enforced
    #    because the same 576 parameters are used for every (h,w) and every
    #    batch example.
    # ------------------------------------------------------------------
    A = np.empty((n_pixels, 576), dtype=np.float32)
    t = np.empty(n_pixels, dtype=np.float32)

    for i in range(N):
        feat = features[i]                 # (64, 16, 16)
        # Pad with zeros so that a “same” conv with padding=1 uses the
        # correct 3×3 neighbourhood (zero outside the image border).
        padded = np.pad(feat, ((0, 0), (1, 1), (1, 1)),
                        mode="constant", constant_values=0.0)

        # sliding_window_view gives a view of shape (64, 16, 16, 3, 3)
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (3, 3), axis=(1, 2))

        # Transpose to (h, w, c, kh, kw) and reshape each patch to a vector
        windows = windows.transpose(1, 2, 0, 3, 4)   # (16,16,64,3,3)
        A_row = windows.reshape(H * W, 576)          # (256, 576)

        # Target ε value at every spatial position for this example
        t_row = y[i].reshape(-1)                     # (256,)

        start = i * H * W
        A[start:start + H * W] = A_row
        t[start:start + H * W] = t_row

    # ------------------------------------------------------------------
    # 3. Closed‑form least‑squares solution (minimum‑norm if rank‑deficient).
    # ------------------------------------------------------------------
    w_flat, residuals, rank, s = np.linalg.lstsq(A, t, rcond=None)
    # Reshape to the original convolutional weight tensor (out.weight):
    new_weight = w_flat.reshape(1, 64, 3, 3).astype(np.float32)

    # Bias = mean residual after the optimal linear fit
    bias = float(np.mean(t - A @ w_flat))
    new_bias = np.array(bias, dtype=np.float32)

    # ------------------------------------------------------------------
    # 4. Update the weight dictionary (in‑place on the copy we return).
    # ------------------------------------------------------------------
    w_out = dict(w)                     # shallow copy so original is untouched
    w_out["out.weight"] = new_weight
    w_out["out.bias"] = new_bias

    # ------------------------------------------------------------------
    # 5. Diagnostics: before/after MSE on the training batch, weight norm,
    #    and the learned bias.
    # ------------------------------------------------------------------
    mse_before = float(np.mean((f["prediction"][ids] - f["y"][ids]) ** 2))

    # Predicted ε after the edit (using the same A matrix):
    pred_after = A @ w_flat + bias
    mse_after = float(np.mean((pred_after - t) ** 2))

    diag = {
        "mse_before": mse_before,
        "mse_after": mse_after,
        "weight_norm": float(np.mean(np.abs(new_weight))),
        "bias": bias,
    }

    # ------------------------------------------------------------------
    # 6. Return the updated weights together with a small statistics dict.
    # ------------------------------------------------------------------
    return {"weights": w_out, "diagnostics": diag}
