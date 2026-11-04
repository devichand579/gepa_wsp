import numpy as np

def edit(w, f):
    """
    Closed‑form ridge‑regression update of the DDPM output convolution
    using the final‑layer features and the target epsilon from the feedback batch.
    Only `out.weight` and `out.bias` are modified; everything else is preserved.
    """
    # ------------------------------------------------------------------
    # 1. Pull the batch‑specific arrays that are permitted.
    # ------------------------------------------------------------------
    batch_ids   = f["batch.ids"]          # shape (128,), int64
    batch_y     = f["batch.y"]            # shape (128, 1, 16, 16), float32
    # current output weight (flattened) and bias
    w_cur_flat  = w["out.weight"].reshape(-1).astype(np.float32)   # (576,)
    b_cur       = float(w["out.bias"].item())                     # scalar

    # ------------------------------------------------------------------
    # 2. Feature maps that feed the output conv (already for the batch).
    # ------------------------------------------------------------------
    feat_in = f["layer.out.input0"]      # shape (128, 64, 16, 16), float32

    # ------------------------------------------------------------------
    # 3. Build the design matrix X (rows = example × spatial position,
    #    columns = 576 output‑weight entries) and the target vector y.
    # ------------------------------------------------------------------
    n_examples = feat_in.shape[0]        # 128
    n_pos      = 16 * 16                 # 256 spatial positions
    n_weights  = 576                     # 1 * 64 * 3 * 3

    # Zero‑pad spatial dims so that “same” conv uses the same indexing as
    # the original network (padding=1 → we add 1 pixel on each side).
    feat_pad = np.pad(feat_in,
                     ((0, 0), (0, 0), (1, 1), (1, 1)),
                     mode="constant",
                     constant_values=0)   # (128, 64, 18, 18)

    X = np.empty((n_examples * n_pos, n_weights), dtype=np.float32)
    # Loop over examples and spatial positions – 32768 iterations, trivial.
    for i in range(n_examples):
        for h in range(16):
            for w in range(16):
                # extract the 3×3 receptive field over all 64 channels
                patch = feat_pad[i, :, h:h+3, w:w+3].ravel()   # (576,)
                X[i * n_pos + h * 16 + w] = patch

    # Target epsilon values flattened to the same order
    y_vec = batch_y.reshape(-1).astype(np.float32)   # (32768,)

    # ------------------------------------------------------------------
    # 4. Closed‑form ridge solution  (λ = 1e‑6 keeps the system well‑conditioned)
    # ------------------------------------------------------------------
    lam = 1e-6
    XtX = X.T @ X                                          # (576,576)
    Xty = X.T @ y_vec                                      # (576,)

    w_new_flat = np.linalg.solve(XtX + lam * np.eye(n_weights), Xty)
    w_new      = w_new_flat.reshape(1, 64, 3, 3).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Compute the predicted epsilon with the new weight (no bias yet)
    # ------------------------------------------------------------------
    out_pred_flat = X @ w_new_flat          # (32768,)
    mse_after   = float(np.mean((out_pred_flat - y_vec) ** 2))

    # ------------------------------------------------------------------
    # 6. Set the bias to centre the residual
    # ------------------------------------------------------------------
    bias_new = float(np.mean(y_vec - out_pred_flat))

    # ------------------------------------------------------------------
    # 7. Update the weight dictionary, clip to safe range, collect diagnostics
    # ------------------------------------------------------------------
    w["out.weight"] = w_new
    w["out.bias"]   = np.float32(bias_new)

    # optional clipping to respect the “absolute value ≤ 1000” rule
    w["out.weight"] = np.clip(w["out.weight"], -1000, 1000)
    w["out.bias"]   = np.clip(w["out.bias"],   -1000, 1000)

    diagnostics = {
        "mse_before":  float(np.mean((X @ w_cur_flat - y_vec) ** 2)),
        "mse_after":   mse_after,
        "bias_new":    bias_new,
        "weight_norm": float(np.linalg.norm(w_new)),
        "n_examples":  n_examples,
        "n_positions": n_pos,
    }

    return {"weights": w, "diagnostics": diagnostics}
