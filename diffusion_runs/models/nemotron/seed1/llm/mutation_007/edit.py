import numpy as np

def edit(w, f):
    """
    Update the network weights using the permitted feedback arrays.
    Only the output convolutional layer (out.weight, out.bias) is modified
    with a single gradient‑descent step on the MSE between the current
    epsilon prediction and the target epsilon y.
    """
    # ------------------------------------------------------------------
    # 1. Unpack current weights and feedback
    # ------------------------------------------------------------------
    out_w = w['out.weight'].astype(np.float32)          # shape (1, 64, 3, 3)
    out_b = w['out.bias'].astype(np.float32)           # shape (1,)

    features = f['features'].astype(np.float32)        # (2000, 64, 16, 16)
    prediction = f['prediction'].astype(np.float32)    # (2000, 1, 16, 16)
    y = f['y'].astype(np.float32)                      # (2000, 1, 16, 16)

    error = prediction - y                             # (2000, 1, 16, 16)
    N = error.shape[0]                                 # 2000

    # ------------------------------------------------------------------
    # 2. Compute gradient of MSE = mean(error²) w.r.t. out.weight/bias
    # ------------------------------------------------------------------
    # Pad features with 1 zero border so that conv2d with padding=1
    # preserves the 16×16 spatial size.
    features_padded = np.pad(features, ((0, 0), (0, 0), (1, 1), (1, 1)),
                             mode='constant', constant_values=0.)
    # shape (2000, 64, 18, 18)

    grad_out_w = np.zeros_like(out_w)   # (1, 64, 3, 3)
    grad_out_b = np.zeros_like(out_b)   # (1,)

    # Loop over the batch; for each sample add the outer‑product of
    # error at each pixel with the corresponding 3×3 patch of features.
    for n in range(N):
        f_n = features_padded[n]                # (64, 18, 18)
        e_n = error[n, 0]                        # (16, 16)

        # accumulate per‑pixel contribution
        for i in range(16):
            for j in range(16):
                patch = f_n[:, i:i+3, j:j+3]   # (64, 3, 3)
                grad_out_w += e_n[i, j] * patch
        grad_out_b += np.sum(e_n)               # sum over all 256 pixels

    grad_out_w /= N
    grad_out_b /= N

    # ------------------------------------------------------------------
    # 3. Apply a small learning‑rate step and clip to finite float32
    # ------------------------------------------------------------------
    lr = 0.01                                          # hand‑picked step size
    out_w_new = (out_w - lr * grad_out_w).clip(-1000, 1000).astype(np.float32)
    out_b_new = (out_b - lr * grad_out_b).clip(-1000, 1000).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Build the updated weight dictionary (preserve all original keys)
    # ------------------------------------------------------------------
    w = w.copy()
    w['out.weight'] = out_w_new
    w['out.bias'] = out_b_new

    # ------------------------------------------------------------------
    # 5. Diagnostics (tiny JSON‑compatible statistics)
    # ------------------------------------------------------------------
    diagnostics = {
        "out_weight_mean": float(np.mean(out_w_new)),
        "out_bias": float(out_b_new[0]),
        "grad_out_weight_mean": float(np.mean(grad_out_w)),
        "mse_before": float(np.mean((prediction - y) ** 2)),
        # mse_after is not computed here to avoid extra forward passes;
        # the learner can run a quick forward if desired.
    }

    return {"weights": w, "diagnostics": diagnostics}
