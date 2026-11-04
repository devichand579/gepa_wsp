import numpy as np

def edit(w, f):
    """
    Closed‑form least‑squares update of the output layer of a DDPM epsilon‑prediction network.
    Only `out.weight` and `out.bias` are modified; all other parameters are left unchanged.
    """
    # ------------------------------------------------------------------
    # 1.  Gather feedback arrays (shapes are fixed by the problem statement)
    # ------------------------------------------------------------------
    features = f["features"]            # [N, 64, 16, 16], N = 2000
    y        = f["y"]                   # [N, 1, 16, 16]
    N        = features.shape[0]        # 2000
    H, W     = 16, 16                   # spatial size

    # ------------------------------------------------------------------
    # 2.  Build the normal‑equation accumulators (576 × 576 and 576 vector)
    # ------------------------------------------------------------------
    K = 64 * 3 * 3          # number of weights in the output conv = 576
    AtA = np.zeros((K, K), dtype=np.float32)
    Aty = np.zeros(K, dtype=np.float32)
    total = N * H * W       # 512 000 samples

    # ------------------------------------------------------------------
    # 3.  First pass: accumulate AᵀA  and  Aᵀy  without materialising A
    # ------------------------------------------------------------------
    for n in range(N):
        feat = features[n]                     # (64, 16, 16)
        # zero‑pad by 1 pixel so that conv‑‘same’ works for every output pixel
        pad_feat = np.pad(feat, ((0, 0), (1, 1), (1, 1)),
                          mode="constant", constant_values=0.0)  # (64, 18, 18)

        for i in range(H):
            for j in range(W):
                # extract the 3×3×64 patch for output pixel (i, j) and flatten
                patch = pad_feat[:, i:i+3, j:j+3].ravel()      # (576,)
                yval = y[n, 0, i, j]                         # scalar target epsilon

                # accumulate outer product and dot product
                AtA += np.outer(patch, patch).astype(np.float32)
                Aty += (patch * yval).astype(np.float32)

    # ------------------------------------------------------------------
    # 4.  Solve for the optimal weight vector  w  (closed‑form fit)
    # ------------------------------------------------------------------
    try:
        w_flat = np.linalg.solve(AtA, Aty)       # exact solution if AtA invertible
    except np.linalg.LinAlgError:
        # fallback: minimum‑norm least‑squares
        w_flat, _, _, _ = np.linalg.lstsq(AtA, Aty, rcond=None)

    # keep values finite and within the allowed range
    w_flat = np.clip(w_flat, -1000, 1000).astype(np.float32)

    # reshape to the original convolution weight tensor
    new_out_weight = w_flat.reshape(1, 64, 3, 3)   # [1, 64, 3, 3]

    # ------------------------------------------------------------------
    # 5.  Determine the optimal bias (mean of residuals)
    # ------------------------------------------------------------------
    sum_y = 0.0
    sum_y_hat = 0.0
    for n in range(N):
        feat = features[n]
        pad_feat = np.pad(feat, ((0, 0), (1, 1), (1, 1)),
                          mode="constant", constant_values=0.0)
        for i in range(H):
            for j in range(W):
                patch = pad_feat[:, i:i+3, j:j+3].ravel()
                yval = y[n, 0, i, j]
                y_hat = np.dot(patch, w_flat)
                sum_y += yval
                sum_y_hat += y_hat

    b = np.float32((sum_y - sum_y_hat) / total)   # optimal bias
    new_out_bias = np.array([b], dtype=np.float32)   # shape [1]

    # ------------------------------------------------------------------
    # 6.  Update the weight dictionary (preserve all other keys/shapes)
    # ------------------------------------------------------------------
    w["current_weights"]["out.weight"] = new_out_weight
    w["current_weights"]["out.bias"]   = new_out_bias

    # ------------------------------------------------------------------
    # 7.  Diagnostics: compare before/after MSE (closed‑form fit)
    # ------------------------------------------------------------------
    mse_before = float(np.mean((f["prediction"] - f["y"])**2))

    # Minimised MSE after the closed‑form fit:
    #   min_w ||y - A w||² / N_total  =  (yᵀy - wᵀ Aᵀy) / N_total
    #   we already have yᵀy (sum of squares of all target epsilons) and Aᵀy = Aty.
    y_sq_sum = float(np.sum(y.astype(np.float32) ** 2))
    mse_after = float((y_sq_sum - np.dot(w_flat, Aty)) / total)

    diagnostics = {
        "mse_before": mse_before,
        "mse_after_closedform": mse_after,
        "updates": {
            "out.weight_shape": list(new_out_weight.shape),
            "out.bias_scalar": float(b)
        }
    }

    # Return the updated weight dict together with a tiny statistics dict.
    return {"weights": w, "diagnostics": diagnostics}
