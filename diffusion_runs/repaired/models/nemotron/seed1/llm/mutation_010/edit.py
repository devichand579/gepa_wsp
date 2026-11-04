import numpy as np

def edit(w, f):
    """
    Closed‑form edit of the DDPM epsilon‑prediction network.
    Only the output (last) convolutional layer is re‑fit via ordinary least squares
    using the provided final‑hidden features and target ε.
    All other weights are copied unchanged.
    Returns the updated weight dict and a small diagnostics dict.
    """
    # ----- feedback arrays we need -----
    features = f['features']          # (N, 64, 16, 16)
    y_target = f['y']                # (N, 1, 16, 16)  target Gaussian ε
    # prediction = f['prediction']   # not needed for the fit

    N = features.shape[0]            # should be 2000

    # Pad features with zeros so that a conv‑out with padding=1 works as before
    padded = np.pad(features, ((0, 0), (1, 1), (1, 1)),
                    mode='constant', constant_values=0.0)
    # padded shape: (N, 66, 66)

    # ------------------------------------------------------------------
    # Build design matrix X and target vector y_vec
    #   For every example n and every output pixel (i,j) we take the 3×3×64
    #   patch, flatten it to a 576‑dim vector, and pair it with the scalar
    #   target ε at that spatial location.
    # ------------------------------------------------------------------
    X_rows = []
    y_vals = []
    for n in range(N):
        ex_pad = padded[n]                # (66, 66)
        for i in range(16):
            for j in range(16):
                patch = ex_pad[i:i+3, j:j+3]     # (64, 3, 3)
                X_rows.append(patch.ravel())     # 576 elements
                y_vals.append(y_target[n, 0, i, j])

    X = np.array(X_rows, dtype=np.float64)          # (N*256, 576)
    y_vec = np.array(y_vals, dtype=np.float64)      # (N*256,)

    # ------------------------------------------------------------------
    # Augment X with a column of ones so the fit also learns out.bias
    # ------------------------------------------------------------------
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
    coeff, _, _, _ = np.linalg.lstsq(X_aug, y_vec, rcond=None)
    # coeff length = 576 + 1 ; first 576 entries are the weight tensor,
    # the last entry is the bias.
    out_weight_raw = coeff[:576].reshape((1, 64, 3, 3))   # (1,64,3,3)
    out_bias_raw   = coeff[576]                          # scalar

    # Clip to finite float32 and keep |value| ≤ 1000
    out_weight = np.clip(out_weight_raw.astype(np.float32), -1000, 1000)
    out_bias   = np.clip(np.float32(out_bias_raw), -1000, 1000)

    # ------------------------------------------------------------------
    # Assemble the new weight dictionary – copy everything unchanged
    # ------------------------------------------------------------------
    w_new = {k: v.copy() for k, v in w.items()}
    w_new['out.weight'] = out_weight
    w_new['out.bias']   = out_bias.reshape(1)           # shape (1,)

    # ------------------------------------------------------------------
    # Diagnostics (tiny, JSON‑compatible)
    # ------------------------------------------------------------------
    diagnostics = {
        "weight_change_norm": float(
            np.linalg.norm(out_weight.astype(np.float64) -
                           w['out.weight'].astype(np.float64).astype(np.float64))
        ),
        "mse_before": float(np.mean((f['prediction'] - f['y'])**2))
    }

    return {"weights": w_new, "diagnostics": diagnostics}
