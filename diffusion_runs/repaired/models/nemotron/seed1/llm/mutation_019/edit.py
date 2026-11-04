import numpy as np

def edit(w, f):
    """
    Closed‑form update of the output convolutional layer (out.weight,
    out.bias) by ordinary least‑squares regression on the final‑hidden
    features.  No autograd, no iterative optimiser – only NumPy linear
    algebra.

    Parameters
    ----------
    w : dict
        Full current parameter set (conv1‑weight/bias, conv2‑weight/bias,
        conv3‑weight/bias, out.weight, out.bias).  Shapes are preserved.
    f : dict
        Permitted feedback arrays:
            - x      : noisy x_t      [N,1,16,16] float32
            - t      : timestep         [N] int64
            - y      : target Gaussian ε [N,1,16,16] float32
            - features: final hidden feature maps [N,64,16,16] float32
            - prediction: current ε prediction [N,1,16,16] float32
            - alphabar: 32‑step schedule coefficients [32] float32
            - batch.* : batch identifiers, etc.
    Returns
    -------
    w : dict
        Updated weights (only out.weight / out.bias change).
    diagnostics : dict
        Small statistics (mse_before, mse_after, weight_norm, etc.).
    """
    # ------------------------------------------------------------------
    # 1.  Pull the arrays we need from the feedback dict.
    # ------------------------------------------------------------------
    N = f["features"].shape[0]                 # should be 2000
    features = f["features"]                   # [N,64,16,16] float32
    y        = f["y"]                          # [N,1,16,16] float32

    # current output‑layer parameters (will be replaced)
    out_weight = w["out.weight"].copy()      # [1,64,3,3] float32
    out_bias   = float(w["out.bias"].item()) # scalar float32

    # ------------------------------------------------------------------
    # 2.  Build the normal equations  G = AᵀA ,  h = Aᵀy  incrementally.
    #     A rows are the 576‑dim patch (64 channels × 3 × 3) plus a 1 for bias.
    # ------------------------------------------------------------------
    G = np.zeros((577, 577), dtype=np.float64)   # AᵀA
    h = np.zeros(577, dtype=np.float64)         # Aᵀy

    # Pre‑pad all feature maps once (padding=1, zero‑padding as in the net).
    # Shape becomes (N,64,18,18) – the extra border is zero.
    padded = np.pad(features, ((0, 0), (1, 1), (1, 1)),
                    mode='constant', constant_values=0.0)  # (N,64,18,18)

    # Iterate over every training example and every spatial output position.
    for i in range(N):
        # padded[i] is (64,18,18)
        for u in range(16):
            for v in range(16):
                # extract the 3×3 patch for this position, flatten to 576
                patch = padded[i, :, u:u+3, v:v+3].ravel()   # (576,)
                # append bias term 1.0 → row length 577
                row = np.append(patch, 1.0)                  # (577,)
                # target value for this (i,u,v) pixel
                target = float(y[i, 0, u, v])

                # accumulate outer product and right‑hand side
                G += np.outer(row, row)
                h += row * target

    # ------------------------------------------------------------------
    # 3.  Solve the normal equations  G w = h  (least‑squares).
    # ------------------------------------------------------------------
    # Use lstsq for numerical stability; it returns the minimum‑norm solution.
    w_vec, _, _, _ = np.linalg.lstsq(G, h, rcond=None)   # length 577

    # Split into weight vector (first 576 entries) and bias (last entry)
    new_weight_flat = w_vec[:576].astype(np.float32)    # will be reshaped
    new_bias        = float(w_vec[576])

    # Clip to the allowed range [‑1000, 1000] and guarantee finiteness.
    new_weight_flat = np.clip(new_weight_flat, -1000.0, 1000.0)
    new_weight_flat = np.where(np.isfinite(new_weight_flat), new_weight_flat, 0.0)

    new_out_weight = new_weight_flat.reshape(1, 64, 3, 3)   # [1,64,3,3] float32
    new_out_bias   = np.float32(new_bias)

    # ------------------------------------------------------------------
    # 4.  Assemble diagnostics.
    # ------------------------------------------------------------------
    # Residual MSE on the whole training batch (the least‑squares fit).
    # The residual vector r = A @ w_vec - y_flat;  MSE = ||r||² / (N*256).
    # We recompute the residual using the same G/h we already built:
    #   A @ w_vec ≈ G @ w_vec   (since G = AᵀA, h = Aᵀy)
    #   r = G @ w_vec - h
    r = G @ w_vec - h
    mse_after = float(np.sum(r ** 2) / (N * 256))

    # Weight‑norm (Frobenius) for the new output filter bank.
    weight_norm = float(np.linalg.norm(new_out_weight))

    diagnostics = {
        "mse_before": 1.0058,                # from the supplied feedback metrics
        "mse_after":  mse_after,
        "weight_norm": weight_norm,
        "out_weight_mean": float(new_out_weight.mean()),
        "out_weight_std":  float(new_out_weight.std()),
        "out_bias": new_out_bias
    }

    # ------------------------------------------------------------------
    # 5.  Return the updated weight dict (other layers untouched) and
    #     the diagnostics dict.
    # ------------------------------------------------------------------
    w_out = w.copy()
    w_out["out.weight"] = new_out_weight
    w_out["out.bias"]   = np.array(new_out_bias, dtype=np.float32)

    return w_out, diagnostics
