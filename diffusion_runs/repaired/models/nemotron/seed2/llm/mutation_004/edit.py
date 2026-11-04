def edit(w, f):
    """
    Edit the DDPM epsilon‑prediction network by a single closed‑form
    least‑squares update of the output convolution’s weights and bias.
    Only `w['out.weight']` and `w['out.bias']` are changed; everything
    else is left untouched.
    """
    import numpy as np

    # -----------------------------------------------------------------
    # 1. Gather feedback arrays (all have the same batch size N = 2000)
    # -----------------------------------------------------------------
    N = f["features"].shape[0]                # 2000 examples
    features = f["features"]                  # (N, 64, 16, 16)
    y        = f["y"]                         # (N, 1, 16, 16), target ε

    # -----------------------------------------------------------------
    # 2. Pad the feature maps so that a 3×3 conv with padding=1 is
    #    equivalent to taking the centred 3×3 patch at every output
    #    position (u, v) ∈ {0,…,15}².
    # -----------------------------------------------------------------
    padded = np.pad(features, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant", constant_values=0.0)
    # padded shape: (N, 64, 18, 18)

    # -----------------------------------------------------------------
    # 3. Loop over every output pixel, build the design matrix (including
    #    a bias column), and solve the normal equations with np.linalg.lstsq.
    # -----------------------------------------------------------------
    weight_sum = np.zeros((64, 3, 3), dtype=np.float32)
    bias_sum   = 0.0
    count      = 0

    for u in range(16):
        for v in range(16):
            # Extract the 3×3×64 patch for this (u,v) across all examples
            X = padded[:, :, u:u+3, v:v+3].reshape(N, -1)   # (N, 576)
            X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float32)])  # (N, 577)

            # Target ε for this spatial location
            t = y[:, 0, u, v]          # (N,)

            # Closed‑form least‑squares: [w ; b] = (XᵀX)⁻¹ Xᵀ t
            w_aug, _, _, _ = np.linalg.lstsq(X_aug, t, rcond=None)
            # w_aug has 576 weight entries followed by the bias scalar

            w_mat = w_aug[:576].reshape(64, 3, 3).astype(np.float32)
            bias_val = float(w_aug[576])

            weight_sum += w_mat
            bias_sum   += bias_val
            count      += 1

    # -----------------------------------------------------------------
    # 4. Average the per‑position solutions to obtain a single weight
    #    tensor and bias that preserve the original shapes.
    # -----------------------------------------------------------------
    out_weight_new = (weight_sum / count).astype(np.float32)   # (64,3,3)
    out_bias_new   = np.float32(bias_sum / count)

    # -----------------------------------------------------------------
    # 5. Write the updated values back into the weight dictionary.
    # -----------------------------------------------------------------
    w["out.weight"] = out_weight_new.reshape(1, 64, 3, 3)   # (1,64,3,3)
    w["out.bias"]   = np.array([out_bias_new], dtype=np.float32)  # (1,)

    # -----------------------------------------------------------------
    # 6. Return the (possibly modified) weight container.
    # -----------------------------------------------------------------
    return w
