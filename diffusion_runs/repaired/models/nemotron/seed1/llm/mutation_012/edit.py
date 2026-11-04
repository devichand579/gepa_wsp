import numpy as np

def edit(w, f):
    """
    Closed‑form edit of the output convolution weights using the
    provided feedback arrays.  The function fits a linear regression
    from the final hidden feature maps to the target epsilon, thereby
    minimising the per‑example MSE in one shot.  All other weights are
    left unchanged.
    """
    # --- feedback arrays -------------------------------------------------
    features = f["features"]                # shape (128, 64, 16, 16)
    y_target = f["y"]                       # shape (128, 1, 16, 16)

    N, C, H, W = features.shape             # 128, 64, 16, 16

    # Pad feature maps with 1 pixel of zero on each side so that a
    # 3×3 conv with padding=1 yields the same 16×16 output size.
    features_pad = np.pad(features,
                         ((0, 0), (1, 1), (1, 1), (0, 0)),
                         mode="constant")        # (128,64,18,18)

    # Extract all 3×3 patches at every spatial position.
    # sliding_window_view returns shape (128,64,16,16,3,3)
    windowed = np.lib.stride_tricks.sliding_window_view(
        features_pad, (3, 3), axis=(2, 3))

    # Flatten examples and spatial positions into one design matrix:
    #   X  : (128*16*16, 64*9) = (32768, 576)
    #   y  : (32768,)
    X = windowed.reshape(-1, C * 9)        # (32768, 576)
    y = y_target.reshape(-1)               # (32768,)

    # Add a bias column (ones) so the regression can learn an offset.
    X_bias = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)],
                       dtype=np.float32)

    # Closed‑form ordinary least‑squares solution.
    w_opt, _, _, _ = np.linalg.lstsq(
        X_bias, y.astype(np.float32), rcond=None)

    # w_opt has length 577: first 576 entries are the conv weights,
    # the last entry is the bias.
    w_weight = w_opt[:576].reshape(1, C, 3, 3).astype(np.float32)
    w_bias   = float(w_opt[576])

    # Update only the output layer; keep everything else exactly as before.
    w["out.weight"] = w_weight
    w["out.bias"]   = np.float32(w_bias)

    # Small diagnostics (all values are finite float32).
    diagnostics = {
        "out_weight_norm": float(np.linalg.norm(w_weight)),
        "out_bias": w_bias,
        "X_rows": int(X.shape[0]),
    }

    return {"weights": w, "diagnostics": diagnostics}
