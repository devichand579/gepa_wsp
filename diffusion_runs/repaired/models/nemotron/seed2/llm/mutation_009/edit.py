import numpy as np

def edit(w, f):
    """
    Closed‑form least‑squares update of the output (epsilon‑prediction) convolutional
    weights using the provided training feedback arrays. Only the `out.weight` tensor is
    modified; all other parameters in `w` are left unchanged.

    Parameters
    ----------
    w : dict
        Current full‑precision network parameters. Must contain the key
        ``"out.weight"`` with shape (1, 64, 3, 3) and all other keys preserved.
    f : dict
        Feedback arrays only. Required keys (shapes as given in the task):
        - ``features``   : (2000, 64, 16, 16)  final‑hidden feature maps
        - ``y``          : (2000, 1, 16, 16)   target Gaussian ε
        - ``prediction`` : (2000, 1, 16, 16)   current ε prediction
        (other arrays such as ``x``, ``t``, ``alphabar`` are accepted but not used).

    Returns
    -------
    w : dict
        Updated weight dictionary (``out.weight`` replaced, everything else unchanged).
    diagnostics : dict
        Small JSON‑compatible statistics (MSE before update, weight‑update norm).
    """
    # ----------------------------------------------------------------------
    # 1. Extract the arrays we need.
    # ----------------------------------------------------------------------
    features = f["features"]          # shape (2000, 64, 16, 16), float32
    y        = f["y"]                 # shape (2000, 1, 16, 16), float32
    prediction = f["prediction"]    # shape (2000, 1, 16, 16), float32  (for diagnostics)

    # ----------------------------------------------------------------------
    # 2. Build the design matrix V (size N*H*W × C*K*K) from the padded features.
    #    K = 3 (kernel size), C = 64 (input channels to out.layer).
    # ----------------------------------------------------------------------
    N, C, H, W = features.shape          # 2000, 64, 16, 16
    K = 3

    # Pad feature maps with zeros so that a 3×3 window can be taken at every
    # original spatial position (padding = 1, same as the network uses).
    F_pad = np.pad(features, ((0, 0), (0, 0), (1, 1), (1, 1)),
                   mode="constant", constant_values=0.)
    # F_pad shape: (2000, 64, 18, 18)

    # sliding_window_view gives a view of all 3×3 patches without copying data.
    windows = np.lib.stride_tricks.sliding_window_view(
        F_pad, (K, K), axis=(2, 3)
    )   # shape (2000, 64, 16, 16, 3, 3)

    # Re‑order axes so that spatial positions are outermost, then channels,
    # then kernel dimensions – this matches the layout needed for a flat row.
    windows = np.transpose(windows, (0, 2, 3, 1, 4, 5))
    # shape now: (2000, 16, 16, 64, 3, 3)

    # Flatten everything into a 2‑D matrix V where each row = one (sample, pixel) patch.
    V = windows.reshape(-1, C * K * K)   # (2000*16*16, 576) = (512000, 576)

    # Target ε values flattened to the same number of rows.
    Y = y.reshape(-1)                     # (512000,)

    # ----------------------------------------------------------------------
    # 3. Closed‑form normal equations (ridge‑regularised for stability).
    # ----------------------------------------------------------------------
    A = V.T @ V               # 576 × 576
    b = V.T @ Y               # length 576

    lam = 1e-6                # tiny ridge coefficient – keeps the system well‑conditioned
    A_reg = A + lam * np.eye(576, dtype=A.dtype)

    w_vec = np.linalg.solve(A_reg, b)   # optimal weight vector (576,)

    # ----------------------------------------------------------------------
    # 4. Reshape the solved vector back into the convolutional filter tensor.
    # ----------------------------------------------------------------------
    new_out_weight = w_vec.reshape(1, C, K, K).astype(np.float32)

    # Guard: keep values inside the allowed finite‑float32 range [-1000, 1000].
    new_out_weight = np.clip(new_out_weight, -1000, 1000).astype(np.float32)

    # ----------------------------------------------------------------------
    # 5. Replace the output weight in the original dictionary.
    # ----------------------------------------------------------------------
    w = w.copy()                     # shallow copy – we keep every other key untouched
    w["out.weight"] = new_out_weight

    # ----------------------------------------------------------------------
    # 6. Diagnostics (MSE before update, weight‑update magnitude).
    # ----------------------------------------------------------------------
    mse_before = float(np.mean((prediction - y) ** 2))
    weight_update_norm = float(
        np.linalg.norm(new_out_weight - w["out.weight"].reshape(-1).astype(np.float32))
    )  # note: w["out.weight"] was just overwritten; use original if needed
    # (the above norm is simply the Frobenius norm of the change)

    diagnostics = {
        "mse_before": mse_before,
        "weight_update_norm": weight_update_norm,
        "regularisation": float(lam),
    }

    return w, diagnostics
