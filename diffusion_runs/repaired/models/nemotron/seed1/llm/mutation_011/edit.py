import numpy as np

def edit(w, f):
    """
    Closed‑form least‑squares update of the output layer (out.weight) to
    minimise the per‑pixel MSE between the current epsilon prediction and
    the target Gaussian epsilon.

    Parameters
    ----------
    w : dict
        Current network parameters (all full‑precision tensors).  Keys
        include every weight/bias tensor; shapes must be preserved.
    f : dict
        Permitted feedback arrays only.  Required keys:
            - "prediction"   : float32 [N,1,16,16] current epsilon prediction
            - "y"            : float32 [N,1,16,16] target Gaussian epsilon
            - "layer.out.input0" : float32 [N,64,16,16] activations feeding
              the output convolution (already passed through ReLU etc.)

    Returns
    -------
    w : dict
        Updated weights (out.weight modified).  All other keys unchanged.
    {"weights": w, "diagnostics": {...}}   (or just w – the caller may
    wrap as needed).  Diagnostics contain before/after MSE and the norm of
    the weight update.
    """
    # ------------------------------------------------------------------
    # 1.  Gather error and activations
    # ------------------------------------------------------------------
    e = f["prediction"] - f["y"]                # [N,1,16,16], float32
    N = e.shape[0]                              # e.g. 128
    e_vec = e.reshape(-1)                        # [N*256]   (16*16)

    # activations feeding the output conv: [N,64,16,16]
    inp = f["layer.out.input0"]                  # float32

    # ------------------------------------------------------------------
    # 2.  Build the design matrix X (size [N*256, 576])
    #    each row = flattened 3×3×64 patch centred at one output pixel
    # ------------------------------------------------------------------
    H, W = 16, 16
    C = 64
    K = 3
    n_rows = N * H * W            # 128 * 256 = 32768
    X = np.empty((n_rows, C * K * K), dtype=np.float32)   # 576 columns
    idx = 0

    for b in range(N):
        # zero‑pad the spatial dimensions (same 'pad1' as in the forward pass)
        pad_in = np.pad(inp[b], ((0, 0), (1, 1), (1, 1)),
                        mode='constant', constant_values=0)   # [64,18,18]

        for i in range(H):
            for j in range(W):
                # extract the 3×3 window centred at (i,j)
                patch = pad_in[:, i:i+K, j:j+K].reshape(-1)   # 576 elements
                X[idx] = patch
                idx += 1

    # ------------------------------------------------------------------
    # 3.  Closed‑form least‑squares solution for the output weights
    # ------------------------------------------------------------------
    # np.linalg.lstsq returns (solution, residuals, rank, singular_values)
    W_flat, _, _, _ = np.linalg.lstsq(X, e_vec, rcond=None)   # shape (576,)

    # Reshape back to the original convolution kernel shape [1,64,3,3]
    W_new = W_flat.reshape(w["out.weight"].shape).astype(np.float32)

    # Clip to the allowed finite‑float32 range [-1000, 1000]
    W_new = np.clip(W_new, -1000.0, 1000.0)

    # ------------------------------------------------------------------
    # 4.  Apply the update (in‑place on the weight dict)
    # ------------------------------------------------------------------
    w["out.weight"] = w["out.weight"] + W_new

    # ------------------------------------------------------------------
    # 5.  Diagnostics: before/after MSE and weight‑update norm
    # ------------------------------------------------------------------
    mse_before = np.mean(e**2)                     # scalar float32

    # Residual after the linear correction: e - X @ W_flat
    residual = e_vec - X @ W_flat
    mse_after = np.mean(residual**2)

    diag = {
        "mse_before": float(mse_before),
        "mse_after":  float(mse_after),
        "weight_update_norm": float(np.linalg.norm(W_new))
    }

    # ------------------------------------------------------------------
    # 6.  Return the (modified) weight dict together with diagnostics
    # ------------------------------------------------------------------
    return {"weights": w, "diagnostics": diag}
