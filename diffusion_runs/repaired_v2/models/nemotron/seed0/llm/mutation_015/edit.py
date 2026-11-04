import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def edit(w, f):
    """
    Closed‑form edit of the DDPM epsilon‑prediction network.
    Only the output convolution (out.weight + out.bias) is updated;
    all earlier weights stay unchanged.
    The update minimises per‑example MSE between the network’s epsilon
    prediction and the supplied target ε y using a single normal‑equation
    solution (no autograd, no iterative optimiser).
    """
    # ------------------------------------------------------------------
    # 1. Pull the permitted feedback arrays
    # ------------------------------------------------------------------
    features = f["features"]          # shape (N, 64, 16, 16), N = 2000
    y        = f["y"]                 # shape (N, 1, 16, 16)  target ε
    N        = features.shape[0]

    C        = 64                     # input channels to out.conv
    H        = W = 16                 # spatial size
    K        = 3                      # kernel size (pad‑1 → same output)
    out_w_shape = w["out.weight"].shape   # (1, 64, 3, 3)
    out_w_flat = C * K * K            # 576

    # ------------------------------------------------------------------
    # 2. Build the normal‑equation matrices G = XᵀX  and  h = Xᵀy
    # ------------------------------------------------------------------
    G = np.zeros((out_w_flat, out_w_flat), dtype=np.float32)
    h = np.zeros(out_w_flat, dtype=np.float32)

    for i in range(N):
        Ai = features[i]          # (64, 16, 16)
        yi = y[i]                 # (1, 16, 16)

        # ---- im2col with zero‑padding to emulate "same" conv ----
        Ai_pad = np.pad(Ai, ((0, 0), (1, 1), (1, 1)), mode="constant")
        # sliding_window_view on the last two axes → (64, 16, 16, 3, 3)
        patches = sliding_window_view(Ai_pad, (3, 3), axis=(1, 2))
        # reorder so each row = one output pixel, columns = 64·9 weights
        patches = patches.transpose(1, 2, 0, 3, 4)   # (16, 16, 64, 3, 3)
        X = patches.reshape(H * W, out_w_flat)       # (256, 576)
        yi_flat = yi.reshape(H * W)                 # (256,)

        # accumulate G and h
        G += X.T @ X
        h += X.T @ yi_flat

    # ------------------------------------------------------------------
    # 3. Solve the normal equations (minimum‑norm solution if rank‑deficient)
    # ------------------------------------------------------------------
    w_flat, residuals, rank, sv = np.linalg.lstsq(G, h, rcond=None)
    new_out_w = w_flat.reshape(out_w_shape).astype(np.float32)

    # enforce finite values and the absolute‑value ≤ 1000 constraint
    new_out_w = np.where(np.isfinite(new_out_w), new_out_w, 0.0)
    new_out_w = np.clip(new_out_w, -1000, 1000)

    # ------------------------------------------------------------------
    # 4. Update the output bias to the mean of all target ε values
    # ------------------------------------------------------------------
    y_all = y.reshape(-1)                # flatten all targets
    new_bias = np.mean(y_all).astype(np.float32)
    new_bias = np.clip(new_bias, -1000, 1000)

    # ------------------------------------------------------------------
    # 5. Return the modified weight dictionary (other layers untouched)
    # ------------------------------------------------------------------
    w_new = {k: v.copy() for k, v in w.items()}
    w_new["out.weight"] = new_out_w
    w_new["out.bias"]   = new_bias

    # ------------------------------------------------------------------
    # 6. Minimal diagnostics (all values finite, within limits)
    # ------------------------------------------------------------------
    diagnostics = {
        "out_weight_norm": float(np.linalg.norm(new_out_w)),
        "out_bias": float(new_bias),
        "G_rank": int(rank),
        "solved": True,
    }

    return {"weights": w_new, "diagnostics": diagnostics}
