import numpy as np

def edit(w, f):
    # --- Extract 128-example batch data ---
    X = f['layer.out.input0'].astype(np.float32)          # [128, 64, 16, 16]
    Y = f['batch.y'].astype(np.float32)                   # [128, 1, 16, 16]
    N, C, H, W = X.shape                                  # 128, 64, 16, 16

    # --- Im2col: unfold X into design matrix D [N*H*W, C*3*3] ---
    # Pad X with 1 zero border (matches network pad1 convention)
    Xp = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)),
                mode='constant', constant_values=0.0)     # [128, 64, 18, 18]
    # Prepare output matrix: each row = flattened 3x3 patch for one spatial position
    n_pixels = N * H * W                                   # 32768
    n_weights = C * 3 * 3                                  # 576
    D = np.zeros((n_pixels, n_weights), dtype=np.float32)
    Yv = Y.reshape(-1).astype(np.float32)                 # [32768]

    # Fill D by iterating over input channels and kernel offsets
    # (vectorized over the batch dimension for speed)
    idx = 0
    for ci in range(C):
        for di in range(3):
            for dj in range(3):
                # input patch at offset (di-1, dj-1) relative to center
                # padded center starts at row/col 1, so patch rows [di : di+H], cols [dj : dj+W]
                patch = Xp[:, ci, di:di+H, dj:dj+W]       # [128, 16, 16]
                # flatten patch and place into column
                D[:, idx] = patch.reshape(-1)
                idx += 1

    # --- Closed-form least-squares solution ---
    # Normal equations: (D^T D) w = D^T Y
    # D^T D is 576x576, D^T Y is 576
    DTY = D.T @ Yv                                         # [576]
    DTD = D.T @ D                                          # [576, 576]
    # Use lstsq for numerical stability; this is a single closed-form solve, no iterative optimizer
    w_new_flat, residuals, rank, s = np.linalg.lstsq(DTD, DTY, rcond=None)

    # --- Reshape to convolutional weight tensor ---
    new_weight = w_new_flat.reshape((1, 64, 3, 3)).astype(np.float32)
    # Clip to finite float32 and |w| <= 1000
    new_weight = np.clip(new_weight, -1000.0, 1000.0)

    # --- Update output bias (single output channel) ---
    # bias = mean(Y - X @ W) over all pixels/examples
    pred = (X.reshape(n_pixels, 64) @ new_weight.reshape(-1)).reshape(Y.shape)
    new_bias = float(np.mean(Yv - pred.flatten()))

    # --- Apply updates to weight dict ---
    w = dict(w)  # shallow copy to preserve original keys/shapes elsewhere
    w['out.weight'] = new_weight
    w['out.bias'] = np.float32(new_bias)

    # --- Diagnostics (small JSON‑compatible dict) ---
    diag = {
        "output_mse": float(np.mean((pred - Y) ** 2)),
        "weight_norm": float(np.linalg.norm(new_weight)),
        "bias": float(new_bias),
        "solved_pixels": int(np.sum(np.isfinite(D))),
        "rank": int(rank)
    }

    return {"weights": w, "diagnostics": diag}
