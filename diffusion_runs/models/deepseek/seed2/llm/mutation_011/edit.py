import numpy as np

def edit(w, f):
    """
    Closed-form least-squares refit of the final output convolution using the
    full set of 2000 cached final hidden feature maps and target epsilons.
    This is a single linear solve, no gradients or iterative training.
    """
    F = f['features']          # (2000, 64, 16, 16) final hidden features (input to out)
    y = f['y']                 # (2000, 1, 16, 16) target epsilon
    N = F.shape[0]
    H = W = 16
    C = 64
    K = 3

    # Accumulate normal equations in float64 for numerical stability
    n_params = C * K * K + 1   # 577
    XtX = np.zeros((n_params, n_params), dtype=np.float64)
    Xty = np.zeros(n_params, dtype=np.float64)

    chunk_size = 200
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        F_chunk = F[start:end]                     # (B,64,16,16)
        y_chunk = y[start:end]                     # (B,1,16,16)
        B = end - start

        # Pad features with zeros for 3x3 conv, padding=1
        F_pad = np.pad(F_chunk, ((0,0),(0,0),(1,1),(1,1)), mode='constant')

        # Build sliding-window view of shape (B,16,16,64,3,3)
        shape = (B, H, W, C, K, K)
        strides = (F_pad.strides[0], F_pad.strides[2], F_pad.strides[3],
                   F_pad.strides[1], F_pad.strides[2], F_pad.strides[3])
        patches = np.lib.stride_tricks.as_strided(F_pad, shape=shape, strides=strides)

        # Reorder to (B, C, K, K, H, W) then flatten to (B, 576, 256)
        patches = patches.transpose(0, 3, 4, 5, 1, 2)   # (B,64,3,3,16,16)
        patches = patches.reshape(B, C*K*K, H*W)        # (B,576,256)

        # Design matrix for this chunk: (576, B*256)
        X_chunk = patches.transpose(1, 0, 2).reshape(C*K*K, -1)
        # Append bias row
        ones = np.ones((1, X_chunk.shape[1]), dtype=np.float32)
        X_aug = np.vstack([X_chunk, ones])              # (577, B*256)

        y_flat = y_chunk.reshape(-1)                    # (B*256,)

        XtX += X_aug @ X_aug.T
        Xty += X_aug @ y_flat

    # Small ridge for numerical stability (negligible with 512k samples)
    ridge = 1e-6
    XtX += ridge * np.eye(n_params, dtype=np.float64)

    # Solve normal equations
    w_new = np.linalg.solve(XtX, Xty)

    # Extract out.weight and out.bias
    out_weight = w_new[:C*K*K].reshape(1, C, K, K).astype(np.float32)
    out_bias = w_new[C*K*K:].astype(np.float32)

    # Clip to safe range
    out_weight = np.clip(out_weight, -1000.0, 1000.0)
    out_bias = np.clip(out_bias, -1000.0, 1000.0)

    # Update weights
    w['out.weight'] = out_weight
    w['out.bias'] = out_bias

    # Diagnostics
    diagnostics = {
        "out_weight_max": float(np.max(np.abs(out_weight))),
        "out_bias": float(out_bias[0]),
        "ridge": ridge,
        "n_samples": int(N),
    }

    return {"weights": w, "diagnostics": diagnostics}
