import numpy as np

def edit(w, f):
    F = f['features']          # (2000, 64, 16, 16)
    Y = f['y']                 # (2000, 1, 16, 16)
    N = F.shape[0]
    D = 64 * 3 * 3 + 1         # 577 (576 weights + 1 bias)

    XtX = np.zeros((D, D), dtype=np.float64)
    Xty = np.zeros(D, dtype=np.float64)
    yty = 0.0

    chunk = 100
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        B = end - start

        F_chunk = F[start:end]          # (B, 64, 16, 16)
        Y_chunk = Y[start:end]          # (B, 1, 16, 16)

        # Pad with one zero border to handle 3x3 conv with padding=1
        Fp = np.pad(F_chunk, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

        # Extract 3x3 patches for every output location
        s = Fp.strides
        patches = np.lib.stride_tricks.as_strided(
            Fp,
            shape=(B, 16, 16, 64, 3, 3),
            strides=(s[0], s[2], s[3], s[1], s[2], s[3]),
            writeable=False
        )
        X = patches.reshape(B * 256, 576)

        # Augment with bias column
        X_aug = np.empty((B * 256, D), dtype=np.float64)
        X_aug[:, :576] = X.astype(np.float64)
        X_aug[:, 576] = 1.0

        y_flat = Y_chunk.reshape(B * 256).astype(np.float64)

        XtX += X_aug.T @ X_aug
        Xty += X_aug.T @ y_flat
        yty += np.sum(y_flat ** 2)

    # Tiny ridge for numerical stability (negligible bias)
    ridge = 1e-8 * np.trace(XtX) / D
    XtX_reg = XtX + np.eye(D) * ridge
    coef = np.linalg.solve(XtX_reg, Xty)

    # Training MSE for the new readout
    mse = (yty - 2.0 * coef @ Xty + coef @ XtX @ coef) / (N * 256)

    # Reshape into the out layer's weight tensor (1, 64, 3, 3) and bias (1,)
    W_new = coef[:576].reshape(1, 64, 3, 3).astype(np.float32)
    b_new = np.array([coef[576]], dtype=np.float32)

    w['out.weight'] = W_new
    w['out.bias'] = b_new

    diagnostics = {
        'new_mse': float(mse),
        'ridge': float(ridge),
        'out_weight_max': float(np.max(np.abs(W_new))),
        'out_bias': float(b_new[0])
    }
    return {"weights": w, "diagnostics": diagnostics}
