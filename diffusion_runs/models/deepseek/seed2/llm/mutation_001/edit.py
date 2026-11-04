import numpy as np
import numpy.lib.stride_tricks as st

def edit(w, f):
    """
    Closed-form least-squares update of the final 3x3 convolution (out layer)
    using the cached final hidden features and target epsilon.
    The features are the input to the out layer, so this update is exact
    and does not invalidate any upstream cached activations.
    """
    F = f['features']          # (2000, 64, 16, 16) final hidden features
    y = f['y']                 # (2000, 1, 16, 16) target Gaussian epsilon
    N = F.shape[0]
    D = 64 * 9 + 1             # 576 weights + 1 bias

    XtX = np.zeros((D, D), dtype=np.float64)
    Xty = np.zeros(D, dtype=np.float64)
    yty = 0.0

    batch_size = 100
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Fb = F[start:end]      # (B, 64, 16, 16)
        yb = y[start:end, 0]   # (B, 16, 16)
        B = Fb.shape[0]

        # Zero-pad spatially to emulate conv2d padding=1
        F_pad = np.pad(Fb, ((0, 0), (0, 0), (1, 1), (1, 1)))

        # Extract 3x3 patches: shape (B, 64, 16, 16, 3, 3)
        windows = st.sliding_window_view(F_pad, (3, 3), axis=(2, 3))

        # Rearrange to (B, 16, 16, 64, 3, 3) and flatten to (B*256, 576)
        X = windows.transpose(0, 2, 3, 1, 4, 5).reshape(B * 256, 576)

        # Append bias column
        X_aug = np.concatenate([X, np.ones((B * 256, 1))], axis=1)

        y_flat = yb.reshape(-1)

        XtX += X_aug.T @ X_aug
        Xty += X_aug.T @ y_flat
        yty += np.sum(y_flat.astype(np.float64) ** 2)

    # Small ridge for numerical stability
    lam = 1e-6 * np.trace(XtX) / D
    XtX_reg = XtX + lam * np.eye(D)

    W = np.linalg.solve(XtX_reg, Xty)

    out_w = W[:576].reshape(1, 64, 3, 3)
    out_b = W[576:577]

    # Enforce absolute value <= 1000
    out_w = np.clip(out_w, -1000, 1000)
    out_b = np.clip(out_b, -1000, 1000)

    w['out.weight'] = out_w.astype(np.float32)
    w['out.bias'] = out_b.astype(np.float32)

    # Diagnostics
    RSS = yty - W @ Xty
    MSE = RSS / (N * 256)
    diagnostics = {
        'new_mse': float(MSE),
        'out_weight_max': float(np.max(np.abs(out_w))),
        'out_bias': float(out_b[0])
    }

    return {"weights": w, "diagnostics": diagnostics}
