import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    # Extract cached features and targets
    features = f['features']          # (2000, 64, 16, 16)
    y        = f['y']                 # (2000, 1, 16, 16)
    N, C, H, W = features.shape
    K = 3
    D = C * K * K + 1                 # 64*9 + 1 = 577

    # Accumulate Gram matrix and X^T y in float64 for stability
    G = np.zeros((D, D), dtype=np.float64)
    b = np.zeros((D, 1), dtype=np.float64)
    y_sq = 0.0

    chunk_size = 200
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        F = features[start:end]       # (n, 64, 16, 16)
        Y = y[start:end]              # (n, 1, 16, 16)
        n = F.shape[0]

        # Pad with 1 zero on each side to match conv padding=1
        F_pad = np.pad(F, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        # Extract 3x3 patches: (n, 64, 16, 16, 3, 3)
        patches = sliding_window_view(F_pad, (K, K), axis=(2, 3))
        # Rearrange to (n, 16, 16, 64, 3, 3) then flatten to (n*256, 576)
        patches = patches.transpose(0, 2, 3, 1, 4, 5)
        X = patches.reshape(n * H * W, C * K * K)

        # Augment with bias column
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X = np.concatenate([X, ones], axis=1)

        # Flatten targets
        Y_flat = Y.transpose(0, 2, 3, 1).reshape(n * H * W, 1)

        # Accumulate normal equations
        G += X.T @ X
        b += X.T @ Y_flat
        y_sq += np.sum(Y_flat ** 2)

    # Small ridge for numerical stability
    lambda_reg = 1e-6 * np.trace(G) / D
    G_reg = G + lambda_reg * np.eye(D, dtype=np.float64)

    # Solve for optimal output weights and bias
    w_sol = np.linalg.solve(G_reg, b).ravel()

    # Extract and reshape
    out_weight = w_sol[:C * K * K].reshape(1, C, K, K).astype(np.float32)
    out_bias   = w_sol[C * K * K:].astype(np.float32)

    # Update weights
    w['out.weight'] = out_weight
    w['out.bias']   = out_bias

    # Diagnostics
    rss = y_sq - w_sol @ b.ravel()
    mse_after = rss / (N * H * W)
    diagnostics = {
        'out_weight_max': float(np.max(np.abs(out_weight))),
        'out_bias': float(out_bias[0]),
        'lambda_reg': float(lambda_reg),
        'mse_before': float(np.mean((f['prediction'] - f['y']) ** 2)),
        'mse_after_linear_fit': float(mse_after)
    }

    return {'weights': w, 'diagnostics': diagnostics}
