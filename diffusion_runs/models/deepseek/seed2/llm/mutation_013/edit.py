import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    # Extract features and targets
    F = f['features']  # (2000, 64, 16, 16)
    Y = f['y']         # (2000, 1, 16, 16)
    N = F.shape[0]
    y_flat_all = Y.reshape(N, -1)  # (2000, 256)

    # Output layer has 64*3*3 = 576 weights + 1 bias = 577 parameters
    n_params = 64 * 9 + 1
    A = np.zeros((n_params, n_params), dtype=np.float64)
    b = np.zeros(n_params, dtype=np.float64)

    batch_size = 200
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        F_batch = F[start:end]          # (B, 64, 16, 16)
        y_batch = y_flat_all[start:end] # (B, 256)

        # im2col with padding=1, kernel=3x3
        padded = np.pad(F_batch, ((0, 0), (0, 0), (1, 1), (1, 1)))
        windows = sliding_window_view(padded, (3, 3), axis=(2, 3))  # (B, 64, 16, 16, 3, 3)
        patches = windows.transpose(0, 2, 3, 1, 4, 5).reshape(-1, 64 * 9)  # (B*256, 576)

        # Augment with bias column
        X = np.concatenate([patches, np.ones((patches.shape[0], 1), dtype=patches.dtype)], axis=1).astype(np.float64)
        y_vec = y_batch.reshape(-1).astype(np.float64)

        A += X.T @ X
        b += X.T @ y_vec

    # Tiny ridge for numerical stability
    ridge = 1e-6
    A += ridge * np.eye(n_params)
    w_out = np.linalg.solve(A, b)  # (577,)

    # Update only the output layer
    w['out.weight'] = w_out[:576].reshape(1, 64, 3, 3).astype(np.float32)
    w['out.bias'] = w_out[576:577].astype(np.float32)

    # Diagnostics: MSE before and after
    pred_before = f['prediction']  # (2000, 1, 16, 16)
    mse_before = float(np.mean((pred_before - Y) ** 2))

    total_sq_err = 0.0
    total_count = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        F_batch = F[start:end]
        y_batch = y_flat_all[start:end]

        padded = np.pad(F_batch, ((0, 0), (0, 0), (1, 1), (1, 1)))
        windows = sliding_window_view(padded, (3, 3), axis=(2, 3))
        patches = windows.transpose(0, 2, 3, 1, 4, 5).reshape(-1, 64 * 9)

        pred = patches @ w_out[:576] + w_out[576]
        y_vec = y_batch.reshape(-1)
        total_sq_err += np.sum((pred - y_vec) ** 2)
        total_count += y_vec.size

    mse_after = float(total_sq_err / total_count)

    diagnostics = {
        'mse_before': mse_before,
        'mse_after': mse_after,
        'out_weight_max': float(np.max(np.abs(w['out.weight']))),
        'out_bias': float(w['out.bias'][0]),
        'ridge': ridge
    }

    return {"weights": w, "diagnostics": diagnostics}
