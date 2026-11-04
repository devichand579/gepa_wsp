import numpy as np
import math

def edit(w, f):
    # Extract features and targets
    features = f['features']          # (2000, 64, 16, 16)
    y = f['y']                        # (2000, 1, 16, 16)
    N, C, H, W = features.shape
    assert C == 64 and H == 16 and W == 16

    # Pad features with 1 zero on each side for 3x3 convolution with padding=1
    features_pad = np.pad(features, ((0,0),(0,0),(1,1),(1,1)), mode='constant')

    # Helper to compute normal equations for a subset of indices
    def compute_stats(indices):
        A = np.zeros((577, 577), dtype=np.float64)
        b = np.zeros(577, dtype=np.float64)
        yty = 0.0
        chunk_size = 200
        for start in range(0, len(indices), chunk_size):
            idx = indices[start:start+chunk_size]
            feats = features_pad[idx]                     # (chunk, 64, 18, 18)
            # Extract 3x3 patches: shape (chunk, 64, 16, 16, 3, 3)
            patches = np.lib.stride_tricks.sliding_window_view(feats, (3,3), axis=(2,3))
            # Transpose to (chunk, 16, 16, 64, 3, 3) so columns are ordered (c, di, dj)
            patches = patches.transpose(0, 2, 3, 1, 4, 5)
            # Flatten to (chunk*256, 64*9)
            X = patches.reshape(-1, 64*9)
            # Add bias column
            X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
            y_chunk = y[idx].reshape(-1).astype(np.float64)
            A += X_aug.T @ X_aug
            b += X_aug.T @ y_chunk
            yty += y_chunk @ y_chunk
        return A, b, yty

    all_indices = np.arange(N)
    A_total, b_total, yty_total = compute_stats(all_indices)

    # 5-fold cross-validation to select ridge
    n_folds = 5
    fold_indices = [np.arange(i, N, n_folds) for i in range(n_folds)]
    ridges = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    best_ridge = 0.0
    best_val_mse = np.inf

    for ridge in ridges:
        val_mses = []
        for i in range(n_folds):
            val_idx = fold_indices[i]
            A_val, b_val, yty_val = compute_stats(val_idx)
            A_train = A_total - A_val
            b_train = b_total - b_val
            reg = ridge * np.eye(577)
            try:
                theta = np.linalg.solve(A_train + reg, b_train)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(A_train + reg, b_train, rcond=None)[0]
            # Validation MSE per pixel
            val_mse = (yty_val - 2.0 * theta @ b_val + theta @ A_val @ theta) / (len(val_idx) * H * W)
            val_mses.append(val_mse)
        mean_val_mse = np.mean(val_mses)
        if mean_val_mse < best_val_mse:
            best_val_mse = mean_val_mse
            best_ridge = ridge

    # Final fit on all data with best ridge
    reg = best_ridge * np.eye(577)
    try:
        theta = np.linalg.solve(A_total + reg, b_total)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A_total + reg, b_total, rcond=None)[0]

    # Extract weights and bias
    # theta[:576] ordered as (c, di, dj) -> reshape to (64,3,3)
    W_new = theta[:576].reshape(64, 3, 3).astype(np.float32)
    b_new = theta[576].astype(np.float32)

    # Update w
    w['out.weight'] = W_new.reshape(1, 64, 3, 3)
    w['out.bias'] = np.array([b_new], dtype=np.float32)

    # Diagnostics
    diagnostics = {
        'best_ridge': float(best_ridge),
        'best_val_mse': float(best_val_mse),
        'out_weight_max': float(np.max(np.abs(W_new))),
        'out_weight_mean': float(np.mean(W_new)),
        'out_bias': float(b_new),
        'n_samples': int(N)
    }

    return {"weights": w, "diagnostics": diagnostics}
