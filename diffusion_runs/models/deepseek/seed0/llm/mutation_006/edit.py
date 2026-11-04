import numpy as np
import numpy.lib.stride_tricks as st

def edit(w, f):
    features = f['features']  # (2000, 64, 16, 16)
    y = f['y']                # (2000, 1, 16, 16)
    N = features.shape[0]

    # Design matrix has 64*3*3 = 576 weights + 1 bias = 577 parameters.
    ATA = np.zeros((577, 577), dtype=np.float64)
    ATb = np.zeros(577, dtype=np.float64)

    batch_size = 200
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        feat_batch = features[start:end]          # (B, 64, 16, 16)
        y_batch = y[start:end, 0]                 # (B, 16, 16)

        # Pad to (B, 64, 18, 18)
        pad = np.pad(feat_batch, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        # Sliding windows: (B, 64, 16, 16, 3, 3)
        windows = st.sliding_window_view(pad, (3, 3), axis=(2, 3))
        # Transpose to (B, 16, 16, 64, 3, 3) and flatten to (B*256, 576)
        A = windows.transpose(0, 2, 3, 1, 4, 5).reshape(B * 256, -1)
        # Add bias column
        A = np.concatenate([A, np.ones((B * 256, 1), dtype=np.float32)], axis=1)
        b = y_batch.reshape(-1)

        ATA += A.T @ A
        ATb += A.T @ b

    # Solve with tiny ridge for stability
    ridge = 1e-6
    ATA_reg = ATA + ridge * np.eye(577)
    w_sol = np.linalg.solve(ATA_reg, ATb)

    out_weight = w_sol[:576].reshape(1, 64, 3, 3).astype(np.float32)
    out_bias = w_sol[576:577].astype(np.float32)

    w['out.weight'] = out_weight
    w['out.bias'] = out_bias

    # Compute training MSE with the new output layer (features unchanged)
    mse_sum = 0.0
    total_pixels = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        feat_batch = features[start:end]
        y_batch = y[start:end, 0]

        pad = np.pad(feat_batch, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        windows = st.sliding_window_view(pad, (3, 3), axis=(2, 3))
        A = windows.transpose(0, 2, 3, 1, 4, 5).reshape(B * 256, -1)
        A = np.concatenate([A, np.ones((B * 256, 1), dtype=np.float32)], axis=1)
        pred = A @ w_sol
        b = y_batch.reshape(-1)
        mse_sum += np.sum((pred - b) ** 2)
        total_pixels += B * 256

    mse = mse_sum / total_pixels
    diagnostics = {
        'mse': float(mse),
        'out_weight_norm': float(np.linalg.norm(out_weight)),
        'out_bias': float(out_bias[0]),
        'ridge': ridge
    }
    return {"weights": w, "diagnostics": diagnostics}
