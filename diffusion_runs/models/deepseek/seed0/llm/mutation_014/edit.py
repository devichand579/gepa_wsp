import numpy as np
import math

def edit(w, f):
    # Extract arrays
    features = f['features'].astype(np.float64)          # [2000,64,16,16]
    y        = f['y'].astype(np.float64)                 # [2000,1,16,16]
    N, C, H, W = features.shape
    assert C == 64 and H == 16 and W == 16
    y_flat = y[:, 0].reshape(N, H, W)                    # [2000,16,16]

    # Build design matrix A in chunks and accumulate A^T A and A^T y
    # Column order: c*9 + di*3 + dj, plus bias column at the end.
    n_cols = C * 9 + 1
    AtA = np.zeros((n_cols, n_cols), dtype=np.float64)
    Aty = np.zeros(n_cols, dtype=np.float64)

    chunk_size = 200
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        B = end - start
        h = features[start:end]                          # [B,64,16,16]
        # Pad with one zero on each spatial side
        pad = np.zeros((B, C, H + 2, W + 2), dtype=np.float64)
        pad[:, :, 1:-1, 1:-1] = h

        # Build shifted versions: for each (di,dj) slice
        shifts = []
        for di in range(3):
            for dj in range(3):
                shifts.append(pad[:, :, di:di+H, dj:dj+W])   # [B,64,16,16]
        # Stack along new axis -> [B,64,9,16,16]
        shifts = np.stack(shifts, axis=2)
        # Rearrange to [B,16,16,64,9] then reshape to [B*H*W, 576]
        A_chunk = shifts.transpose(0, 3, 4, 1, 2).reshape(B * H * W, C * 9)
        # Append bias column
        A_chunk = np.concatenate([A_chunk, np.ones((B * H * W, 1))], axis=1)

        y_chunk = y_flat[start:end].reshape(B * H * W, 1)

        AtA += A_chunk.T @ A_chunk
        Aty += (A_chunk.T @ y_chunk).ravel()

    # Tiny ridge for numerical stability (relative to trace)
    ridge = 1e-8 * np.trace(AtA) / n_cols
    AtA += ridge * np.eye(n_cols)

    # Solve normal equations
    w_new = np.linalg.solve(AtA, Aty)

    # Reshape weights: first 576 are conv weights, last is bias
    conv_w = w_new[:C * 9].reshape(C, 3, 3)              # [c, di, dj]
    bias   = w_new[C * 9]

    # Map to PyTorch layout [1,64,3,3]
    out_weight = conv_w.reshape(1, C, 3, 3).astype(np.float32)
    out_bias   = np.array([bias], dtype=np.float32)

    # Diagnostics: compute residual gradient norm and MSE estimate
    # (using the same chunked approach, but only for stats)
    # We can compute the MSE of the new weights on the full set quickly:
    # Since we have AtA and Aty, the optimal MSE = (y^T y - w^T Aty) / (N*H*W)
    yty = np.sum(y_flat ** 2)
    mse_opt = (yty - w_new @ Aty) / (N * H * W)
    mse_current = np.mean((f['prediction'] - f['y']) ** 2)

    # Update weights
    w['out.weight'] = out_weight
    w['out.bias']   = out_bias

    diagnostics = {
        "mse_current": float(mse_current),
        "mse_optimal_out": float(mse_opt),
        "ridge": float(ridge),
        "out_weight_norm": float(np.linalg.norm(out_weight)),
        "out_bias": float(out_bias[0]),
        "n_params": int(n_cols)
    }
    return {"weights": w, "diagnostics": diagnostics}
