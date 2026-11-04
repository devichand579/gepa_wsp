import numpy as np
import math

def edit(w, f):
    # Extract features and targets
    F = f['features']          # (2000, 64, 16, 16)
    y = f['y']                 # (2000, 1, 16, 16)
    N, C, H, W = F.shape
    M = N * H * W

    # Pad features with zeros for 3x3 convolution (padding=1)
    F_pad = np.pad(F, ((0,0),(0,0),(1,1),(1,1)), mode='constant')

    # Build shifted feature matrices for each of the 9 kernel positions
    shifts = []
    for kh in range(3):
        for kw in range(3):
            # Slice the padded features to get the shifted version
            F_shift = F_pad[:, :, kh:kh+H, kw:kw+W]  # (N, C, H, W)
            # Reshape to (M, C) with row order (n, i, j)
            F_shift = F_shift.transpose(0, 2, 3, 1).reshape(M, C)
            shifts.append(F_shift)

    # Concatenate all shifts along columns -> (M, 9*C) = (M, 576)
    F_all = np.concatenate(shifts, axis=1)

    # Flatten targets to (M, 1)
    y_flat = y.transpose(0, 2, 3, 1).reshape(M, 1)

    # Compute normal equations
    A_blocks = F_all.T @ F_all               # (576, 576)
    b_blocks = F_all.T @ y_flat              # (576, 1)
    sum_F = F_all.sum(axis=0)                # (576,)
    sum_y = y_flat.sum()

    # Build full system (577 x 577)
    A = np.zeros((577, 577), dtype=np.float64)
    b = np.zeros(577, dtype=np.float64)

    A[:576, :576] = A_blocks
    A[:576, 576] = sum_F
    A[576, :576] = sum_F
    A[576, 576] = M

    b[:576] = b_blocks.flatten()
    b[576] = sum_y

    # Small ridge for numerical stability
    ridge = 1e-6 * np.trace(A) / 577.0
    A += ridge * np.eye(577)

    # Solve least squares
    W = np.linalg.solve(A, b)

    # Reshape weights back to (1, 64, 3, 3)
    W_conv = W[:576].reshape(9, 64)
    W_new = np.zeros((1, 64, 3, 3), dtype=np.float32)
    for s in range(9):
        kh = s // 3
        kw = s % 3
        W_new[0, :, kh, kw] = W_conv[s, :]

    b_new = np.array([W[576]], dtype=np.float32)

    # Update weights
    w['out.weight'] = W_new
    w['out.bias'] = b_new

    # Diagnostics
    pred = F_all @ W[:576] + W[576]
    mse = np.mean((pred - y_flat.flatten())**2)
    diagnostics = {
        "mse_after_fit": float(mse),
        "out_weight_norm": float(np.linalg.norm(W_new)),
        "out_bias": float(b_new[0]),
        "ridge": float(ridge),
        "n_params": 577
    }

    return {"weights": w, "diagnostics": diagnostics}
