import numpy as np
import math

def edit(w, f):
    # Extract feedback arrays
    features = f['features']          # (2000, 64, 16, 16)
    y = f['y']                        # (2000, 1, 16, 16)
    N, C, H, W = features.shape
    assert C == 64 and H == 16 and W == 16
    M = N * H * W                     # total number of output pixels
    K = 3                             # kernel size
    P = C * K * K                     # number of weights = 576

    # Reshape features to (N, C, H*W) for easier slicing
    feats = features.reshape(N, C, H * W)
    y_flat = y.reshape(N, 1, H * W)

    # Precompute sum of features and y for bias
    sum_feats = feats.sum(axis=(0, 2))          # (C,)
    sum_y = y_flat.sum()                        # scalar

    # We will accumulate the Gram matrix A (P x P) and vector b (P)
    A = np.zeros((P, P), dtype=np.float64)
    b = np.zeros(P, dtype=np.float64)
    # Also need S_off: sum of features for each offset (i,j) over the valid region
    # We'll compute it as part of the augmented system later.
    # For now, compute the cross-correlation matrices C[di,dj] for di,dj in [-2,2]
    # and cross-correlation vectors V[i,j] for i,j in [-1,1].

    # Helper to get shifted features with zero padding
    # We'll work with feats of shape (N, C, H, W) for spatial slicing
    feats_sp = features  # (N, C, H, W)

    # Compute cross-correlation matrices C[di,dj] for di,dj in [-2,2]
    # C[di,dj][c1,c2] = sum_{n,r,c} feats[n,c1,r,c] * feats[n,c2,r+di,c+dj]
    # where both are in bounds.
    Cmat = {}
    for di in range(-2, 3):
        for dj in range(-2, 3):
            # Determine valid slices
            if di >= 0:
                r1 = slice(0, H - di)
                r2 = slice(di, H)
            else:
                r1 = slice(-di, H)
                r2 = slice(0, H + di)
            if dj >= 0:
                c1 = slice(0, W - dj)
                c2 = slice(dj, W)
            else:
                c1 = slice(-dj, W)
                c2 = slice(0, W + dj)
            # Extract A and B: shape (N, C, len_r, len_c)
            A_feat = feats_sp[:, :, r1, c1]
            B_feat = feats_sp[:, :, r2, c2]
            # Reshape to (N*len_r*len_c, C)
            A_mat = A_feat.transpose(0, 2, 3, 1).reshape(-1, C)
            B_mat = B_feat.transpose(0, 2, 3, 1).reshape(-1, C)
            # Compute C = A^T @ B
            Cmat[(di, dj)] = A_mat.T @ B_mat

    # Compute cross-correlation vectors V[i,j] for i,j in [-1,1]
    # V[i,j][c] = sum_{n,r,c} feats[n,c,r+i,c+j] * y[n,0,r,c]
    Vvec = {}
    for i in range(-1, 2):
        for j in range(-1, 2):
            # We need to align features at (r+i, c+j) with y at (r,c)
            # So for output position (r,c), feature is at (r+i, c+j).
            # Valid r,c such that 0 <= r+i < H and 0 <= c+j < W.
            if i >= 0:
                r_feat = slice(i, H)
                r_y = slice(0, H - i)
            else:
                r_feat = slice(0, H + i)
                r_y = slice(-i, H)
            if j >= 0:
                c_feat = slice(j, W)
                c_y = slice(0, W - j)
            else:
                c_feat = slice(0, W + j)
                c_y = slice(-j, W)
            # Extract feature and y slices
            F_slice = feats_sp[:, :, r_feat, c_feat]   # (N, C, len_r, len_c)
            Y_slice = y[:, 0, r_y, c_y]                # (N, len_r, len_c)
            # Reshape to (M', C) and (M',)
            F_mat = F_slice.transpose(0, 2, 3, 1).reshape(-1, C)
            Y_vec = Y_slice.reshape(-1)
            Vvec[(i, j)] = F_mat.T @ Y_vec

    # Compute S_off[i,j][c] = sum over n and valid (r,c) of feats[n,c,r+i,c+j]
    # This is the sum of features over the region valid for offset (i,j).
    Soff = {}
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i >= 0:
                r_slice = slice(i, H)
            else:
                r_slice = slice(0, H + i)
            if j >= 0:
                c_slice = slice(j, W)
            else:
                c_slice = slice(0, W + j)
            # Sum over N and spatial dims
            Soff[(i, j)] = feats_sp[:, :, r_slice, c_slice].sum(axis=(0, 2, 3))  # (C,)

    # Now assemble the P x P matrix A and vector b
    # Index mapping: idx = c * 9 + (i+1)*3 + (j+1)
    for i in range(-1, 2):
        for j in range(-1, 2):
            idx1 = (i + 1) * 3 + (j + 1)
            for i2 in range(-1, 2):
                for j2 in range(-1, 2):
                    idx2 = (i2 + 1) * 3 + (j2 + 1)
                    di = i - i2
                    dj = j - j2
                    # di,dj in [-2,2]
                    block = Cmat[(di, dj)]  # (C, C)
                    # Place into A: rows for c1, cols for c2
                    # A[c1*9+idx1, c2*9+idx2] = block[c1,c2]
                    # We can use broadcasting: rows = c1*9+idx1, cols = c2*9+idx2
                    # Create index arrays
                    row_idx = np.arange(C) * 9 + idx1
                    col_idx = np.arange(C) * 9 + idx2
                    # Use np.ix_ to assign block
                    A[np.ix_(row_idx, col_idx)] = block

    # Assemble b: b[c*9+idx] = V[i,j][c]
    for i in range(-1, 2):
        for j in range(-1, 2):
            idx = (i + 1) * 3 + (j + 1)
            b[np.arange(C) * 9 + idx] = Vvec[(i, j)]

    # Now build augmented system for weights and bias
    # We need S_off_flat: length P, where S_off_flat[c*9+idx] = Soff[i,j][c]
    S_off_flat = np.zeros(P, dtype=np.float64)
    for i in range(-1, 2):
        for j in range(-1, 2):
            idx = (i + 1) * 3 + (j + 1)
            S_off_flat[np.arange(C) * 9 + idx] = Soff[(i, j)]

    # Augmented matrix: [[A, S_off_flat], [S_off_flat^T, M]]
    Aug = np.zeros((P + 1, P + 1), dtype=np.float64)
    Aug[:P, :P] = A
    Aug[:P, P] = S_off_flat
    Aug[P, :P] = S_off_flat
    Aug[P, P] = M

    # Augmented RHS: [b, sum_y]
    rhs = np.zeros(P + 1, dtype=np.float64)
    rhs[:P] = b
    rhs[P] = sum_y

    # Add tiny ridge for numerical stability
    ridge = 1e-8
    Aug[:P, :P] += np.eye(P) * ridge

    # Solve
    sol = np.linalg.solve(Aug, rhs)
    w_out = sol[:P].reshape(1, C, K, K)  # (1, 64, 3, 3)
    b_out = sol[P]

    # Update weights
    w['out.weight'] = w_out.astype(np.float32)
    w['out.bias'] = np.array([b_out], dtype=np.float32)

    # Compute resulting MSE on the full 2000 examples
    # We can compute it efficiently using the same Gram matrix
    # MSE = (1/M) * (y^T y - 2 * w^T b - 2 * b_out * sum_y + w^T A w + 2 * b_out * w^T S_off_flat + b_out^2 * M)
    # Actually, we can just compute the prediction error using the new weights on the features.
    # But to avoid recomputing the full forward, we can use the normal equations.
    # Let's compute y^T y
    yTy = (y_flat ** 2).sum()
    # w^T A w
    w_vec = w_out.reshape(-1)
    wAw = w_vec @ A @ w_vec
    # w^T b
    wb = w_vec @ b
    # w^T S_off_flat
    wS = w_vec @ S_off_flat
    # MSE = (1/M) * (yTy - 2*wb - 2*b_out*sum_y + wAw + 2*b_out*wS + b_out^2 * M)
    mse = (yTy - 2 * wb - 2 * b_out * sum_y + wAw + 2 * b_out * wS + b_out**2 * M) / M
    mse = float(mse)

    diagnostics = {
        "mse": mse,
        "out_weight_norm": float(np.linalg.norm(w_out)),
        "out_bias": float(b_out),
        "ridge": ridge,
        "n_params": P + 1
    }

    return {"weights": w, "diagnostics": diagnostics}
