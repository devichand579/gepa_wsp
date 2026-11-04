import numpy as np
import math

def edit(w, f):
    # Extract features and targets
    F = f['features']  # (2000, 64, 16, 16)
    Y = f['y']         # (2000, 1, 16, 16)
    N, C, H, W = F.shape
    assert C == 64 and H == 16 and W == 16
    assert Y.shape == (N, 1, 16, 16)
    
    # Flatten targets: (N*H*W, 1)
    y_flat = Y.transpose(0, 2, 3, 1).reshape(-1, 1)  # (N*H*W, 1)
    
    # We need to compute A = X^T X (577x577) and b = X^T y (577)
    # X columns: 576 for (channel, di, dj) where di,dj in 0..2, plus 1 for bias.
    # We can compute A efficiently using cross-correlations of the feature maps.
    # Let F be reshaped to (N, 16, 16, 64) for easier spatial slicing.
    F_resh = F.transpose(0, 2, 3, 1)  # (N, 16, 16, 64)
    
    # We will compute the 25 cross-correlation matrices for offsets (di, dj) in -2..2.
    # For offset (di, dj), we want sum_{n,i,j} F[n,i,j,c1] * F[n,i+di,j+dj,c2]
    # with zero padding. The valid i range: max(0,-di) to min(15, 15-di)
    # and similarly for j.
    # We'll compute a 64x64 matrix for each offset.
    # To do this efficiently, we can extract the valid slices and compute A^T @ B.
    
    # Pre-allocate the 576x576 matrix A and 576-vector b.
    # Actually, we can build A from the 25 offset matrices.
    # The index for (c, di, dj) in the 576-dim vector: idx = c*9 + di*3 + dj.
    # For two indices (c1, di1, dj1) and (c2, di2, dj2), the entry is
    # sum_{n,i,j} F[n,i+di1-1,j+dj1-1,c1] * F[n,i+di2-1,j+dj2-1,c2]
    # Let i' = i+di1-1, j' = j+dj1-1. Then i = i' - di1 + 1, j = j' - dj1 + 1.
    # The sum becomes sum_{n,i',j'} F[n,i',j',c1] * F[n,i'+ (di2-di1), j'+ (dj2-dj1), c2]
    # over the valid range where both indices are in [0,15].
    # This is exactly the cross-correlation at offset (di2-di1, dj2-dj1).
    # So we can compute the 25 cross-correlation matrices for offsets d = (di2-di1, dj2-dj1) in -2..2.
    # Let's compute these matrices.
    
    # We'll compute cross_corr[d_i, d_j] = 64x64 matrix for offset (d_i, d_j) in -2..2.
    # We can map d_i, d_j to indices 0..4.
    cross_corr = np.zeros((5, 5, C, C), dtype=np.float64)
    
    # For each offset (d_i, d_j) in -2..2
    for d_i in range(-2, 3):
        for d_j in range(-2, 3):
            # Valid i range for the first index: i such that i and i+d_i are in [0,15]
            # i in [max(0, -d_i), min(15, 15-d_i)]
            i_start = max(0, -d_i)
            i_end = min(15, 15 - d_i)
            j_start = max(0, -d_j)
            j_end = min(15, 15 - d_j)
            if i_start > i_end or j_start > j_end:
                continue
            # Extract the two slices
            # A: F_resh[:, i_start:i_end+1, j_start:j_end+1, :] -> shape (N, ni, nj, C)
            # B: F_resh[:, i_start+d_i:i_end+d_i+1, j_start+d_j:j_end+d_j+1, :] -> shape (N, ni, nj, C)
            # We want sum over n, i, j of A[n,i,j,c1] * B[n,i,j,c2] = A_flat^T @ B_flat
            A_slice = F_resh[:, i_start:i_end+1, j_start:j_end+1, :]  # (N, ni, nj, C)
            B_slice = F_resh[:, i_start+d_i:i_end+d_i+1, j_start+d_j:j_end+d_j+1, :]  # (N, ni, nj, C)
            # Reshape to (N*ni*nj, C)
            A_flat = A_slice.reshape(-1, C)
            B_flat = B_slice.reshape(-1, C)
            # Compute cross-correlation matrix: (C, C)
            # We want sum over rows of A_flat[:, c1] * B_flat[:, c2]
            # This is A_flat.T @ B_flat
            cross_corr[d_i+2, d_j+2] = A_flat.T @ B_flat
    
    # Now build the 576x576 matrix A and 576-vector b for the linear system.
    # The 576 dimensions correspond to (c, di, dj) with di,dj in 0..2.
    # We can map (c, di, dj) to index idx = c*9 + di*3 + dj.
    # For two indices (c1, di1, dj1) and (c2, di2, dj2), the entry is
    # cross_corr[di2-di1+2, dj2-dj1+2, c1, c2].
    # We can build A by looping over di1,dj1,di2,dj2 and assigning the 64x64 block.
    A_mat = np.zeros((576, 576), dtype=np.float64)
    for di1 in range(3):
        for dj1 in range(3):
            for di2 in range(3):
                for dj2 in range(3):
                    d_i = di2 - di1
                    d_j = dj2 - dj1
                    # d_i, d_j in -2..2
                    block = cross_corr[d_i+2, d_j+2]  # (64, 64)
                    # Place block at rows for (c1, di1, dj1) and cols for (c2, di2, dj2)
                    # Row index: c1*9 + di1*3 + dj1
                    # Col index: c2*9 + di2*3 + dj2
                    # We can use slicing: rows = slice(di1*3+dj1, 576, 9)? No, the index is c*9 + offset.
                    # So for a fixed (di1,dj1), the rows are c*9 + di1*3 + dj1 for c=0..63.
                    # That's not a contiguous block. We can use advanced indexing or just loop over c1,c2.
                    # Since 64x64 is small, we can do a double loop or use np.add.at.
                    # But we can also reshape A_mat to (64, 9, 64, 9) and assign.
                    # Let's reshape A_mat to (64, 3, 3, 64, 3, 3) for easier assignment.
                    # Actually, we can just build A_mat by iterating over c1 and c2? That's 64*64=4096 iterations, each assigning a 9x9 block.
                    # Better: reshape A_mat to (64, 9, 64, 9) and assign block to A_mat[c1, di1*3+dj1, c2, di2*3+dj2] for all c1,c2.
                    # We can do this with broadcasting.
                    # Let's do it efficiently:
                    # A_mat_reshaped = A_mat.reshape(64, 9, 64, 9)
                    # A_mat_reshaped[:, di1*3+dj1, :, di2*3+dj2] = block
                    # This is a 64x64 assignment.
                    pass
    
    # Let's build A_mat using the reshaped view.
    A_mat = np.zeros((64, 9, 64, 9), dtype=np.float64)
    for di1 in range(3):
        for dj1 in range(3):
            for di2 in range(3):
                for dj2 in range(3):
                    d_i = di2 - di1
                    d_j = dj2 - dj1
                    block = cross_corr[d_i+2, d_j+2]  # (64, 64)
                    A_mat[:, di1*3+dj1, :, di2*3+dj2] = block
    A_mat = A_mat.reshape(576, 576)
    
    # Now compute b = X^T y. We need sum_{n,i,j} F[n,i+di-1,j+dj-1,c] * y[n,i,j]
    # This is a cross-correlation between F and y at offset (di-1, dj-1).
    # For each (c, di, dj), we want sum_{n,i,j} F[n,i+di-1,j+dj-1,c] * y[n,i,j].
    # Let's compute this by looping over di,dj.
    b_vec = np.zeros(576, dtype=np.float64)
    # y_flat is (N*H*W, 1). We can reshape y to (N, 16, 16, 1) for spatial slicing.
    y_resh = Y.transpose(0, 2, 3, 1)  # (N, 16, 16, 1)
    for di in range(3):
        for dj in range(3):
            # We want sum_{n,i,j} F[n,i+di-1,j+dj-1,c] * y[n,i,j]
            # Let i' = i+di-1, j' = j+dj-1. Then i = i' - di + 1, j = j' - dj + 1.
            # The sum is over i,j such that i+di-1 in [0,15] and j+dj-1 in [0,15].
            # So i in [max(0, 1-di), min(15, 16-di)]? Let's just use the valid range.
            # We can extract the slice of F that aligns with y.
            # For a given di, the valid i for F is i_f = i + di - 1.
            # We want sum over i,j of F[n, i+di-1, j+dj-1, c] * y[n, i, j].
            # Let's find the overlapping region.
            # i ranges from 0 to 15. i_f = i + di - 1.
            # i_f must be in [0,15]. So i must be in [max(0, 1-di), min(15, 16-di)].
            i_start = max(0, 1 - di)
            i_end = min(15, 16 - di)
            j_start = max(0, 1 - dj)
            j_end = min(15, 16 - dj)
            if i_start > i_end or j_start > j_end:
                continue
            # F slice: F_resh[:, i_start+di-1 : i_end+di-1+1, j_start+dj-1 : j_end+dj-1+1, :]
            # y slice: y_resh[:, i_start:i_end+1, j_start:j_end+1, :]
            F_slice = F_resh[:, i_start+di-1 : i_end+di, j_start+dj-1 : j_end+dj, :]  # (N, ni, nj, 64)
            y_slice = y_resh[:, i_start:i_end+1, j_start:j_end+1, :]  # (N, ni, nj, 1)
            # Reshape to (N*ni*nj, 64) and (N*ni*nj, 1)
            F_flat = F_slice.reshape(-1, 64)
            y_flat_slice = y_slice.reshape(-1, 1)
            # Compute b for this (di,dj): sum over rows of F_flat[:, c] * y_flat_slice
            b_block = F_flat.T @ y_flat_slice  # (64, 1)
            # Place into b_vec at indices c*9 + di*3 + dj
            # We can do: b_vec[di*3+dj::9] = b_block.flatten()
            # Because index = c*9 + di*3 + dj, so for c=0..63, the indices are di*3+dj, di*3+dj+9, ...
            b_vec[di*3+dj::9] = b_block.flatten()
    
    # Add bias column: we need to augment A and b with the bias term.
    # The design matrix X has an extra column of ones. So A becomes 577x577, b becomes 577.
    # A_bias = [[A, sum(F)], [sum(F)^T, N*H*W]]
    # where sum(F) is the sum over all positions of the 576 features.
    # We can compute sum_F = sum over n,i,j of the 3x3 patches. That is the sum of all features in the 3x3 neighborhoods.
    # Actually, the bias column is just 1. So the cross terms are sum_{n,i,j} F[n,i+di-1,j+dj-1,c] * 1 = sum over all valid positions of F.
    # This is simply the sum of F over all positions, but for each (c,di,dj), it's the sum of F[c] over the valid region.
    # We can compute sum_F_vec of length 576.
    sum_F_vec = np.zeros(576, dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            i_start = max(0, 1 - di)
            i_end = min(15, 16 - di)
            j_start = max(0, 1 - dj)
            j_end = min(15, 16 - dj)
            if i_start > i_end or j_start > j_end:
                continue
            F_slice = F_resh[:, i_start+di-1 : i_end+di, j_start+dj-1 : j_end+dj, :]  # (N, ni, nj, 64)
            sum_F = F_slice.sum(axis=(0,1,2))  # (64,)
            sum_F_vec[di*3+dj::9] = sum_F
    
    # Total number of equations: N*H*W = 2000*256 = 512000
    n_equations = N * H * W
    
    # Build the 577x577 matrix
    A_full = np.zeros((577, 577), dtype=np.float64)
    A_full[:576, :576] = A_mat
    A_full[:576, 576] = sum_F_vec
    A_full[576, :576] = sum_F_vec
    A_full[576, 576] = n_equations
    
    # Build the 577-vector b
    b_full = np.zeros(577, dtype=np.float64)
    b_full[:576] = b_vec
    b_full[576] = y_flat.sum()
    
    # Solve with a small ridge for numerical stability
    ridge = 1e-4
    A_full += ridge * np.eye(577)
    
    # Solve linear system
    try:
        sol = np.linalg.solve(A_full, b_full)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A_full, b_full, rcond=None)[0]
    
    # Extract weights and bias
    out_weight = sol[:576].reshape(64, 3, 3)  # (64, 3, 3)
    out_bias = sol[576]
    
    # The out layer weight shape is (1, 64, 3, 3). We need to match the original shape.
    # The original out.weight is (1, 64, 3, 3). Our out_weight is (64, 3, 3).
    # We need to reshape to (1, 64, 3, 3).
    new_out_weight = out_weight.reshape(1, 64, 3, 3).astype(np.float32)
    new_out_bias = np.array([out_bias], dtype=np.float32)
    
    # Update the weights dictionary
    w['out.weight'] = new_out_weight
    w['out.bias'] = new_out_bias
    
    # Compute diagnostics: training MSE after update
    # We can compute the new prediction using the features and new out layer.
    # Since we solved the normal equations, the MSE should be the minimum for the current features.
    # We can compute it directly: MSE = (1/(N*H*W)) * sum (y - X w)^2
    # We can compute the residual sum of squares: y^T y - 2 w^T X^T y + w^T X^T X w
    # But we already have A_full and b_full. The optimal w solves A w = b.
    # The residual sum of squares = y^T y - w^T b (since w^T A w = w^T b).
    # Actually, for least squares with intercept, the residual sum of squares is y^T y - w^T b.
    # Let's compute y^T y.
    y_sq_sum = (y_flat ** 2).sum()
    # w^T b = sol^T b_full
    wTb = np.dot(sol, b_full)
    rss = y_sq_sum - wTb
    mse = rss / n_equations
    
    # Also compute the MSE using the original prediction for comparison
    # f['prediction'] is the current prediction. We can compute its MSE.
    pred = f['prediction']
    mse_before = ((pred - Y) ** 2).mean()
    
    diagnostics = {
        'mse_before': float(mse_before),
        'mse_after_estimate': float(mse),
        'ridge': float(ridge),
        'out_weight_norm': float(np.linalg.norm(new_out_weight)),
        'out_bias': float(new_out_bias[0]),
        'n_params': 577
    }
    
    return {"weights": w, "diagnostics": diagnostics}
