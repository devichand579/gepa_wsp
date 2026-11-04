import numpy as np
import math

def edit(w, f):
    # Extract features and targets
    F = f['features'].astype(np.float32)          # (N, 64, 16, 16)
    Y = f['y'].astype(np.float32)                 # (N, 1, 16, 16)
    N, C, H, W = F.shape
    assert C == 64 and H == 16 and W == 16

    # We will fit out.weight (1,64,3,3) and out.bias (1)
    # Design: for each output pixel (i,j), prediction = sum_{c,ki,kj} w[c,ki,kj] * F[c, i+ki-1, j+kj-1] + b
    # We solve normal equations for the 577 parameters (576 weights + 1 bias).

    # Number of parameters
    P = C * 9 + 1  # 577
    G = np.zeros((P, P), dtype=np.float64)
    b = np.zeros(P, dtype=np.float64)

    # Precompute sum of features for bias column
    # For bias, the design column is all ones.
    # G[bias, bias] = number of output pixels = N*H*W
    # G[bias, (c,ki,kj)] = sum_{n,i,j} F[n,c, i+ki-1, j+kj-1] = sum_{n} sum_{p,q} F[n,c,p,q] = sum over all positions of F[c]
    # b[bias] = sum_{n,i,j} Y[n,0,i,j] = sum of all Y

    # Compute total sum of features per channel
    sum_F = F.sum(axis=(0, 2, 3))  # (C,)
    sum_Y = Y.sum()

    # Fill bias-related entries
    G[0, 0] = N * H * W
    for c in range(C):
        # index of (c, ki, kj) is 1 + c*9 + ki*3 + kj
        for ki in range(3):
            for kj in range(3):
                idx = 1 + c * 9 + ki * 3 + kj
                G[0, idx] = sum_F[c]
                G[idx, 0] = sum_F[c]
    b[0] = sum_Y

    # Now compute the weight-weight Gram blocks.
    # For each relative shift (dki, dkj) in -2..2, compute C_shift[c,c'] = sum_n sum_{p,q} F[n,c,p,q] * F[n,c', p+dki, q+dkj]
    # Then assign to G[(c,ki,kj),(c',ki',kj')] where ki'-ki = dki, kj'-kj = dkj.

    # We'll iterate over shifts.
    for dki in range(-2, 3):
        for dkj in range(-2, 3):
            # Determine valid slices for p and q
            if dki >= 0:
                p1_start, p1_end = 0, H - dki
                p2_start, p2_end = dki, H
            else:
                p1_start, p1_end = -dki, H
                p2_start, p2_end = 0, H + dki
            if dkj >= 0:
                q1_start, q1_end = 0, W - dkj
                q2_start, q2_end = dkj, W
            else:
                q1_start, q1_end = -dkj, W
                q2_start, q2_end = 0, W + dkj

            # Extract A and B
            A = F[:, :, p1_start:p1_end, q1_start:q1_end]  # (N, C, H', W')
            B = F[:, :, p2_start:p2_end, q2_start:q2_end]  # (N, C, H', W')
            # Reshape to (N, C, -1)
            A_flat = A.reshape(N, C, -1)
            B_flat = B.reshape(N, C, -1)
            # Compute C_shift = sum_n A_n @ B_n^T
            # We can do this by transposing to (C, N*H'*W') and doing a matrix multiply
            A_mat = A_flat.transpose(1, 0, 2).reshape(C, -1)  # (C, N*H'*W')
            B_mat = B_flat.transpose(1, 0, 2).reshape(C, -1)  # (C, N*H'*W')
            C_shift = A_mat @ B_mat.T  # (C, C)

            # Assign to G
            for c in range(C):
                for c2 in range(C):
                    val = C_shift[c, c2]
                    if val == 0.0:
                        continue
                    # For all ki, kj, ki', kj' such that ki'-ki = dki, kj'-kj = dkj
                    for ki in range(3):
                        ki2 = ki + dki
                        if ki2 < 0 or ki2 > 2:
                            continue
                        for kj in range(3):
                            kj2 = kj + dkj
                            if kj2 < 0 or kj2 > 2:
                                continue
                            idx1 = 1 + c * 9 + ki * 3 + kj
                            idx2 = 1 + c2 * 9 + ki2 * 3 + kj2
                            G[idx1, idx2] = val

    # Now compute b for weight columns: b[(c,ki,kj)] = sum_{n,i,j} F[n,c, i+ki-1, j+kj-1] * Y[n,0,i,j]
    # This is cross-correlation of F[c] with Y at shift (ki-1, kj-1).
    # For each (c,ki,kj), we need sum_{n} sum_{i,j} F[n,c, i+ki-1, j+kj-1] * Y[n,0,i,j]
    # Let p = i+ki-1, q = j+kj-1. Then i = p - ki + 1, j = q - kj + 1.
    # So sum_{p,q} F[n,c,p,q] * Y[n,0, p - ki + 1, q - kj + 1] (with zero padding for Y)
    # This is a correlation of F with Y at shift (ki-1, kj-1) but with Y shifted in the opposite direction.
    # We can compute this by iterating over (ki,kj) and doing a matrix multiply similar to above.
    # For a given (ki,kj), the shift for Y is (ki-1, kj-1). Let's define dki_y = ki-1, dkj_y = kj-1.
    # We want sum_n sum_{p,q} F[n,c,p,q] * Y[n,0, p + dki_y, q + dkj_y] where Y is zero-padded.
    # This is exactly the cross-correlation of F and Y at shift (dki_y, dkj_y).
    # We can compute it for all (c,ki,kj) by iterating over (ki,kj) and doing a matrix multiply.

    # We'll compute b for each (c,ki,kj).
    # To be efficient, we can loop over (ki,kj) and compute the vector over c.
    for ki in range(3):
        for kj in range(3):
            dki_y = ki - 1
            dkj_y = kj - 1
            # Determine valid slices for F and Y
            if dki_y >= 0:
                p1_start, p1_end = 0, H - dki_y
                p2_start, p2_end = dki_y, H
            else:
                p1_start, p1_end = -dki_y, H
                p2_start, p2_end = 0, H + dki_y
            if dkj_y >= 0:
                q1_start, q1_end = 0, W - dkj_y
                q2_start, q2_end = dkj_y, W
            else:
                q1_start, q1_end = -dkj_y, W
                q2_start, q2_end = 0, W + dkj_y

            # F part: (N, C, H', W')
            F_sub = F[:, :, p1_start:p1_end, q1_start:q1_end]
            # Y part: (N, 1, H', W')
            Y_sub = Y[:, :, p2_start:p2_end, q2_start:q2_end]

            # Reshape
            F_flat = F_sub.reshape(N, C, -1)  # (N, C, M)
            Y_flat = Y_sub.reshape(N, 1, -1)  # (N, 1, M)

            # We want sum_n F_n @ Y_n^T -> (C, 1)
            # Use matrix multiply: F_mat (C, N*M) @ Y_mat (N*M, 1)
            F_mat = F_flat.transpose(1, 0, 2).reshape(C, -1)  # (C, N*M)
            Y_mat = Y_flat.transpose(1, 0, 2).reshape(1, -1).T  # (N*M, 1)
            b_vec = F_mat @ Y_mat  # (C, 1)

            # Assign to b
            for c in range(C):
                idx = 1 + c * 9 + ki * 3 + kj
                b[idx] = b_vec[c, 0]

    # Solve the linear system G w = b
    # Add small ridge for numerical stability (though data is plenty)
    ridge = 1e-6 * np.trace(G) / P
    G_reg = G + ridge * np.eye(P)
    try:
        w_sol = np.linalg.solve(G_reg, b)
    except np.linalg.LinAlgError:
        w_sol = np.linalg.lstsq(G_reg, b, rcond=None)[0]

    # Extract weights and bias
    out_weight = np.zeros((1, C, 3, 3), dtype=np.float32)
    for c in range(C):
        for ki in range(3):
            for kj in range(3):
                idx = 1 + c * 9 + ki * 3 + kj
                out_weight[0, c, ki, kj] = w_sol[idx]
    out_bias = np.array([w_sol[0]], dtype=np.float32)

    # Update w
    w['out.weight'] = out_weight
    w['out.bias'] = out_bias

    # Diagnostics
    diagnostics = {
        'mse_before': float(np.mean((f['prediction'] - Y) ** 2)),
        'mse_after_estimate': float(np.mean((np.einsum('ncij,ckl->nij', F, out_weight[0]) + out_bias[0] - Y[:, 0]) ** 2)) if False else None,
        'ridge': float(ridge),
        'out_weight_norm': float(np.linalg.norm(out_weight)),
        'out_bias': float(out_bias[0])
    }

    return {"weights": w, "diagnostics": diagnostics}
