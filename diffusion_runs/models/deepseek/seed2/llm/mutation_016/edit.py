import numpy as np
import math

def edit(w, f):
    # Extract arrays
    feat = f["features"]          # (2000, 64, 16, 16)
    y = f["y"]                    # (2000, 1, 16, 16)
    N, C, H, W = feat.shape
    y = y[:, 0, :, :]             # (2000, 16, 16)

    # We will solve for out.weight (1,64,3,3) and out.bias (1,)
    # Flatten out.weight to 576 vector: order c, di, dj -> index c*9 + di*3 + dj
    # But for normal equations we use block structure: 9 blocks of 64.
    # Let k = di*3 + dj, block k is 64-vector w_k[c] = out.weight[0,c,di,dj]
    # We need to compute:
    #   C[dx,dy][c,c'] = sum_{n,i,j} f[n,c,i,j] * f[n,c',i+dx,j+dy]  for dx,dy in -2..2
    #   Corr_fy[dx,dy][c] = sum_{n,i,j} f[n,c,i,j] * y[n,i+dx,j+dy]  for dx,dy in -1..1
    #   S[di,dj][c] = sum_{n,i,j} f[n,c,i+di-1,j+dj-1]  (sum over valid region)
    #   sum_y = sum_{n,i,j} y[n,i,j]
    # Then assemble A (577x577) and rhs (577).

    # Precompute shifted feature sums for S
    # For each di,dj in 0..2, the valid u range is max(0,di-1)..min(15,di+14)
    # and v range max(0,dj-1)..min(15,dj+14)
    S = np.zeros((9, C), dtype=np.float64)
    for di in range(3):
        u0 = max(0, di - 1)
        u1 = min(H - 1, di + 14)
        for dj in range(3):
            v0 = max(0, dj - 1)
            v1 = min(W - 1, dj + 14)
            k = di * 3 + dj
            # sum over n, c, u, v in the slice
            S[k] = feat[:, :, u0:u1+1, v0:v1+1].sum(axis=(0, 2, 3))

    sum_y = float(y.sum())

    # Compute C[dx,dy] for dx,dy in -2..2
    # We'll store in a dict keyed by (dx,dy)
    C = {}
    # To save memory, process in chunks over N
    chunk_size = 100
    n_chunks = (N + chunk_size - 1) // chunk_size

    # Precompute all needed shifts for C: dx,dy in -2..2
    shifts_C = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]
    # Initialize accumulators
    C_acc = {s: np.zeros((C, C), dtype=np.float64) for s in shifts_C}

    # Precompute Corr_fy shifts: dx,dy in -1..1
    shifts_fy = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]
    fy_acc = {s: np.zeros(C, dtype=np.float64) for s in shifts_fy}

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(N, start + chunk_size)
        f_chunk = feat[start:end]          # (B, C, H, W)
        y_chunk = y[start:end]             # (B, H, W)
        B = f_chunk.shape[0]

        # For C: we need f_shift for each (dx,dy)
        # We can compute f_flat = f_chunk.transpose(0,2,3,1).reshape(-1, C)  # (B*H*W, C)
        f_flat = f_chunk.transpose(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        for (dx, dy) in shifts_C:
            # Create shifted version with zero padding
            # We want f_shift[n,c,i,j] = f[n,c,i+dx,j+dy] if in bounds else 0
            # We can build an array of shape (B, C, H, W) with zeros, then fill
            f_shift = np.zeros_like(f_chunk)
            # Determine valid i range for source: i+dx in [0,H-1] => i in [-dx, H-1-dx]
            i0 = max(0, -dx)
            i1 = min(H, H - dx)
            j0 = max(0, -dy)
            j1 = min(W, W - dy)
            if i0 < i1 and j0 < j1:
                # source indices: i+dx, j+dy
                # target indices: i, j
                f_shift[:, :, i0:i1, j0:j1] = f_chunk[:, :, i0+dx:i1+dx, j0+dy:j1+dy]
            f_shift_flat = f_shift.transpose(0, 2, 3, 1).reshape(-1, C)
            # Accumulate C += f_flat.T @ f_shift_flat
            C_acc[(dx, dy)] += f_flat.T @ f_shift_flat

        # For Corr_fy: we need y_shift for each (dx,dy)
        for (dx, dy) in shifts_fy:
            y_shift = np.zeros_like(y_chunk)  # (B, H, W)
            i0 = max(0, -dx)
            i1 = min(H, H - dx)
            j0 = max(0, -dy)
            j1 = min(W, W - dy)
            if i0 < i1 and j0 < j1:
                y_shift[:, i0:i1, j0:j1] = y_chunk[:, i0+dx:i1+dx, j0+dy:j1+dy]
            y_shift_flat = y_shift.reshape(-1)  # (B*H*W,)
            # Accumulate fy += f_flat.T @ y_shift_flat
            fy_acc[(dx, dy)] += f_flat.T @ y_shift_flat

    # Now assemble A (577x577) and rhs (577)
    # Order: w_k[c] for k=0..8, c=0..63, then bias.
    # A is 9x9 blocks of 64x64, plus bias row/col.
    dim = 9 * C + 1
    A = np.zeros((dim, dim), dtype=np.float64)
    rhs = np.zeros(dim, dtype=np.float64)

    # Fill w-w blocks
    for k1 in range(9):
        di1 = k1 // 3
        dj1 = k1 % 3
        for k2 in range(9):
            di2 = k2 // 3
            dj2 = k2 % 3
            dx = di2 - di1
            dy = dj2 - dj1
            # block (k1,k2) = C[dx,dy]
            # Note: C[dx,dy][c,c'] = sum f[c] * f[c'] shifted by (dx,dy)
            # Our C_acc[(dx,dy)] is exactly that.
            block = C_acc[(dx, dy)]
            A[k1*C:(k1+1)*C, k2*C:(k2+1)*C] = block

    # Fill bias row/col
    for k in range(9):
        di = k // 3
        dj = k % 3
        # bias row: A[9*C, k*C:(k+1)*C] = S[k]
        A[9*C, k*C:(k+1)*C] = S[k]
        # bias col: A[k*C:(k+1)*C, 9*C] = S[k]
        A[k*C:(k+1)*C, 9*C] = S[k]
    A[9*C, 9*C] = float(N * H * W)

    # Fill rhs for w
    for k in range(9):
        di = k // 3
        dj = k % 3
        # rhs block k = Corr_fy[1-di, 1-dj]
        dx = 1 - di
        dy = 1 - dj
        rhs[k*C:(k+1)*C] = fy_acc[(dx, dy)]
    rhs[9*C] = sum_y

    # Add tiny ridge for numerical stability
    ridge = 1e-6
    A += np.eye(dim) * ridge

    # Solve linear system
    sol = np.linalg.solve(A, rhs)

    # Extract w and b
    w_new = sol[:9*C].reshape(9, C)  # (9, C)
    b_new = sol[9*C]

    # Map back to out.weight shape (1,64,3,3)
    out_weight = np.zeros((1, C, 3, 3), dtype=np.float32)
    for k in range(9):
        di = k // 3
        dj = k % 3
        out_weight[0, :, di, dj] = w_new[k]

    # Update weights
    w["out.weight"] = out_weight
    w["out.bias"] = np.array([b_new], dtype=np.float32)

    # Diagnostics
    # Compute new prediction on a small subset to report MSE? Not necessary, but we can compute training MSE on the fly? 
    # We can compute the MSE of the LS solution on the full training set using the normal equations? 
    # Actually we can compute the residual sum of squares: ||y - Xw - b||^2 = y^T y - 2 w^T X^T y - 2 b 1^T y + w^T X^T X w + 2 b 1^T X w + M b^2
    # But we don't have y^T y. We can compute it from y.
    # For diagnostics, just return small stats.
    diagnostics = {
        "out_weight_max": float(np.max(np.abs(out_weight))),
        "out_weight_mean": float(np.mean(out_weight)),
        "out_bias": float(b_new),
        "ridge": ridge,
        "n_samples": int(N),
    }

    return {"weights": w, "diagnostics": diagnostics}
