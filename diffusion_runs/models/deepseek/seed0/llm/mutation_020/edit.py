import numpy as np

def edit(w, f):
    # ---- Extract full training features and targets ----
    F = f['features']          # [2000, 64, 16, 16]
    y = f['y']                 # [2000, 1, 16, 16]
    N, C, H, W = F.shape

    # ---- Build normal equations for the final linear layer ----
    # Columns: 64 channels * 9 kernel positions = 576, plus 1 bias = 577
    n_cols = C * 9 + 1
    ATA = np.zeros((n_cols, n_cols), dtype=np.float64)
    ATy = np.zeros(n_cols, dtype=np.float64)

    chunk_size = 200
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        F_chunk = F[start:end]          # [chunk, 64, 16, 16]
        y_chunk = y[start:end]          # [chunk, 1, 16, 16]
        n_chunk = F_chunk.shape[0]

        # Pad spatial dims by 1 (zero padding)
        F_pad = np.pad(F_chunk, ((0,0),(0,0),(1,1),(1,1)), mode='constant')

        # Build design matrix for this chunk
        blocks = []
        for di in range(3):
            for dj in range(3):
                # F_pad[:, :, di:di+H, dj:dj+W] -> [chunk, C, H, W]
                block = F_pad[:, :, di:di+H, dj:dj+W]
                # -> [chunk, H, W, C] -> [chunk*H*W, C]
                block = block.transpose(0, 2, 3, 1).reshape(-1, C)
                blocks.append(block)
        A_chunk = np.concatenate(blocks, axis=1)          # [chunk*H*W, 576]
        ones = np.ones((A_chunk.shape[0], 1), dtype=np.float64)
        A_chunk = np.concatenate([A_chunk, ones], axis=1) # [chunk*H*W, 577]

        y_flat = y_chunk.transpose(0, 2, 3, 1).reshape(-1) # [chunk*H*W]

        ATA += A_chunk.T @ A_chunk
        ATy += A_chunk.T @ y_flat

    # Small ridge for numerical stability
    ridge = 1e-6
    ATA += ridge * np.eye(n_cols)
    solution = np.linalg.solve(ATA, ATy)

    # ---- Extract weights and bias ----
    W_flat = solution[:C*9]
    # solution order: (di,dj) outer, c inner -> reshape to (3,3,C) then transpose
    W_new = W_flat.reshape(3, 3, C).transpose(2, 0, 1)  # [C, 3, 3]
    out_weight = W_new.reshape(1, C, 3, 3).astype(np.float32)
    out_bias = solution[C*9:C*9+1].astype(np.float32)

    # ---- Diagnostics: MSE on the paired 128-example batch ----
    batch_features = f['layer.out.input0']   # [128, 64, 16, 16]
    batch_y = f['batch.y']                   # [128, 1, 16, 16]

    def apply_out(F, W, b):
        N, C, H, Wd = F.shape
        F_pad = np.pad(F, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        out = np.zeros((N, 1, H, Wd), dtype=np.float32)
        for di in range(3):
            for dj in range(3):
                out[:, 0] += np.tensordot(
                    F_pad[:, :, di:di+H, dj:dj+Wd],
                    W[0, :, di, dj],
                    axes=([1], [0])
                )
        out += b[0]
        return out

    # Current batch MSE (before update)
    pred_batch_old = apply_out(batch_features, w['out.weight'], w['out.bias'])
    mse_batch_old = float(np.mean((pred_batch_old - batch_y)**2))

    # New batch MSE
    pred_batch_new = apply_out(batch_features, out_weight, out_bias)
    mse_batch_new = float(np.mean((pred_batch_new - batch_y)**2))

    # ---- Update weights ----
    w['out.weight'] = out_weight
    w['out.bias'] = out_bias

    diagnostics = {
        'mse_batch_old': mse_batch_old,
        'mse_batch_new': mse_batch_new,
        'out_weight_norm': float(np.linalg.norm(out_weight)),
        'out_bias': float(out_bias[0]),
    }

    return {"weights": w, "diagnostics": diagnostics}
