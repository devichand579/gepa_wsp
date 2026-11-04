import numpy as np
import numpy.lib.stride_tricks as stride_tricks

def edit(w, f):
    # Extract cached features (input to out layer) and targets
    F = f['features']          # (2000, 64, 16, 16)
    Y = f['y']                 # (2000, 1, 16, 16)
    N, C, H, W = F.shape
    assert C == 64 and H == 16 and W == 16

    # We solve for out.weight (1,64,3,3) and out.bias (1)
    # Design matrix columns: (c, di, dj) for c=0..63, di,dj in {-1,0,1}, plus bias.
    # Total columns = 64*9 + 1 = 577
    n_cols = C * 9 + 1
    XtX = np.zeros((n_cols, n_cols), dtype=np.float64)
    XtY = np.zeros((n_cols, 1), dtype=np.float64)

    # Process in chunks to keep memory low
    chunk_size = 256
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Fc = F[start:end]      # (chunk, 64, 16, 16)
        Yc = Y[start:end]      # (chunk, 1, 16, 16)
        n_chunk = end - start

        # Pad features with 1 pixel border (zero padding)
        F_pad = np.pad(Fc, ((0,0),(0,0),(1,1),(1,1)), mode='constant')

        # Extract 3x3 patches: shape (chunk, 64, 16, 16, 3, 3)
        patches = stride_tricks.sliding_window_view(F_pad, (3,3), axis=(2,3))
        # Rearrange to (chunk, 16, 16, 64, 3, 3) then flatten to (chunk*256, 576)
        patches = patches.transpose(0, 2, 3, 1, 4, 5)
        X_chunk = patches.reshape(-1, C * 9)   # (chunk*256, 576)

        # Add bias column
        X_chunk = np.concatenate([X_chunk, np.ones((X_chunk.shape[0], 1), dtype=X_chunk.dtype)], axis=1)

        # Flatten targets: (chunk*256, 1)
        Y_chunk = Yc.transpose(0, 2, 3, 1).reshape(-1, 1)

        # Accumulate normal equations
        XtX += X_chunk.T @ X_chunk
        XtY += X_chunk.T @ Y_chunk

    # Solve normal equations (with tiny ridge for numerical stability)
    ridge = 1e-6 * np.eye(n_cols)
    W = np.linalg.solve(XtX + ridge, XtY)

    # Extract weights and bias
    W_conv = W[:C*9].reshape(1, C, 3, 3)
    b_out = W[C*9:].reshape(1)

    # Update the weights dictionary
    w['out.weight'] = W_conv.astype(np.float32)
    w['out.bias'] = b_out.astype(np.float32)

    # Diagnostics
    pred = np.einsum('nchw,oc->nohw', F, W_conv.reshape(1, C, 3, 3))  # not exact due to padding, but we can compute MSE on the batch
    # Actually compute MSE on the provided batch for diagnostics
    batch_F = f['layer.out.input0']  # (128, 64, 16, 16)
    batch_Y = f['batch.y']           # (128, 1, 16, 16)
    # Forward pass of out layer on batch
    batch_F_pad = np.pad(batch_F, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    batch_patches = stride_tricks.sliding_window_view(batch_F_pad, (3,3), axis=(2,3))
    batch_patches = batch_patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, C*9)
    batch_X = np.concatenate([batch_patches, np.ones((batch_patches.shape[0], 1))], axis=1)
    batch_pred = batch_X @ W
    batch_pred = batch_pred.reshape(128, 16, 16, 1).transpose(0, 3, 1, 2)
    mse_batch = np.mean((batch_pred - batch_Y) ** 2)

    diagnostics = {
        "mse_batch_before": float(np.mean((f['prediction'][f['batch.ids']] - batch_Y) ** 2)),
        "mse_batch_after": float(mse_batch),
        "out_weight_std": float(np.std(W_conv)),
        "out_bias": float(b_out[0])
    }

    return {"weights": w, "diagnostics": diagnostics}
