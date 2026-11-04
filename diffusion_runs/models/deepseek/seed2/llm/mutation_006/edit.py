def edit(w, f):
    import numpy as np

    # Closed-form least-squares refit of the final 3x3 convolutional readout
    # using the provided full-training final hidden features and epsilon targets.
    # This is exact for the fixed upstream representation and does not require
    # any iterative optimisation.
    F = f['features']          # (2000, 64, 16, 16)
    Y = f['y']                 # (2000, 1, 16, 16)
    N, C, H, W = F.shape
    K = 3
    D = C * K * K + 1          # 64*9 + 1 = 577

    ATA = np.zeros((D, D), dtype=np.float64)
    ATy = np.zeros((D, 1), dtype=np.float64)

    chunk = 100
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        F_chunk = F[start:end]
        Y_chunk = Y[start:end]
        B = F_chunk.shape[0]

        F_pad = np.pad(F_chunk, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
        patches = np.lib.stride_tricks.sliding_window_view(F_pad, (K, K), axis=(2, 3))
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(B * H * W, C * K * K)

        ones = np.ones((B * H * W, 1), dtype=np.float64)
        A_chunk = np.hstack([patches, ones])
        y_chunk = Y_chunk.transpose(0, 2, 3, 1).reshape(B * H * W, 1)

        ATA += A_chunk.T @ A_chunk
        ATy += A_chunk.T @ y_chunk

    # Tiny ridge for numerical stability; negligible effect on MSE.
    ridge = 1e-6
    ATA += ridge * np.eye(D, dtype=np.float64)
    sol = np.linalg.solve(ATA, ATy)

    out_weight = sol[:C * K * K].reshape(1, C, K, K).astype(np.float32)
    out_bias = sol[C * K * K:].reshape(1).astype(np.float32)

    w['out.weight'] = out_weight
    w['out.bias'] = out_bias

    return w
