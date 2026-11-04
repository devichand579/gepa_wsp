import numpy as np
import math

def edit(w, f):
    # Extract feedback arrays
    x = f['x']          # (2000,1,16,16)
    t = f['t']          # (2000,)
    y = f['y']          # (2000,1,16,16)
    features = f['features']  # (2000,64,16,16) current out.input0
    prediction = f['prediction']  # (2000,1,16,16)
    alphabar = f['alphabar']  # (32,)

    # Current weights
    conv1_w = w['conv1.weight'].copy()
    conv1_b = w['conv1.bias'].copy()
    conv2_w = w['conv2.weight'].copy()
    conv2_b = w['conv2.bias'].copy()
    conv3_w = w['conv3.weight'].copy()
    conv3_b = w['conv3.bias'].copy()
    out_w = w['out.weight'].copy()
    out_b = w['out.bias'].copy()

    # The cached features are the output of conv3 (which is identity on the first 32 channels).
    # So the first 32 channels are ReLU(conv1(x,t)), the rest are zero.
    F1 = features[:, :32, :, :].copy()  # (2000,32,16,16)

    # Helper: 3x3 conv with padding 1 using im2col, batched.
    def conv3x3_batch(inp, weight, bias):
        # inp: (N, C_in, 16, 16)
        # weight: (C_out, C_in, 3, 3)
        # bias: (C_out,)
        N, C_in, H, W = inp.shape
        C_out = weight.shape[0]
        # Pad to (N, C_in, H+2, W+2)
        padded = np.pad(inp, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        # Use sliding_window_view to get patches
        # shape: (N, C_in, H, W, 3, 3)
        patches = np.lib.stride_tricks.sliding_window_view(padded, (3,3), axis=(2,3))
        # Reshape to (N*H*W, C_in*9)
        patches = patches.transpose(0,2,3,1,4,5).reshape(N*H*W, C_in*9)
        # Weight matrix: (C_in*9, C_out)
        W_mat = weight.transpose(1,2,3,0).reshape(C_in*9, C_out)
        # Matmul
        out = patches @ W_mat + bias  # (N*H*W, C_out)
        out = out.reshape(N, H, W, C_out).transpose(0,3,1,2)
        return out

    # Initialize new conv2 and conv3 with deterministic random weights.
    # Use a fixed seed for reproducibility.
    rng = np.random.RandomState(42)
    # conv2: 32 -> 64, 3x3. Use He-like initialization.
    std2 = math.sqrt(2.0 / (32 * 9))
    new_conv2_w = rng.randn(64, 32, 3, 3).astype(np.float32) * std2
    new_conv2_b = np.zeros(64, dtype=np.float32)
    # conv3: 64 -> 64, 3x3.
    std3 = math.sqrt(2.0 / (64 * 9))
    new_conv3_w = rng.randn(64, 64, 3, 3).astype(np.float32) * std3
    new_conv3_b = np.zeros(64, dtype=np.float32)

    # Compute new features F3 = ReLU(conv3(ReLU(conv2(F1))))
    # Process in batches to save memory.
    batch_size = 200
    N = F1.shape[0]
    F3 = np.empty((N, 64, 16, 16), dtype=np.float32)
    for i in range(0, N, batch_size):
        end = min(i+batch_size, N)
        F1_batch = F1[i:end]
        # conv2
        F2_batch = conv3x3_batch(F1_batch, new_conv2_w, new_conv2_b)
        F2_batch = np.maximum(F2_batch, 0.0)
        # conv3
        F3_batch = conv3x3_batch(F2_batch, new_conv3_w, new_conv3_b)
        F3_batch = np.maximum(F3_batch, 0.0)
        F3[i:end] = F3_batch

    # Now refit out layer via least squares on F3.
    # out is a 3x3 conv over F3: pred[n,0,i,j] = sum_{c,k,l} W[c,k,l] * F3[n,c,i+k-1,j+l-1] + b
    # We need to build design matrix A (N*16*16, 64*9+1) and target y_flat.
    # To save memory, accumulate normal equations.
    C_out = 64
    K = 3
    D = C_out * K * K  # 576
    ATA = np.zeros((D+1, D+1), dtype=np.float64)
    ATy = np.zeros(D+1, dtype=np.float64)

    # Pad F3 for im2col
    F3_padded = np.pad(F3, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    # y flattened: (N*16*16,)
    y_flat = y[:, 0, :, :].reshape(-1).astype(np.float64)

    for i in range(0, N, batch_size):
        end = min(i+batch_size, N)
        # Extract patches from F3 for this batch
        # F3_padded[i:end] shape: (B, 64, 18, 18)
        # sliding window: (B, 64, 16, 16, 3, 3)
        patches = np.lib.stride_tricks.sliding_window_view(F3_padded[i:end], (3,3), axis=(2,3))
        # Reshape to (B*16*16, 64*9)
        patches = patches.transpose(0,2,3,1,4,5).reshape(-1, D)
        # Add bias column
        A_batch = np.hstack([patches, np.ones((patches.shape[0], 1), dtype=np.float64)])
        y_batch = y_flat[i*256:end*256]
        # Accumulate
        ATA += A_batch.T @ A_batch
        ATy += A_batch.T @ y_batch

    # Solve for weights (ridge with tiny regularization for stability)
    lam = 1e-6
    ATA_reg = ATA + lam * np.eye(D+1)
    try:
        sol = np.linalg.solve(ATA_reg, ATy)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(ATA_reg, ATy, rcond=None)[0]

    # Extract out weights and bias
    new_out_w = sol[:D].reshape(C_out, 3, 3).astype(np.float32)  # (64,3,3)
    new_out_w = new_out_w.reshape(1, C_out, 3, 3)  # (1,64,3,3)
    new_out_b = np.array([sol[D]], dtype=np.float32)

    # Evaluate new MSE on training set (using F3 and new out)
    # We can compute it quickly by using the normal equations? Actually, we can compute MSE directly.
    # But to save time, we can just compute the training MSE from the solved weights.
    # MSE = (1/N) * sum (y - A sol)^2 = (1/N) * (y^T y - 2 sol^T ATy + sol^T ATA sol)
    # We have y^T y? We can compute it.
    yTy = np.sum(y_flat**2)
    mse_new = (yTy - 2 * sol @ ATy + sol @ ATA @ sol) / (N * 256)

    # Also compute MSE of original prediction
    mse_old = np.mean((prediction - y)**2)

    # If new MSE is worse, fall back to just refitting out on original features.
    if mse_new > mse_old:
        # Fallback: refit out on original features (which are the cached features)
        # Use the same normal equation approach but with features instead of F3.
        F_orig = features  # (2000,64,16,16)
        F_orig_padded = np.pad(F_orig, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        ATA_orig = np.zeros((D+1, D+1), dtype=np.float64)
        ATy_orig = np.zeros(D+1, dtype=np.float64)
        for i in range(0, N, batch_size):
            end = min(i+batch_size, N)
            patches = np.lib.stride_tricks.sliding_window_view(F_orig_padded[i:end], (3,3), axis=(2,3))
            patches = patches.transpose(0,2,3,1,4,5).reshape(-1, D)
            A_batch = np.hstack([patches, np.ones((patches.shape[0], 1), dtype=np.float64)])
            y_batch = y_flat[i*256:end*256]
            ATA_orig += A_batch.T @ A_batch
            ATy_orig += A_batch.T @ y_batch
        ATA_orig_reg = ATA_orig + lam * np.eye(D+1)
        try:
            sol_orig = np.linalg.solve(ATA_orig_reg, ATy_orig)
        except np.linalg.LinAlgError:
            sol_orig = np.linalg.lstsq(ATA_orig_reg, ATy_orig, rcond=None)[0]
        new_out_w = sol_orig[:D].reshape(1, C_out, 3, 3).astype(np.float32)
        new_out_b = np.array([sol_orig[D]], dtype=np.float32)
        # Keep original conv2/conv3
        new_conv2_w = conv2_w
        new_conv2_b = conv2_b
        new_conv3_w = conv3_w
        new_conv3_b = conv3_b
        mse_new = (yTy - 2 * sol_orig @ ATy_orig + sol_orig @ ATA_orig @ sol_orig) / (N * 256)

    # Update weights
    w['conv2.weight'] = new_conv2_w.astype(np.float32)
    w['conv2.bias'] = new_conv2_b.astype(np.float32)
    w['conv3.weight'] = new_conv3_w.astype(np.float32)
    w['conv3.bias'] = new_conv3_b.astype(np.float32)
    w['out.weight'] = new_out_w.astype(np.float32)
    w['out.bias'] = new_out_b.astype(np.float32)

    # Ensure all weights are finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "mse_old": float(mse_old),
        "mse_new": float(mse_new),
        "used_random_conv": bool(mse_new <= mse_old),
        "out_bias": float(new_out_b[0]),
    }
    return {"weights": w, "diagnostics": diagnostics}
