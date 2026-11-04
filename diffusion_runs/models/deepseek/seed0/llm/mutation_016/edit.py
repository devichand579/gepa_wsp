import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    # Extract weights
    c1w = w['conv1.weight'].astype(np.float64)
    c1b = w['conv1.bias'].astype(np.float64)
    c2w = w['conv2.weight'].astype(np.float64)
    c2b = w['conv2.bias'].astype(np.float64)
    c3w = w['conv3.weight'].astype(np.float64)
    c3b = w['conv3.bias'].astype(np.float64)
    ow = w['out.weight'].astype(np.float64)
    ob = w['out.bias'].astype(np.float64)

    # Extract data
    x = f['x'].astype(np.float64)          # (2000,1,16,16)
    t = f['t'].astype(np.float64)          # (2000,)
    y = f['y'].astype(np.float64)          # (2000,1,16,16)
    N = x.shape[0]

    # Build input: concat x with t/31
    t_norm = (t / 31.0).reshape(N, 1, 1, 1)
    t_ch = np.broadcast_to(t_norm, (N, 1, 16, 16))
    inp = np.concatenate([x, t_ch], axis=1)  # (N,2,16,16)

    # Helper: 2D convolution with padding=1, stride=1
    def conv2d(x_in, weight, bias):
        # x_in: (N, C_in, H, W)
        # weight: (C_out, C_in, 3, 3)
        # bias: (C_out,)
        N, C_in, H, W = x_in.shape
        C_out = weight.shape[0]
        # Pad
        x_pad = np.pad(x_in, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
        # im2col
        # shape: (N, C_in, 3, 3, H, W)
        patches = sliding_window_view(x_pad, (3,3), axis=(2,3))
        # patches shape: (N, C_in, H, W, 3, 3)
        # We need (N, H, W, C_in*3*3)
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, H, W, C_in*9)
        # weight reshape: (C_out, C_in*9)
        w_flat = weight.reshape(C_out, C_in*9)
        # matmul: (N, H, W, C_in*9) @ (C_in*9, C_out) -> (N, H, W, C_out)
        out = patches @ w_flat.T
        out = out.transpose(0, 3, 1, 2)  # (N, C_out, H, W)
        out += bias.reshape(1, C_out, 1, 1)
        return out

    # Forward pass
    c1_pre = conv2d(inp, c1w, c1b)
    c1_out = np.maximum(c1_pre, 0)
    c2_pre = conv2d(c1_out, c2w, c2b)
    c2_out = np.maximum(c2_pre, 0)
    c3_pre = conv2d(c2_out, c3w, c3b)
    c3_out = np.maximum(c3_pre, 0)
    out_pred = conv2d(c3_out, ow, ob)  # (N,1,16,16)

    # --- Refit out layer ---
    # We want to solve: out(c3_out) = y
    # out is linear: y = A * vec(c3_out) + ob
    # Build design matrix for out layer using c3_out
    # We'll compute normal equations for out weights and bias.
    # Parameters: ow (1,64,3,3) + ob (1) -> 577 params
    # We'll accumulate A^T A and A^T y over all examples.
    # To save memory, process in chunks of examples.
    chunk_size = 200
    n_params = 64*9 + 1  # 577
    ATA = np.zeros((n_params, n_params), dtype=np.float64)
    ATy = np.zeros(n_params, dtype=np.float64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        c3_chunk = c3_out[start:end]  # (B,64,16,16)
        y_chunk = y[start:end]        # (B,1,16,16)
        B = c3_chunk.shape[0]
        # Build patches for out layer: for each output pixel, gather 3x3 neighborhood from 64 channels
        # Pad c3_chunk
        c3_pad = np.pad(c3_chunk, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
        # patches: (B, 64, 16, 16, 3, 3)
        patches = sliding_window_view(c3_pad, (3,3), axis=(2,3))
        # reshape to (B*256, 576)
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(B*256, 64*9)
        # add bias column
        ones = np.ones((B*256, 1), dtype=np.float64)
        X = np.concatenate([patches, ones], axis=1)  # (B*256, 577)
        y_flat = y_chunk.transpose(0, 2, 3, 1).reshape(B*256, 1)  # (B*256, 1)
        # Accumulate
        ATA += X.T @ X
        ATy += (X.T @ y_flat).ravel()

    # Solve ridge-regularized least squares
    ridge = 1e-6
    ATA_reg = ATA + ridge * np.eye(n_params)
    w_out_new = np.linalg.solve(ATA_reg, ATy)
    ow_new = w_out_new[:576].reshape(1, 64, 3, 3)
    ob_new = w_out_new[576:577]

    # --- Refit conv3 using target propagation ---
    # We have new out weights. We want to find target for c3_out (after ReLU) that minimizes final MSE.
    # For each example, we want T3 such that out(T3) = y.
    # out is linear: y = A_out * vec(T3) + ob_new.
    # We want T3 closest to current c3_out.
    # Solution: T3 = c3_out + A_out^T (A_out A_out^T)^{-1} (y - ob_new - A_out c3_out)
    # We can build A_out explicitly for one example (256 x 16384) and compute pseudo-inverse.
    # Since A_out is the same for all examples, we can precompute P = A_out^T (A_out A_out^T)^{-1} (16384 x 256).
    # Then T3 = c3_out + P @ (y - ob_new - A_out c3_out).
    # Build A_out matrix: maps (64,16,16) -> (1,16,16)
    # We'll build it as a sparse-like dense matrix for one example.
    # Actually, we can compute A_out and A_out^T using convolution operations.
    # But building the 256x16384 matrix is fine (4M elements, 32MB).
    # Let's build A_out for a single example.
    # A_out: for each output pixel (i,j), it's a row of length 64*16*16 = 16384.
    # We can construct it by applying the out convolution to basis vectors, but that's slow.
    # Instead, we can use the fact that out is a 3x3 conv. We can build the matrix by iterating over output pixels and input channels.
    # But 256x16384 is small enough to build directly.
    # We'll build A_out as a dense matrix.
    H = W = 16
    C = 64
    A_out = np.zeros((H*W, C*H*W), dtype=np.float64)
    # For each output pixel (i,j), and each input channel c, and input pixel (u,v), the weight is ow_new[0,c, i-u+1, j-v+1] if in bounds.
    # We can fill it by looping over output pixels and kernel offsets.
    # This is 256 * 64 * 9 = 147k operations, trivial.
    for i in range(H):
        for j in range(W):
            out_idx = i*W + j
            for c in range(C):
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        u = i + di
                        v = j + dj
                        if 0 <= u < H and 0 <= v < W:
                            # kernel index: di+1, dj+1
                            k_i = di + 1
                            k_j = dj + 1
                            in_idx = c*H*W + u*W + v
                            A_out[out_idx, in_idx] += ow_new[0, c, k_i, k_j]
    # Now compute P = A_out^T (A_out A_out^T)^{-1}
    # A_out is (256, 16384). A_out A_out^T is (256, 256).
    # We can compute it directly: A_out @ A_out.T
    AAT = A_out @ A_out.T
    # Add small ridge for stability
    AAT_reg = AAT + 1e-6 * np.eye(H*W)
    # Solve for P: P = A_out^T @ inv(AAT_reg)
    # Instead of inverting, we can solve AAT_reg @ X = A_out, then P = X^T? Wait.
    # We want P = A_out^T @ inv(AAT_reg). So P^T = inv(AAT_reg) @ A_out.
    # We can solve AAT_reg @ Y = A_out, then Y = inv(AAT_reg) @ A_out. Then P = Y^T.
    Y = np.linalg.solve(AAT_reg, A_out)  # (256, 16384)
    P = Y.T  # (16384, 256)

    # Now compute target T3 for all examples.
    # T3 = c3_out + P @ (y - ob_new - A_out @ c3_out)
    # We can compute A_out @ c3_out for all examples by reshaping c3_out to (N, 16384) and doing matmul.
    c3_flat = c3_out.reshape(N, -1)  # (N, 16384)
    y_flat = y.reshape(N, -1)        # (N, 256)
    # Compute residual: y - ob_new - A_out @ c3_out
    # A_out @ c3_out: (N, 16384) @ (16384, 256) -> (N, 256)
    # But A_out is (256, 16384), so c3_flat @ A_out.T gives (N, 256)
    pred_out = c3_flat @ A_out.T + ob_new  # (N, 256)
    residual = y_flat - pred_out
    # T3_flat = c3_flat + residual @ P.T  (since P is (16384, 256), P.T is (256, 16384))
    T3_flat = c3_flat + residual @ P.T
    T3 = T3_flat.reshape(N, C, H, W)

    # Now refit conv3. We have c2_out (input to conv3) and target T3.
    # conv3 is: c3_pre = conv2d(c2_out, c3w, c3b), c3_out = ReLU(c3_pre).
    # We want c3_out ≈ T3. We can fit c3_pre to T3, ignoring ReLU, or use fixed ReLU mask.
    # Let's use fixed ReLU mask from current c3_out: mask = (c3_out > 0).
    # We want conv2d(c2_out, c3w, c3b) ≈ T3 on mask, and <= 0 on off-mask.
    # This is a linear least squares with inequality constraints. To keep it simple, we'll just fit c3_pre to T3 directly (no mask), then apply ReLU.
    # Actually, fitting c3_pre to T3 directly might be better because T3 can be negative, and we want c3_pre to match T3, then ReLU will clip.
    # Let's do a linear least squares for conv3 weights and bias using c2_out and T3.
    # Parameters: c3w (64,64,3,3) + c3b (64) -> 64*64*9 + 64 = 36928 params.
    # This is larger. We can solve normal equations for conv3.
    # We have N=2000 examples. Input to conv3: c2_out (N,64,16,16). Target: T3 (N,64,16,16).
    # We want to find W3, b3 such that conv2d(c2_out, W3, b3) ≈ T3.
    # This is a linear regression with 36928 parameters. We have 2000*64*16*16 = 32.7M equations.
    # Normal equations: A^T A is 36928 x 36928. That's 1.36e9 elements = 10.9 GB in float64. Too large!
    # We cannot solve full conv3 normal equations directly.
    # We need a more efficient approach. Since conv3 is a convolution, we can solve it in the Fourier domain? Or use the fact that it's a linear convolution.
    # Alternatively, we can refit conv3 per output channel? No, because input channels are shared.
    # We can use the fact that the convolution is small. We can solve for each output channel separately? No, because the input channels are the same for all output channels, but the weights are independent per output channel. Actually, conv3 maps 64 input channels to 64 output channels. For each output channel, it's a linear combination of all input channels. So we can solve for each output channel's weights independently! Because the output channels are independent given the input. The loss is sum over output channels of ||conv2d(c2_out, W3[c], b3[c]) - T3[c]||^2. So we can solve for each output channel separately. That's 64 separate linear regressions, each with 64*9 + 1 = 577 parameters. This is exactly the same size as the out layer! So we can do it efficiently.
    # For each output channel c_out, we want to fit a 3x3 conv from 64 input channels to 1 output channel, plus bias.
    # This is identical to the out layer problem, but with 64 input channels and target T3[:, c_out, :, :].
    # So we can reuse the same chunked normal equations approach for each output channel.
    # We have 64 output channels. For each, we need to accumulate A^T A and A^T y over all examples.
    # A is the design matrix for one output pixel: 64*9 + 1 = 577 params.
    # We can compute the Gram matrix for the input c2_out once, and then use it for all 64 output channels!
    # Because the design matrix X is the same for all output channels (it depends only on c2_out). The target y changes per output channel.
    # So we can compute ATA (577x577) once from c2_out, and then for each output channel, compute ATy (577) and solve.
    # This is very efficient!

    # Let's compute ATA for conv3 input (c2_out).
    # We need to build patches from c2_out for all examples and all pixels.
    # c2_out: (N, 64, 16, 16). We want for each output pixel (i,j), the 3x3 neighborhood from all 64 channels.
    # This is exactly the same as the out layer design matrix, but with 64 input channels instead of 64? Wait, out layer had 64 input channels. conv3 also has 64 input channels. So the design matrix X for conv3 is exactly the same as for out layer! Because both take 64 input channels and use a 3x3 conv. The only difference is the target: out layer target is y (1 channel), conv3 target is T3 (64 channels). So we can reuse the ATA matrix we already computed for the out layer! But wait, the out layer ATA was computed from c3_out (the output of conv3). For conv3, the input is c2_out. So the design matrix is built from c2_out, not c3_out. So we need to compute ATA for c2_out. But c2_out has 64 channels, same as c3_out. So we can compute ATA for c2_out using the same chunked approach. Then for each of the 64 output channels, we compute ATy using the target T3 for that channel, and solve for the 577 weights. This gives new c3w and c3b.

    # Let's do that.
    # Compute ATA for c2_out
    ATA_c3 = np.zeros((n_params, n_params), dtype=np.float64)
    # We'll also need ATy for each output channel. We can accumulate ATy for all 64 channels at once.
    # ATy shape: (64, 577)
    ATy_c3 = np.zeros((64, n_params), dtype=np.float64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        c2_chunk = c2_out[start:end]  # (B,64,16,16)
        T3_chunk = T3[start:end]      # (B,64,16,16)
        B = c2_chunk.shape[0]
        c2_pad = np.pad(c2_chunk, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
        patches = sliding_window_view(c2_pad, (3,3), axis=(2,3))
        patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(B*256, 64*9)
        ones = np.ones((B*256, 1), dtype=np.float64)
        X = np.concatenate([patches, ones], axis=1)  # (B*256, 577)
        # Accumulate ATA
        ATA_c3 += X.T @ X
        # For each output channel, accumulate ATy
        # T3_chunk: (B,64,16,16) -> (B*256, 64)
        T3_flat = T3_chunk.transpose(0, 2, 3, 1).reshape(B*256, 64)
        # ATy_c3: (64, 577) += T3_flat.T @ X
        ATy_c3 += T3_flat.T @ X

    # Solve for each output channel
    ridge_c3 = 1e-6
    ATA_c3_reg = ATA_c3 + ridge_c3 * np.eye(n_params)
    # Solve for all 64 channels at once: ATA_c3_reg @ W = ATy_c3.T  -> W is (577, 64)
    W_c3 = np.linalg.solve(ATA_c3_reg, ATy_c3.T)  # (577, 64)
    c3w_new = W_c3[:576].T.reshape(64, 64, 3, 3)
    c3b_new = W_c3[576:577].T.ravel()  # (64,)

    # Update weights
    w['out.weight'] = ow_new.astype(np.float32)
    w['out.bias'] = ob_new.astype(np.float32)
    w['conv3.weight'] = c3w_new.astype(np.float32)
    w['conv3.bias'] = c3b_new.astype(np.float32)

    # Diagnostics
    diagnostics = {
        'mse_before': float(np.mean((out_pred - y)**2)),
        'out_weight_norm': float(np.linalg.norm(ow_new)),
        'conv3_weight_norm': float(np.linalg.norm(c3w_new)),
        'ridge_out': ridge,
        'ridge_conv3': ridge_c3,
    }

    return {'weights': w, 'diagnostics': diagnostics}
