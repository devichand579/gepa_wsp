import numpy as np
import math

def edit(w, f):
    # Extract weights
    out_w = w['out.weight'].copy()   # (1,64,3,3)
    out_b = w['out.bias'].copy()     # (1,)
    conv3_w = w['conv3.weight'].copy()
    conv3_b = w['conv3.bias'].copy()
    conv2_w = w['conv2.weight'].copy()
    conv2_b = w['conv2.bias'].copy()
    conv1_w = w['conv1.weight'].copy()
    conv1_b = w['conv1.bias'].copy()

    # Feedback arrays
    features = f['features']         # (2000,64,16,16)
    y = f['y']                       # (2000,1,16,16)
    batch_ids = f['batch.ids']       # (128,)
    batch_y = f['batch.y']           # (128,1,16,16)
    batch_features = f['layer.out.input0']  # (128,64,16,16)

    # Check if full features match batch trace for the batch indices
    # If not, use only the batch for fitting.
    use_full = True
    try:
        # Compare a few samples to avoid huge memory
        idx_check = batch_ids[:min(10, len(batch_ids))]
        diff = np.max(np.abs(features[idx_check] - batch_features[:len(idx_check)]))
        if diff > 1e-4:
            use_full = False
    except Exception:
        use_full = False

    if use_full:
        X = features.reshape(2000, 64, 256)  # (2000,64,256)
        Y = y.reshape(2000, 1, 256)          # (2000,1,256)
        N = 2000
    else:
        X = batch_features.reshape(len(batch_ids), 64, 256)
        Y = batch_y.reshape(len(batch_ids), 1, 256)
        N = len(batch_ids)

    # We want to fit out_w (1,64,3,3) and out_b (1) to minimize
    # sum_{n,i,j} ( sum_{c,di,dj} out_w[0,c,di,dj] * X[n,c,i+di,j+dj] + out_b - Y[n,0,i,j] )^2
    # with zero padding. We can build the design matrix for each output pixel.
    # Since out_w is 3x3, for each output pixel (i,j) we have 64*9 = 576 features.
    # Total output pixels = N * 16 * 16 = N * 256.
    # We can solve the normal equations: (A^T A + lambda I) w = A^T y
    # where A is (N*256, 577) and w is (577,). 577 is small, so we can form A^T A.
    # We'll build A by shifting X.

    # Pad X spatially with zeros: (N,64,18,18)
    X_pad = np.zeros((N, 64, 18, 18), dtype=np.float32)
    X_pad[:, :, 1:17, 1:17] = X.reshape(N, 64, 16, 16)

    # Build A: shape (N*256, 577)
    # For each output pixel (i,j) in 0..15,0..15, the features are X_pad[:, :, i:i+3, j:j+3] flattened (64*9)
    # We can build A by looping over i,j and stacking.
    # To save memory, we can accumulate A^T A and A^T y directly.
    # A^T A is 577x577, A^T y is 577x1.
    n_features = 64 * 9 + 1  # 576 + bias
    ATA = np.zeros((n_features, n_features), dtype=np.float64)
    ATy = np.zeros((n_features, 1), dtype=np.float64)

    # We'll process in chunks over N to avoid large memory.
    chunk_size = 200
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        n_chunk = end - start
        # Build A_chunk: (n_chunk*256, 577)
        # We can build it by iterating over i,j and concatenating.
        # Actually, we can build a list of arrays for each (i,j) and stack.
        # But we can also compute ATA and ATy by summing over i,j.
        # For each (i,j), the features for all n in chunk are X_pad[start:end, :, i:i+3, j:j+3].reshape(n_chunk, 576)
        # And the target is Y[start:end, 0, i, j] (n_chunk,)
        # We can accumulate ATA += F^T F and ATy += F^T y for each (i,j).
        # This is efficient because F is (n_chunk, 576).
        for i in range(16):
            for j in range(16):
                # F: (n_chunk, 576)
                F = X_pad[start:end, :, i:i+3, j:j+3].reshape(n_chunk, 576)
                # Add bias column
                F_b = np.concatenate([F, np.ones((n_chunk, 1), dtype=np.float32)], axis=1)  # (n_chunk, 577)
                # Target: (n_chunk,)
                t = Y[start:end, 0, i, j]
                # Accumulate
                ATA += F_b.T @ F_b
                ATy += F_b.T @ t.reshape(-1, 1)

    # Ridge regularization for numerical stability
    lambda_reg = 1e-6
    ATA += lambda_reg * np.eye(n_features, dtype=np.float64)
    # Solve
    w_sol = np.linalg.solve(ATA, ATy)  # (577,1)
    w_sol = w_sol.flatten()

    # Extract out_w and out_b
    new_out_w = w_sol[:576].reshape(1, 64, 3, 3).astype(np.float32)
    new_out_b = np.array([w_sol[576]], dtype=np.float32)

    # Update weights
    w['out.weight'] = new_out_w
    w['out.bias'] = new_out_b

    # Diagnostics
    # Compute MSE on full data if available
    if use_full:
        # Compute prediction with new out layer
        # We can compute MSE using the normal equations solution
        # Actually, we can compute residual sum of squares: ||A w - y||^2 = y^T y - w^T A^T y
        # But we need y^T y. We can compute it.
        y_flat = Y.reshape(-1)
        yTy = np.sum(y_flat.astype(np.float64)**2)
        # w^T ATy
        wT_ATy = float(w_sol @ ATy)
        rss = yTy - wT_ATy
        mse = rss / (N * 256)
    else:
        mse = float('nan')

    diagnostics = {
        'mse_new': float(mse),
        'use_full': use_full,
        'n_samples': N,
        'ridge': lambda_reg,
        'out_bias': float(new_out_b[0])
    }

    return {"weights": w, "diagnostics": diagnostics}
