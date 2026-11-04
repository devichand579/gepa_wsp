import numpy as np
import math

def edit(w, f):
    # Extract features and targets
    F = f['features'].astype(np.float32)  # (2000, 64, 16, 16)
    y = f['y'].astype(np.float32)          # (2000, 1, 16, 16)
    N, C, H, W = F.shape
    assert C == 64 and H == 16 and W == 16
    assert y.shape == (N, 1, H, W)

    # We will solve for out.weight (64,3,3) and out.bias (1)
    # Design matrix columns: 64*9 = 576 weights + 1 bias = 577
    # We use a ridge penalty lambda chosen by holdout.

    # Pad to 18x18 to avoid circular wrap-around for shifts up to 2
    pad = 2
    Hp, Wp = H + 2*pad, W + 2*pad  # 20x20
    F_pad = np.zeros((N, C, Hp, Wp), dtype=np.float32)
    F_pad[:, :, pad:pad+H, pad:pad+W] = F
    y_pad = np.zeros((N, 1, Hp, Wp), dtype=np.float32)
    y_pad[:, :, pad:pad+H, pad:pad+W] = y

    # FFT of padded features and targets
    F_fft = np.fft.fft2(F_pad, axes=(2,3))  # (N, C, Hp, Wp) complex64
    y_fft = np.fft.fft2(y_pad, axes=(2,3))  # (N, 1, Hp, Wp) complex64

    # Compute cross-spectral density for all channel pairs: M[c1,c2,f] = sum_n F_fft[n,c1,f] * conj(F_fft[n,c2,f])
    # Reshape to (N, C, Hp*Wp)
    F_fft_flat = F_fft.reshape(N, C, Hp*Wp)
    y_fft_flat = y_fft.reshape(N, 1, Hp*Wp)

    # M shape (C, C, Hp*Wp)
    M = np.einsum('ncf,ndf->cdf', F_fft_flat, np.conj(F_fft_flat), optimize=True)
    # My shape (C, Hp*Wp)
    My = np.einsum('ncf,nf->cf', F_fft_flat, np.conj(y_fft_flat[:,0,:]), optimize=True)

    # Inverse FFT to get cross-correlations
    # M_corr shape (C, C, Hp, Wp)
    M_corr = np.fft.ifft2(M.reshape(C, C, Hp, Wp), axes=(2,3)).real
    My_corr = np.fft.ifft2(My.reshape(C, Hp, Wp), axes=(1,2)).real

    # We need the correlation values for shifts (dr, dc) in {-2,-1,0,1,2} for M,
    # and (di, dj) in {-1,0,1} for My.
    # The zero shift is at index (pad, pad) = (2,2) in the padded array? Wait.
    # Our F_pad has original at indices 2..17. The cross-correlation at shift (dr,dc)
    # is sum_{i,j} F[i,j] * F[i+dr, j+dc]. In the FFT result, this corresponds to
    # index (dr mod Hp, dc mod Wp). Since we zero-padded, for dr in [-2,2], the index is
    # (dr + Hp) % Hp. For dr=-2 -> 18, -1 -> 19, 0 -> 0, 1 -> 1, 2 -> 2.
    # So we can extract the 5x5 block by taking indices [18,19,0,1,2] for both dims.
    idx = np.array([18, 19, 0, 1, 2])  # for shifts -2,-1,0,1,2
    # For My, shifts are -1,0,1 -> indices [19,0,1]
    idx_y = np.array([19, 0, 1])

    # Build A = X^T X (577x577) and b = X^T y (577)
    # Weight columns: k = c*9 + (di+1)*3 + (dj+1), where di,dj in {-1,0,1}
    # Bias column: index 576
    A = np.zeros((577, 577), dtype=np.float64)
    b = np.zeros(577, dtype=np.float64)

    # Fill A for weight-weight pairs
    for c1 in range(C):
        for di1 in range(-1, 2):
            for dj1 in range(-1, 2):
                k1 = c1*9 + (di1+1)*3 + (dj1+1)
                for c2 in range(C):
                    for di2 in range(-1, 2):
                        for dj2 in range(-1, 2):
                            k2 = c2*9 + (di2+1)*3 + (dj2+1)
                            dr = di2 - di1
                            dc = dj2 - dj1
                            # dr,dc in [-2,2]
                            i_r = dr + 2  # index into idx
                            i_c = dc + 2
                            A[k1, k2] = M_corr[c1, c2, idx[i_r], idx[i_c]]

    # Fill A for weight-bias and bias-bias
    # Bias column is all ones. The entry for weight k=(c,di,dj) and bias is sum_{n,i,j} F[n,c,i+di,j+dj]
    # This is the sum of F over the valid region for shift (di,dj).
    # We can compute this directly from F_pad or F.
    # For shift (di,dj), the valid i range is max(0, -di) to min(15, 15-di).
    # We can just sum the appropriate slice of F.
    for c in range(C):
        for di in range(-1, 2):
            for dj in range(-1, 2):
                k = c*9 + (di+1)*3 + (dj+1)
                i_start = max(0, -di)
                i_end = min(H, H - di)
                j_start = max(0, -dj)
                j_end = min(W, W - dj)
                # Sum over n and the valid region
                s = np.sum(F[:, c, i_start:i_end, j_start:j_end])
                A[k, 576] = s
                A[576, k] = s
    A[576, 576] = N * H * W  # sum of 1^2

    # Fill b for weight columns
    for c in range(C):
        for di in range(-1, 2):
            for dj in range(-1, 2):
                k = c*9 + (di+1)*3 + (dj+1)
                # b[k] = sum_{n,i,j} F[n,c,i+di,j+dj] * y[n,0,i,j]
                # This is the cross-correlation of F[:,c] with y at shift (di,dj).
                # In My_corr, the shift (di,dj) corresponds to index (di mod Hp, dj mod Wp).
                # For di=-1 -> 19, 0 -> 0, 1 -> 1.
                i_r = di + 1  # index into idx_y
                i_c = dj + 1
                b[k] = My_corr[c, idx_y[i_r], idx_y[i_c]]

    # b for bias column: sum_{n,i,j} y[n,0,i,j]
    b[576] = np.sum(y)

    # Now solve ridge regression: (A + lambda * I) theta = b
    # We'll use a holdout to pick lambda.
    # Split into train/val (e.g., 1800/200)
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_train = 1800
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # To do holdout efficiently, we need to compute A and b for the training subset.
    # But we already computed A and b for all data. We can't easily subset the normal equations
    # without recomputing. However, we can just use the full A and b and evaluate on the
    # full data? That would not help choose lambda. Alternatively, we can compute A and b
    # for the training subset by redoing the FFT on the subset. That's cheap enough.
    # Let's just do a quick holdout by recomputing for the subset.
    # Actually, we can just use a small set of lambdas and pick the one that minimizes
    # validation MSE. We'll recompute A and b for the training subset.

    def compute_Ab(F_sub, y_sub):
        Ns = F_sub.shape[0]
        F_pad_sub = np.zeros((Ns, C, Hp, Wp), dtype=np.float32)
        F_pad_sub[:, :, pad:pad+H, pad:pad+W] = F_sub
        y_pad_sub = np.zeros((Ns, 1, Hp, Wp), dtype=np.float32)
        y_pad_sub[:, :, pad:pad+H, pad:pad+W] = y_sub
        F_fft_sub = np.fft.fft2(F_pad_sub, axes=(2,3))
        y_fft_sub = np.fft.fft2(y_pad_sub, axes=(2,3))
        F_fft_flat_sub = F_fft_sub.reshape(Ns, C, Hp*Wp)
        y_fft_flat_sub = y_fft_sub.reshape(Ns, 1, Hp*Wp)
        M_sub = np.einsum('ncf,ndf->cdf', F_fft_flat_sub, np.conj(F_fft_flat_sub), optimize=True)
        My_sub = np.einsum('ncf,nf->cf', F_fft_flat_sub, np.conj(y_fft_flat_sub[:,0,:]), optimize=True)
        M_corr_sub = np.fft.ifft2(M_sub.reshape(C, C, Hp, Wp), axes=(2,3)).real
        My_corr_sub = np.fft.ifft2(My_sub.reshape(C, Hp, Wp), axes=(1,2)).real

        A_sub = np.zeros((577, 577), dtype=np.float64)
        b_sub = np.zeros(577, dtype=np.float64)
        for c1 in range(C):
            for di1 in range(-1, 2):
                for dj1 in range(-1, 2):
                    k1 = c1*9 + (di1+1)*3 + (dj1+1)
                    for c2 in range(C):
                        for di2 in range(-1, 2):
                            for dj2 in range(-1, 2):
                                k2 = c2*9 + (di2+1)*3 + (dj2+1)
                                dr = di2 - di1
                                dc = dj2 - dj1
                                i_r = dr + 2
                                i_c = dc + 2
                                A_sub[k1, k2] = M_corr_sub[c1, c2, idx[i_r], idx[i_c]]
        for c in range(C):
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    k = c*9 + (di+1)*3 + (dj+1)
                    i_start = max(0, -di)
                    i_end = min(H, H - di)
                    j_start = max(0, -dj)
                    j_end = min(W, W - dj)
                    s = np.sum(F_sub[:, c, i_start:i_end, j_start:j_end])
                    A_sub[k, 576] = s
                    A_sub[576, k] = s
        A_sub[576, 576] = Ns * H * W
        for c in range(C):
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    k = c*9 + (di+1)*3 + (dj+1)
                    i_r = di + 1
                    i_c = dj + 1
                    b_sub[k] = My_corr_sub[c, idx_y[i_r], idx_y[i_c]]
        b_sub[576] = np.sum(y_sub)
        return A_sub, b_sub

    # Compute for training subset
    F_train = F[train_idx]
    y_train = y[train_idx]
    A_train, b_train = compute_Ab(F_train, y_train)

    # Validation set
    F_val = F[val_idx]
    y_val = y[val_idx]

    # We need to evaluate MSE on validation for different lambdas.
    # To do that, we need to compute the prediction using the weights.
    # We can compute the design matrix for validation and then multiply by theta.
    # But that's expensive. Instead, we can just use the full A and b? No.
    # We can compute the validation MSE by using the fact that MSE = (1/N_val) * ||X_val theta - y_val||^2.
    # We can compute X_val^T X_val and X_val^T y_val for the validation set, but that's
    # not enough to compute the MSE for a given theta. We need the actual predictions.
    # Since we have the features, we can compute the convolution directly for validation.
    # That's 200 examples, 64 channels, 16x16, 3x3 kernel. That's fast.
    # Let's write a function to compute predictions given weights and bias.
    def predict(F_sub, weights, bias):
        # weights shape (64,3,3), bias scalar
        # F_sub shape (Ns, 64, 16, 16)
        Ns = F_sub.shape[0]
        # Pad F_sub
        F_pad_sub = np.zeros((Ns, C, Hp, Wp), dtype=np.float32)
        F_pad_sub[:, :, pad:pad+H, pad:pad+W] = F_sub
        # Convolution: for each output pixel (i,j), sum over c,di,dj weights[c,di+1,dj+1] * F_pad_sub[:, c, i+di+pad, j+dj+pad]
        # We can use a simple loop over the 9 shifts.
        out = np.zeros((Ns, 1, H, W), dtype=np.float32)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                # weights for this shift: shape (64,)
                w_shift = weights[:, di+1, dj+1]  # (64,)
                # F_pad_sub slice: for output (i,j), we need F_pad_sub[:, c, i+di+pad, j+dj+pad]
                # i from 0..15, so i+di+pad from pad+di to pad+di+15.
                # We can just take the slice of F_pad_sub that corresponds to the shifted input.
                # Actually, we can compute the convolution using matrix multiplication:
                # out[n,0,i,j] = sum_c w_shift[c] * F_pad_sub[n,c,i+di+pad,j+dj+pad]
                # This is a 1x1 convolution. We can do it by reshaping.
                # F_pad_sub[:, :, pad+di:pad+di+H, pad+dj:pad+dj+W] has shape (Ns, C, H, W)
                # We want sum_c w_shift[c] * F_slice[:, c, i, j] -> (Ns, H, W)
                # This is a tensordot.
                F_slice = F_pad_sub[:, :, pad+di:pad+di+H, pad+dj:pad+dj+W]
                out[:, 0, :, :] += np.tensordot(w_shift, F_slice, axes=([0],[1]))
        out += bias
        return out

    # Choose lambda by holdout
    lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_lambda = 0.0
    best_val_mse = float('inf')
    for lam in lambdas:
        A_reg = A_train + lam * np.eye(577)
        try:
            theta = np.linalg.solve(A_reg, b_train)
        except np.linalg.LinAlgError:
            continue
        weights = theta[:576].reshape(C, 3, 3)
        bias = theta[576]
        pred_val = predict(F_val, weights, bias)
        mse = np.mean((pred_val - y_val)**2)
        if mse < best_val_mse:
            best_val_mse = mse
            best_lambda = lam

    # Now solve on full data with best_lambda
    A_reg = A + best_lambda * np.eye(577)
    theta = np.linalg.solve(A_reg, b)
    weights = theta[:576].reshape(C, 3, 3).astype(np.float32)
    bias = np.float32(theta[576])

    # Update w
    w['out.weight'] = weights
    w['out.bias'] = np.array([bias], dtype=np.float32)

    # Diagnostics
    diag = {
        'best_lambda': float(best_lambda),
        'val_mse': float(best_val_mse),
        'out_weight_max': float(np.max(np.abs(weights))),
        'out_bias': float(bias)
    }
    return {"weights": w, "diagnostics": diag}
