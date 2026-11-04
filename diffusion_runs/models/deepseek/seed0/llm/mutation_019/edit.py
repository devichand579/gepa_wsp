import numpy as np
import math

def edit(w, f):
    """
    Hypothesis: The final out layer is a linear 3x3 convolution over the frozen
    64-channel feature maps.  We can refit its weights and bias exactly by
    closed-form ridge least squares on the full 2000-example training set.
    This strictly minimizes training MSE for the frozen upstream features.
    Earlier layers are left untouched so cached features stay valid.
    """
    # ------------------------------------------------------------------
    # 0.  Extract feedback arrays
    # ------------------------------------------------------------------
    F = f['features']          # (2000, 64, 16, 16) float32
    y = f['y']                 # (2000,  1, 16, 16) float32
    N, C, H, W = F.shape
    assert C == 64 and H == 16 and W == 16

    # ------------------------------------------------------------------
    # 1.  Zero-pad to 32x32 for linear cross-correlation via FFT
    # ------------------------------------------------------------------
    pad = 32
    Fpad = np.zeros((N, C, pad, pad), dtype=np.float32)
    Fpad[:, :, :H, :W] = F
    Ypad = np.zeros((N, 1, pad, pad), dtype=np.float32)
    Ypad[:, :, :H, :W] = y

    # ------------------------------------------------------------------
    # 2.  Accumulate frequency-domain Gram matrices in chunks
    # ------------------------------------------------------------------
    # M[u,v] : (64,64) complex  = sum_n FFT(F_c) * conj(FFT(F_c'))
    # V[u,v] : (64,)   complex  = sum_n FFT(F_c) * conj(FFT(y))
    M = np.zeros((pad, pad//2+1, C, C), dtype=np.complex64)
    V = np.zeros((pad, pad//2+1, C),    dtype=np.complex64)

    chunk = 200
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        Fc = Fpad[start:end]          # (chunk, 64, 32, 32)
        Yc = Ypad[start:end]          # (chunk,  1, 32, 32)

        Ff = np.fft.rfft2(Fc, axes=(-2, -1))   # (chunk, 64, 32, 17)
        Yf = np.fft.rfft2(Yc, axes=(-2, -1))   # (chunk,  1, 32, 17)

        # Loop over frequency bins (32*17 = 544)
        for u in range(pad):
            for v in range(pad//2 + 1):
                A = Ff[:, :, u, v]                 # (chunk, 64)
                B = Yf[:, 0, u, v]                 # (chunk,)
                M[u, v] += A.T @ np.conj(A)
                V[u, v] += A.T @ np.conj(B)

    # ------------------------------------------------------------------
    # 3.  Inverse FFT to obtain spatial cross-correlations
    # ------------------------------------------------------------------
    # C[di, dj, c, c'] = sum_n sum_{u,v} F_c[u,v] * F_c'[u+di, v+dj]
    C = np.fft.irfft2(M, s=(pad, pad), axes=(0, 1))   # (32,32,64,64)
    # Cy[di, dj, c] = sum_n sum_{u,v} F_c[u,v] * y[u+di, v+dj]
    Cy = np.fft.irfft2(V, s=(pad, pad), axes=(0, 1))  # (32,32,64)

    # ------------------------------------------------------------------
    # 4.  Extract the lags needed for a 3x3 convolution with padding 1
    # ------------------------------------------------------------------
    # For C we need delta_i, delta_j in -2..2.
    # In the 32x32 circular array: 0->0, 1->1, 2->2, 30->-2, 31->-1
    idx = [0, 1, 2, 30, 31]
    C_sub = C[np.ix_(idx, idx)]          # (5,5,64,64)

    # For Cy we need lag (1-di, 1-dj) for di,dj in 0..2.
    # lag 1 -> index 1, lag 0 -> index 0, lag -1 -> index 31
    idx_y = [1, 0, 31]
    Cy_sub = Cy[np.ix_(idx_y, idx_y)]    # (3,3,64)

    # ------------------------------------------------------------------
    # 5.  Assemble the 577x577 normal-equation matrix and RHS
    # ------------------------------------------------------------------
    # Parameters: out.weight[0, c, di, dj] for c=0..63, di,dj=0..2, and out.bias
    # Total = 64*9 + 1 = 577
    P = C * 9 + 1
    A = np.zeros((P, P), dtype=np.float64)
    b = np.zeros(P, dtype=np.float64)

    # Helper to map (di,dj) to parameter index
    def pidx(c, di, dj):
        return c * 9 + di * 3 + dj

    # Fill blocks for weight-weight interactions
    for di in range(3):
        for dj in range(3):
            for di2 in range(3):
                for dj2 in range(3):
                    # delta = (di2 - di, dj2 - dj)
                    d_i = di2 - di
                    d_j = dj2 - dj
                    # Map delta to index in C_sub
                    # C_sub indices: 0->-2, 1->-1, 2->0, 3->1, 4->2
                    ii = d_i + 2
                    jj = d_j + 2
                    block = C_sub[ii, jj]          # (64,64)
                    for c in range(C):
                        for c2 in range(C):
                            A[pidx(c, di, dj), pidx(c2, di2, dj2)] = block[c, c2]

    # Bias-weight interactions: bias is last index
    # The design matrix column for bias is 1 for every sample.
    # So A[pidx, bias] = sum_n sum_{i,j} F[n,c,i+di-1,j+dj-1] * 1
    # This is the sum of the shifted feature maps over all samples and pixels.
    # We can compute it from C_sub?  Actually it's the sum over n,i,j of F[n,c,i+di-1,j+dj-1].
    # That equals the cross-correlation of F_c with a constant 1 map.
    # We can compute it directly from F: sum over n,i,j of F[n,c,i+di-1,j+dj-1].
    # Since padding is 1, for di=0, index i-1; di=1, index i; di=2, index i+1.
    # We can just compute the sum of F over all n and pixels, with appropriate shifts.
    # Simpler: compute the mean feature map sum over n and pixels.
    # Actually, we can compute the sum of F over n and spatial dims, then shift.
    # Let S[c] = sum_{n,i,j} F[n,c,i,j].
    # Then sum_{n,i,j} F[n,c,i+di-1,j+dj-1] = S[c] minus boundary terms.
    # But easier: just compute it directly using the already computed C? No.
    # Let's compute it directly from F.  It's cheap.
    # We need sum_{n,i,j} F[n,c,i+di-1,j+dj-1] for di,dj in 0..2.
    # We can compute by summing F over n and then convolving with a 3x3 ones kernel? 
    # Actually, it's just the sum of F over n, then for each (di,dj) sum over i,j of the shifted map.
    # Since F is zero-padded, we can compute the sum of F over n and all pixels, then subtract the edges.
    # But it's easier to just compute the sum of F over n, then for each di,dj, sum the appropriate region.
    # Let's do it directly: 
    # S_full = F.sum(axis=0)  # (64,16,16)
    # For each di,dj, the sum over i,j of F[i+di-1, j+dj-1] with zero padding is:
    # sum over u,v of F[u,v] where u in [di-1, di+14] and v in [dj-1, dj+14] intersect [0,15].
    # This is just the sum of a sub-rectangle of S_full.
    # We can compute it by summing the appropriate slice.
    S_full = F.sum(axis=0)  # (64,16,16)
    for di in range(3):
        for dj in range(3):
            # u range: max(0, di-1) to min(15, di+14)
            u0 = max(0, di - 1)
            u1 = min(15, di + 14)
            v0 = max(0, dj - 1)
            v1 = min(15, dj + 14)
            # sum over u0..u1, v0..v1
            # Actually the condition is: i+di-1 in [0,15] => i in [1-di, 16-di]
            # So u = i+di-1 ranges from max(0, di-1) to min(15, di+14)
            # But we need sum over i,j of F[i+di-1, j+dj-1] where i,j in 0..15.
            # This is exactly sum over u in [di-1, di+14] intersect [0,15] and v in [dj-1, dj+14] intersect [0,15] of F[u,v].
            # So we can just sum the slice.
            # However, note that for di=0, u ranges from -1 to 14, so valid u: 0..14.
            # For di=1, u: 0..15.
            # For di=2, u: 1..15.
            # So the slice is S_full[:, u0:u1+1, v0:v1+1].sum(axis=(1,2))
            # Let's compute it.
            # Actually, we can just compute the sum of F over n and the valid region.
            # Let's do it directly from F to avoid confusion.
            # We can compute the sum over n and i,j of F[n,c,i+di-1,j+dj-1] by:
            # sum over i,j of F[n,c,i+di-1,j+dj-1] = sum over u,v of F[n,c,u,v] where u = i+di-1, v = j+dj-1, i,j in 0..15.
            # This is sum over u in [di-1, di+14] ∩ [0,15], v in [dj-1, dj+14] ∩ [0,15].
            # So we can just sum the appropriate slice of F.sum(axis=0).
            # Let's do that.
            # But wait, we need this for the bias column. The bias column is 1 for all samples.
            # So A[pidx(c,di,dj), bias] = sum_{n,i,j} F[n,c,i+di-1,j+dj-1] * 1 = sum_{n} sum_{i,j} F[n,c,i+di-1,j+dj-1].
            # That's exactly the sum over n of the slice.
            # So we can compute it as S_full[:, u0:u1+1, v0:v1+1].sum(axis=(1,2)).
            # Let's do it.
            u0 = max(0, di - 1)
            u1 = min(15, di + 14)
            v0 = max(0, dj - 1)
            v1 = min(15, dj + 14)
            # The sum over n and i,j of F[n,c,i+di-1,j+dj-1] is the sum of F over n and the valid region.
            # But careful: the region of u,v is exactly the valid indices. So it's just the sum of F over n and u in [u0,u1], v in [v0,v1].
            # Let's compute it directly from F to be safe.
            # We can compute S = F.sum(axis=0)  # (64,16,16)
            # Then sum over the slice.
            # Actually, we can just compute the sum over n and the slice.
            # Let's do it.
            # We'll compute the sum of F over n and the slice.
            # Since F is (2000,64,16,16), we can do F[:, :, u0:u1+1, v0:v1+1].sum(axis=(0,2,3))
            # That gives (64,).
            # Let's do that.
            # But we need to do this for each di,dj. 9 times. That's fine.
            # However, we can also compute it from S_full.
            # Let's just do it directly.
            # We'll compute it in the loop.
            pass

    # Actually, let's compute the bias column properly.
    # We'll compute the sum of F over n and the valid region for each (di,dj).
    # We can do this efficiently by summing F over n first: S = F.sum(axis=0)  # (64,16,16)
    S = F.sum(axis=0)  # (64,16,16)
    for di in range(3):
        for dj in range(3):
            u0 = max(0, di - 1)
            u1 = min(15, di + 14)
            v0 = max(0, dj - 1)
            v1 = min(15, dj + 14)
            # sum over u0..u1, v0..v1
            # Note: the condition is i+di-1 in [0,15] => i in [1-di, 16-di]
            # So u = i+di-1 ranges from max(0, di-1) to min(15, di+14)
            # This is exactly the slice.
            # Let's compute the sum.
            col = S[:, u0:u1+1, v0:v1+1].sum(axis=(1,2))  # (64,)
            for c in range(C):
                A[pidx(c, di, dj), -1] = col[c]
                A[-1, pidx(c, di, dj)] = col[c]

    # Bias-bias interaction: sum over n,i,j of 1*1 = N * H * W = 2000*256 = 512000
    A[-1, -1] = N * H * W

    # RHS: X^T y
    # For weight parameters: sum_{n,i,j} F[n,c,i+di-1,j+dj-1] * y[n,0,i,j]
    # This is exactly Cy_sub at the appropriate lag.
    # Cy_sub has shape (3,3,64) where index (di,dj) corresponds to lag (1-di, 1-dj).
    # Wait, we defined Cy_sub = Cy[np.ix_([1,0,31], [1,0,31])].
    # So Cy_sub[di, dj, c] = Cy at lag (1-di, 1-dj) for di,dj in 0..2.
    # And Cy at lag (k,l) = sum_n sum_{u,v} F_c[u,v] * y[u+k, v+l].
    # We need sum_{n,i,j} F[n,c,i+di-1,j+dj-1] * y[n,0,i,j].
    # Let u = i+di-1, v = j+dj-1. Then i = u - di + 1, j = v - dj + 1.
    # So sum_{u,v} F_c[u,v] * y[u - di + 1, v - dj + 1] = Cy at lag (1-di, 1-dj).
    # Yes! So Cy_sub[di, dj, c] is exactly the RHS for weight (c,di,dj).
    for di in range(3):
        for dj in range(3):
            for c in range(C):
                b[pidx(c, di, dj)] = Cy_sub[di, dj, c]

    # For bias: sum_{n,i,j} 1 * y[n,0,i,j] = sum of y over all samples and pixels.
    b[-1] = y.sum()

    # ------------------------------------------------------------------
    # 6.  Solve ridge-regularized least squares
    # ------------------------------------------------------------------
    # Add small ridge to diagonal for numerical stability
    lam = 1e-6
    A += lam * np.eye(P, dtype=np.float64)
    # Solve A x = b
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, b, rcond=None)[0]

    # ------------------------------------------------------------------
    # 7.  Write back into w
    # ------------------------------------------------------------------
    # out.weight shape (1, 64, 3, 3)
    out_w = x[:64*9].reshape(64, 3, 3)
    out_b = x[-1]
    w['out.weight'][0] = out_w.astype(np.float32)
    w['out.bias'][0] = np.float32(out_b)

    # ------------------------------------------------------------------
    # 8.  Diagnostics
    # ------------------------------------------------------------------
    # Compute new MSE on the full training set using the new out layer.
    # We can do this efficiently by convolving features with the new weights.
    # But to save time, we can just report the closed-form residual.
    # Actually, let's compute the new MSE properly.
    # We can use the normal equations: MSE = (y^T y - 2 x^T b + x^T A x) / (N*H*W)
    # But A includes ridge.  Let's compute it directly for a few samples? 
    # Better: compute the full MSE using the new weights.
    # We can do a quick convolution using the features and new weights.
    # Since we have F (2000,64,16,16) and out_w (64,3,3), we can compute the prediction.
    # We can use scipy? No, only numpy.  We can do a simple loop over the 9 offsets.
    # That's 9 * 2000 * 64 * 16 * 16 operations, which is 9 * 32M = 288M ops.  Fine.
    # Let's do it to get the actual MSE.
    pred = np.zeros((N, 1, H, W), dtype=np.float32)
    for di in range(3):
        for dj in range(3):
            # shifted features: F[n,c,i+di-1,j+dj-1]
            # We can use np.pad or just slice.
            # For simplicity, use zero-padding.
            # We can compute the contribution by shifting F.
            # Actually, we can use the fact that we already have the design matrix.
            # But let's just do a direct convolution.
            pass

    # Let's compute the new MSE using the closed-form solution.
    # The residual sum of squares is y^T y - 2 x^T b + x^T A x (ignoring ridge).
    # We can compute y^T y = sum(y^2).
    yty = np.sum(y.astype(np.float64)**2)
    # x^T b
    xtb = np.dot(x, b)
    # x^T A x
    xtAx = np.dot(x, np.dot(A, x))
    # RSS = yty - 2*xtb + xtAx
    rss = yty - 2*xtb + xtAx
    mse_new = rss / (N * H * W)

    # Also compute current MSE for reference
    pred_cur = f['prediction']
    mse_cur = np.mean((pred_cur - y)**2)

    diagnostics = {
        "mse_current": float(mse_cur),
        "mse_new_estimated": float(mse_new),
        "ridge_lambda": lam,
        "out_bias_new": float(out_b),
        "out_weight_norm": float(np.linalg.norm(out_w)),
    }

    return {"weights": w, "diagnostics": diagnostics}
