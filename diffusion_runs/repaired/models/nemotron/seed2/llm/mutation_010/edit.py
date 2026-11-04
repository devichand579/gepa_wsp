import numpy as np

def edit(w, f):
    """
    Closed‑form edit of the DDPM epsilon‑prediction network.
    Only the output convolution (out.weight + out.bias) is updated;
    all other parameters are left untouched.
    """
    # ------------------------------------------------------------------
    # 1.  Gather feedback arrays (shapes are fixed by the task)
    # ------------------------------------------------------------------
    features = f['features']            # (N, 64, 16, 16), N = 2000
    target   = f['y']                   # (N, 1, 16, 16)
    # current prediction is available for diagnostics but not needed for the fit
    # prediction = f['prediction']

    N, C, H, W = features.shape        # C=64, H=W=16
    lam = 1e-6                         # tiny ridge regulariser (prevents singularity)

    # ------------------------------------------------------------------
    # 2.  Pre‑pad feature maps so that “same‑padding” 3×3 kernels can be
    #     extracted without worrying about border handling.
    # ------------------------------------------------------------------
    pad_width = ((0, 0), (0, 0), (1, 1), (1, 1))
    features_padded = np.pad(features, pad_width, mode='constant',
                            constant_values=0.)   # (N,64,18,18)

    # ------------------------------------------------------------------
    # 3.  For each output pixel (i,j) solve a small ridge‑regression problem:
    #         min_w ‖ A_n(w) – y_n(i,j) ‖²   over the N examples.
    #     A_n contains the 3×3×64 patch flattened to 576 entries.
    #     The solution is w_vec = (AᵀA + λI)⁻¹ Aᵀ y.
    # ------------------------------------------------------------------
    sum_res = 0.0          # accumulated (target – output) over all (n,i,j)
    weight_tensors = []   # list of the 256 × (1,64,3,3) weight matrices

    for i in range(H):
        for j in range(W):
            # ---- build design matrix A (N × 576) -------------------------
            patches = features_padded[:, :, i:i+3, j:j+3]   # (N,64,3,3)
            A = patches.reshape(N, -1)                     # (N,576)

            # ---- target values for this spatial position ----------------
            b = target[:, 0, i, j]                         # (N,)

            # ---- normal equations with ridge -----------------------------
            AtA = A.T @ A                                   # (576,576)
            AtA_reg = AtA + lam * np.eye(576, dtype=np.float32)
            Atb = A.T @ b                                   # (576,)
            w_vec = np.linalg.solve(AtA_reg, Atb)          # (576,)

            # ---- compute output for this position -----------------------
            output_n = A @ w_vec                           # (N,)
            res = np.sum(b - output_n)                     # scalar
            sum_res += res

            # ---- keep the weight tensor for later averaging ------------
            w_mat = w_vec.reshape(1, 64, 3, 3).astype(np.float32)
            weight_tensors.append(w_mat)

    # ------------------------------------------------------------------
    # 4.  Average the 256 position‑specific weight tensors → new out.weight
    # ------------------------------------------------------------------
    new_out_weight = np.mean(weight_tensors, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # 5.  Compute the new bias from the accumulated residuals.
    #     bias = (1/(N·H·W)) * Σ_{n,i,j} ( y_{n,i,j} – output_{n,i,j} )
    # ------------------------------------------------------------------
    new_bias = (sum_res / (N * H * W)).astype(np.float32)

    # ------------------------------------------------------------------
    # 6.  Apply the updates, clip to allowable range, preserve shapes.
    # ------------------------------------------------------------------
    w['out.weight'] = np.clip(new_out_weight, -1000, 1000)
    w['out.bias']   = new_bias

    # ------------------------------------------------------------------
    # 7.  Diagnostics: MSE before/after, weight statistics.
    # ------------------------------------------------------------------
    # (re‑compute the prediction with the new weights for the diagnostics)
    # build the full output tensor (no bias yet) using the averaged weight
    # we reuse the already‑filled A matrices: for each (n,i,j) the output
    # is A[n,:] @ w_vec_for_that_position; we already have the averaged
    # weight, so we can compute a single output image.
    # Efficient vectorised version:
    X = np.empty((N * H * W, 576), dtype=np.float32)
    idx = 0
    for n in range(N):
        for i in range(H):
            for j in range(W):
                patches = features_padded[n:n+1, :, i:i+3, j:j+3]  # (1,64,3,3)
                X[idx, :] = patches.reshape(-1)
                idx += 1
    w_flat = new_out_weight.reshape(-1)
    output_flat = X @ w_flat                     # (N·H·W,)
    output_img = output_flat.reshape(N, 1, H, W).astype(np.float32)

    pred_after = output_img + new_bias           # (N,1,16,16)

    mse_before = float(np.mean((f['prediction'] - target)**2))
    mse_after  = float(np.mean((pred_after - target)**2))

    diagnostics = {
        "out_weight_mean": float(np.mean(w['out.weight'])),
        "out_weight_std":  float(np.std(w['out.weight'])),
        "out_bias":        float(w['out.bias']),
        "mse_before":      mse_before,
        "mse_after":       mse_after,
        "reduction":       mse_before - mse_after,
    }

    # ------------------------------------------------------------------
    # 8.  Return the modified weight dictionary together with diagnostics.
    # ------------------------------------------------------------------
    return {"weights": w, "diagnostics": diagnostics}
