import numpy as np

def edit(w, f):
    """
    Closed-form ridge regression on the final layer (out.weight, out.bias) of a
    MNIST16x16 unconditional DDPM epsilon predictor. Uses cached features (output
    of conv3 ReLU) and target Gaussian epsilon from the feedback batch. No
    autograd, backprop, or iterative optimisation – only linear algebra and
    normal‑equation solutions.
    
    Parameters
    ----------
    w : dict
        Current network parameters with keys matching 'current_weights' from the
        task JSON (conv1.weight, conv1.bias, conv2.weight, conv2.bias, conv3.weight,
        conv3.bias, out.weight, out.bias). All arrays are float32; shapes must be
        preserved.
    f : dict
        Feedback arrays as specified: 'features' [N,64,16,16], 'y' [N,1,16,16],
        'batch.ids' [128], 'batch.y' [128,1,16,16], etc. Only permitted arrays
        are used.
    
    Returns
    -------
    w : dict
        Updated weights with out.weight and out.bias replaced by ridge‑fitted values.
    """
    # --- select a single 128‑example batch for stable resource usage ---
    N_batch = 128
    batch_ids = f['batch.ids'][:N_batch]  # assume ids are 0‑based indices into features/y
    features = f['features'][batch_ids]    # [128, 64, 16, 16]  (post‑conv3 ReLU)
    y_target = f['batch.y']                # [128, 1, 16, 16]   (target epsilon)

    # --- zero‑pad features so that conv3→out with padding=1 is exactly
    #     a standard cross‑correlation over the 18×18 padded map ---
    padded = np.pad(features, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode='constant', constant_values=0.0)  # [128, 64, 18, 18]

    # --- extract all 3×3 patches for every spatial position (16×16 per example)
    #     resulting design matrix X of shape [128*256, 64*3*3] = [32768, 576]
    # Use stride‑based view to avoid Python loops over 512k rows.
    # Reshape padded to (128*16*16, 64, 3, 3) by extracting patches via
    # sliding window over the last two axes.
    # We'll build X row‑by‑row in a memory‑efficient way using as_strided.
    n_ex, C, H, W = padded.shape  # 128, 64, 18, 18
    # Create a view of all 3x3 patches: (n_ex*16*16, 64, 3, 3)
    # First, transpose to (64, 128, 18, 18) then use stride tricks.
    # Simpler (and still fast enough): iterate over output pixels, accumulate.
    # Given 128*256 = 32768 rows, a Python loop is fine (< 0.5 s).
    
    n_pixels = H - 2  # 16
    n_total = n_ex * n_pixels * n_pixels  # 32768
    K = 64 * 3 * 3  # 576
    
    X = np.empty((n_total, K), dtype=np.float32)
    y_vec = np.empty(n_total, dtype=np.float32)
    
    idx = 0
    for ex in range(n_ex):
        # padded[ex] is [64, 18, 18]
        pf = padded[ex]  # view as (64, 18, 18)
        for ih in range(n_pixels):
            for iw in range(n_pixels):
                # patch from pf[:, ih:ih+3, iw:iw+3] -> (64, 3, 3)
                patch = pf[:, ih:ih+3, iw:iw+3]  # (64, 3, 3)
                X[idx] = patch.flatten()         # (576,)
                # target is y_target[ex, 0, ih, iw]
                y_vec[idx] = y_target[ex, 0, ih, iw]
                idx += 1

    # --- add bias term ---
    X_b = np.column_stack([X, np.ones(n_total, dtype=np.float32)])  # [32768, 577]

    # --- ridge normal equation: (XᵀX + λI)β = Xᵀy  with small λ for stability ---
    lam = 1e-3  # ridge coefficient; does not affect the fitted function meaningfully
    XtX = X_b.T @ X_b  # [577, 577]
    Xty = X_b.T @ y_vec  # [577,]
    I = np.eye(K + 1, dtype=np.float32) * lam
    beta = np.linalg.solve(XtX + I, Xty)  # [577,]

    # --- split into weight vector and bias ---
    w_out_flat = beta[:-1].reshape(1, 64, 3, 3).astype(np.float32)  # [1, 64, 3, 3]
    b_out = np.float32(beta[-1])

    # --- clip to finite float32, |val| <= 1000 ---
    w_out = np.clip(w_out_flat, -1000, 1000).astype(np.float32)
    b_out = np.clip(b_out, -1000, 1000).astype(np.float32)

    # --- update only the output layer; keep all other weights unchanged ---
    w = dict(w)  # shallow copy to preserve original keys/shapes
    w['out.weight'] = w_out
    w['out.bias'] = b_out

    # --- diagnostics (small JSON‑compatible dict) ---
    diagnostics = {
        "ridge_lambda": float(lam),
        "out_weight_mean": float(np.mean(np.abs(w_out))),
        "out_weight_std": float(np.std(w_out)),
        "out_bias": float(b_out),
        "fit_mse": float(np.mean((y_target - np.clip(
            np.zeros_like(y_target), -1, 1))**2))  # placeholder; actual MSE after edit
    }
    # compute actual MSE on the batch after edit (no autograd, just forward via conv)
    # we can't run full forward here without recomputing all activations, 
    # so we report the closed‑form residual:
    pred_flat = (X @ w_out.flatten() + b_out).reshape(n_total)
    resids = y_vec - pred_flat
    diag_mse = float(np.mean(resids**2))
    diagnostics["fit_mse"] = diag_mse
    diagnostics["n_examples"] = int(n_ex)
    diagnostics["n_patches"] = int(n_total)

    # attach diagnostics as an extra key; the task allows returning
    # {"weights":w,"diagnostics":small_dict}
    return {"weights": w, "diagnostics": diagnostics}
