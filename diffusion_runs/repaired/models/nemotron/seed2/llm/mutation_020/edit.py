import numpy as np

def edit(w, f):
    """
    Closed‑form weight update for the output layer of a DDPM epsilon‑prediction network.
    Only the output convolution (`out.weight`, `out.bias`) is modified.
    The update minimises the mean‑squared error between `prediction` and `y`
    using a ridge‑regression normal equation built from all output‑pixel patches
    extracted from the saved feature maps.
    """
    # ------------------------------------------------------------------
    # 1. Pull the required feedback arrays (keys are guaranteed to exist)
    # ------------------------------------------------------------------
    # Shapes (from the problem statement):
    #   x:      [2000, 1, 16, 16]   noisy x_t
    #   t:      [2000]               timestep indices
    #   y:      [2000, 1, 16, 16]    target Gaussian epsilon
    #   features:[2000, 64, 16, 16]  final hidden feature maps (after conv3)
    #   prediction:[2000, 1, 16, 16] current epsilon prediction
    #   alphabar:[32]                32‑step cosine schedule coefficients
    #   batch.* etc. – not needed for this edit
    x       = f['x']            # [N,1,16,16]
    t       = f['t']            # [N]
    y       = f['y']            # [N,1,16,16]
    features= f['features']     # [N,64,16,16]
    prediction = f['prediction']# [N,1,16,16]
    alphabar= f['alphabar']     # [32]

    N, C, H, W = features.shape   # N=2000, C=64, H=W=16
    # ------------------------------------------------------------------
    # 2. Form the error signal e = y – prediction  (same shape as y)
    # ------------------------------------------------------------------
    error = y - prediction        # [N,1,H,W]  (float32)

    # ------------------------------------------------------------------
    # 3. Prepare patches for the output conv (kernel 3×3, padding=1)
    #    The out conv receives `features` (64 channels) and produces a
    #    single‑channel output.  With padding=1 the output spatial size
    #    stays 16×16.  We zero‑pad the feature maps on all sides, then
    #    slide a 3×3 window so that output position (h,w) uses the patch
    #    features_pad[:, :, h:h+3, w:w+3].
    # ------------------------------------------------------------------
    pad = 1
    features_pad = np.pad(features,                     # [N,64,16,16]
                         ((0,0),(0,0),(pad,pad),(pad,pad)),
                         mode='constant',
                         constant_values=0)    # -> [N,64,18,18]

    # Use stride‑tricks to obtain a view of all 3×3 patches without copying.
    # sliding_window_view works on the last two axes.
    from numpy.lib.stride_tricks import sliding_window_view
    # shape becomes [N,64,H,W,3,3] – the last two dims are the window
    patches = sliding_window_view(features_pad, window_shape=(3,3), axis=(-2,-1))
    # patches[n,c,h,w,kh,kw] is the feature value at the (kh,kw) offset
    # for output position (h,w) of example n.

    # ------------------------------------------------------------------
    # 4. Flatten everything to per‑pixel samples.
    # ------------------------------------------------------------------
    # error_flat: [M] where M = N * H * W
    error_flat = error.reshape(-1)                # shape (N*H*W,)

    # patches_flat: [M, C*3*3] = [M, 576]  (C=64, 3*3=9)
    # Reshape by moving the spatial dimensions into the first axis.
    patches_flat = patches.reshape(N * H * W, C * 3 * 3)  # (M, 576)

    # ------------------------------------------------------------------
    # 5. Closed‑form ridge regression for each of the 576 output‑weight
    #    parameters (and the bias).  We solve  (XᵀX + λI) w = Xᵀ y
    #    where X = patches_flat,  y = error_flat,  λ = 1e‑3 (small ridge).
    # ------------------------------------------------------------------
    lam = 1e-3                     # ridge regularisation (prevents division by zero)
    XtX = patches_flat.T @ patches_flat          # (576, 576)
    Xty = patches_flat.T @ error_flat            # (576,)

    # Add ridge to the diagonal
    XtX_reg = XtX + lam * np.eye(XtX.shape[0], dtype=np.float32)

    # Solve the normal equations once (576×576 system – tiny)
    delta_w_flat = np.linalg.solve(XtX_reg, Xty)   # (576,)

    # Reshape back to the original weight tensor [1,64,3,3]
    delta_weight = delta_w_flat.reshape(1, C, 3, 3).astype(np.float32)

    # Bias update: scalar bias = mean(error) / (N*H*W + λ)  (closed‑form)
    bias_delta = float(np.sum(error_flat)) / (N * H * W + lam)

    # ------------------------------------------------------------------
    # 6. Apply the updates to the current weights (in‑place on a copy)
    # ------------------------------------------------------------------
    new_w = {k: v.copy() for k, v in w.items()}   # preserve all keys/shapes

    # out.weight shape is [1,64,3,3]
    new_w['out.weight'] = (w['out.weight'].astype(np.float32) + delta_weight)
    # Clip to the allowed absolute‑value bound (≤1000) just in case
    new_w['out.weight'] = np.clip(new_w['out.weight'], -1000, 1000)

    # out.bias shape is [1]
    new_w['out.bias'] = (w['out.bias'].astype(np.float32) + bias_delta)
    new_w['out.bias'] = np.clip(new_w['out.bias'], -1000, 1000)

    # ------------------------------------------------------------------
    # 7. Diagnostics (tiny JSON‑compatible dict)
    # ------------------------------------------------------------------
    # Compute the MSE before and after the edit on the same batch for reporting.
    pred_before = (w['out.weight'] * np.ones_like).sum()  # placeholder – we just report the change magnitude
    # Actually compute actual MSE using the updated weights would require a forward pass,
    # which is disallowed after the edit.  We instead report the norm of the update
    # and the regularisation parameter.
    diagnostics = {
        "weight_update_norm": float(np.linalg.norm(delta_weight)),
        "bias_update": float(bias_delta),
        "ridge_lambda": float(lam),
        "num_samples": int(N * H * W),
        "mse_before": float(np.mean(error**2)),   # MSE before edit (still valid)
    }

    # ------------------------------------------------------------------
    # 8. Return the modified weights and diagnostics
    # ------------------------------------------------------------------
    return {"weights": new_w, "diagnostics": diagnostics}
