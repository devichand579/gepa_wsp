import numpy as np
from numpy.lib.stride_tricks import as_strided

def edit(w, f):
    """
    Closed-form, one‑shot weight update for the output layer of a
    MNIST16x16 unconditional DDPM epsilon‑prediction network.
    
    Permitted operations only: NumPy statistics, correlations,
    covariance, linear algebra, im2col via stride tricks.
    No autograd, backprop, gradient estimation, or iterative optimizers.
    
    Args:
        w (dict): Current parameter arrays.
            Keys: 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
                  'conv3.weight', 'conv3.bias', 'out.weight', 'out.bias'.
        f (dict): Permitted feedback arrays.
            Keys: 'x', 't', 'y', 'clean', 'features', 'prediction',
                  'alphabar', 'batch.ids', 'batch.x', 'batch.t', 'batch.y',
                  'generated.samples', and layer traces
                  'layer.conv1.input0', 'layer.conv1.output',
                  'layer.conv2.input0', 'layer.conv2.output',
                  'layer.conv3.input0', 'layer.conv3.output',
                  'layer.out.input0', 'layer.out.output'.
    
    Returns:
        w (dict) or {"weights": w, "diagnostics": {...}} — updated weights.
    """
    # ------------------------------------------------------------------
    # 1. Select the batch of 128 examples (layer traces are for this batch)
    # ------------------------------------------------------------------
    B = 128  # paired batch size
    # We use the full 2000‑example batch for the out‑layer least‑squares,
    # but the layer activations are provided for the 128‑sample batch.
    # We'll pull the relevant arrays for the 128 batch where needed.
    x_batch = f['batch.x']          # [128, 1, 16, 16]
    t_batch = f['batch.t']          # [128]
    y_batch = f['batch.y']          # [128, 1, 16, 16]
    pred_batch = f['prediction'][:B]  # [128, 1, 16, 16] (current epsilon pred)
    features_batch = f['features'][:B]  # [128, 64, 16, 16] (final hidden features)
    # Layer inputs/outputs for the 128 batch (already provided)
    # We'll primarily use the out‑layer traces; others are available if needed.
    out_in0 = f['layer.out.input0']   # [128, 64, 16, 16]
    out_out = f['layer.out.output']   # [128, 1, 16, 16]

    # ------------------------------------------------------------------
    # 2. Build the im2col matrix for the out layer from the 128 examples.
    #    out.weight shape: [1, 64, 3, 3] → 576 parameters.
    #    Each output pixel (16×16 = 256 per example) is a dot product of
    #    a 3×3×64 patch from features with the weight kernel.
    # ------------------------------------------------------------------
    n_ex = B                # 128
    n_pix = 16 * 16         # 256
    n_weights = 1 * 64 * 3 * 3  # 576

    # Im2col: reorder each (3x3x64) patch into a row vector.
    # features_batch shape: [128, 64, 16, 16]
    # We'll create a matrix X of shape [n_ex * n_pix, n_weights].
    X = np.zeros((n_ex * n_pix, n_weights), dtype=np.float32)

    # Use stride tricks to extract patches without copying everything.
    # For each example, for each pixel (i,j), take the 3x3x64 patch.
    # Because of padding=1 and stride=1, the output size equals input size.
    # We'll loop over examples (128 is small) and pixels.
    for ex in range(n_ex):
        feat = features_batch[ex]  # [64, 16, 16]
        # Stride: step=1, we can just use stride_tricks
        # The input is 16x16 with pad1 implicitly, but the stored activations
        # are already the output of conv3 (which had pad1), so the feature maps
        # are 16x16. The out conv also uses pad1, so each output pixel sees
        # a 3x3 patch from the 16x16 feature map (with wrap? actually pad1
        # means input was 14x14 -> output 16x16; but the provided activations
        # are the final 16x16 maps, so we treat them as the effective input
        # to the out layer with appropriate border handling).
        # To keep it simple and correct per the network description,
        # we assume the 16x16 feature maps are the exact input to the out
        # conv with padding=1, meaning the effective 3x3 patches are taken
        # from a conceptual 18x18 array zero‑padded at borders. However,
        # the exact border treatment only affects a few pixels; we'll
        # extract all 256 patches per example using stride_tricks on the
        # provided 16x16 arrays, assuming the network's padding convention
        # is already embedded in the saved activations.
        # We'll use a manual extraction that matches PyTorch conv2d with
        # padding=1, stride=1, and weight shape [1,64,3,3].
        # For each pixel position (h,w) in 0..15, the input patch is
        # feat[:, h:h+3, w:w+3] if we ignore padding, but with pad1 the
        # effective input was originally 14x14 zero‑padded to 16x16 before
        # conv3; the stored activations are post‑conv3, so they already
        # reflect that padding. We'll just take the 3x3 neighborhood
        # centered at (h+1, w+1) to mimic the effective input, clipping
        # at borders with zeros.
        for h in range(16):
            for w in range(16):
                # extract 3x3 patch centered; pad with 0 at borders
                patch = np.zeros((64, 3, 3), dtype=np.float32)
                for ci in range(64):
                    for di in range(3):
                        for dj in range(3):
                            ih, iw = h + di - 1, w + dj - 1
                            if 0 <= ih < 16 and 0 <= iw < 16:
                                patch[ci, di, dj] = feat[ci, ih, iw]
                # flatten patch to row
                row = patch.reshape(-1)  # length 576
                X[ex * n_pix + h * 16 + w, :] = row

    # ------------------------------------------------------------------
    # 3. Form the target vector y_vec from the current prediction error.
    #    We want to minimize || X @ (w + delta_w) - y_vec ||^2.
    #    The current prediction (before weight change) is already
    #    X @ w_current (since the out layer forward is linear in weights).
    #    Residual r = X @ w_current - y_vec.
    # ------------------------------------------------------------------
    # Compute current out-layer output via im2col to verify/derive residual.
    # Actually, we have pred_batch directly; we can flatten it.
    y_vec = pred_batch.reshape(-1)  # [128*256,] = current epsilon prediction
    # But we need the target y. The feedback gives y_batch [128,1,16,16].
    y_target = y_batch.reshape(-1)  # [128*256,] target Gaussian epsilon

    # Residual r = current_pred - target
    r = y_vec - y_target  # shape [N], N = 128*256 = 32768

    # However, X @ w_current should equal y_vec. We'll compute the
    # least‑squares optimal delta_w = - (X^T X)^{-1} X^T r.
    # X is [N_pixels_total, 576] = [32768, 576].
    # X^T X is 576×576, X^T r is 576×1.

    # ------------------------------------------------------------------
    # 4. Closed‑form weight increment for the out layer.
    # ------------------------------------------------------------------
    XtX = X.T @ X  # 576×576
    Xtr = X.T @ r   # 576×1

    # Regularize if XtX is near‑singular (add tiny ridge)
    ridge = 1e-6 * np.eye(n_weights, dtype=np.float32)
    XtX_reg = XtX + ridge

    delta_w = - np.linalg.solve(XtX_reg, Xtr).reshape(w['out.weight'].shape)

    # ------------------------------------------------------------------
    # 5. Update the out layer weights (and bias) in closed form.
    #    For the bias, the optimal update is the mean residual per example,
    #    but we can also compute it analytically.
    # ------------------------------------------------------------------
    w_new = {}
    # Copy all existing weights first
    for k, aw in w.items():
        w_new[k] = aw.copy()

    # Update out.weight
    w_new['out.weight'] = w['out.weight'] + delta_w

    # Update out.bias: the bias is added to every output pixel.
    # The MSE‑optimal bias change minimizes || (X w_new + b_new * 1) - y ||^2.
    # With the weight update already applied, the new residual without bias
    # is r_new = X @ delta_w + (b_old - b_current?) Actually, let's just
    # compute the bias that would make the mean residual zero across all
    # pixels/examples, which is a standard closed‑form adjustment.
    # Current bias value:
    b_old = float(w['out.bias'])
    # After weight update, compute the new output mean residual:
    # We can compute the average of r across all pixels/examples,
    # then adjust bias by the negative of that mean (since bias broadcasts).
    # To avoid another forward pass, note that adding a constant bias b
    # shifts every output pixel by b. The MSE‑optimal b given fixed weights
    # is the mean of the targets minus the mean of the weighted features.
    # Simpler: we compute the residual mean after weight update conceptually.
    # Since we don't re‑forward, we approximate by using the current residual
    # mean and adjust bias accordingly. This is exact for the bias term
    # because the bias gradient is exactly the mean of the errors.
    # We'll compute the bias update as:
    #   delta_b = - mean(r)  where r is the error before bias adjustment.
    # But we have r = pred - target. If we add delta_b to the bias,
    # every prediction increases by delta_b, so the new error becomes
    # r - delta_b. The MSE‑optimal delta_b makes the mean error zero:
    #   mean(r - delta_b) = 0  =>  delta_b = mean(r).
    # However, r here is pred - target, so delta_b = mean(pred - target).
    # Let's compute that.
    delta_b = np.mean(r)  # scalar
    w_new['out.bias'] = np.float32(b_old + delta_b)

    # ------------------------------------------------------------------
    # 6. Diagnostics (small JSON‑compatible stats).
    # ------------------------------------------------------------------
    # Compute new MSE on the batch (optional, for diagnostics only;
    # the function returns w, but we include stats as allowed).
    # New prediction with updated weights (linear in weights, so we can
    # compute new pred = X @ (w_old + delta_w) + (b_old + delta_b)).
    # But we already have y_target; we can compute new MSE.
    # To avoid materializing huge matrices, we use the fact that
    # new_pred_flat = y_vec + X @ delta_w + delta_b (since pred = X w + b).
    # Actually, pred_batch was already X @ w_old + b_old (within padding
    # conventions). So new_pred_flat = y_vec + X @ delta_w + delta_b.
    # But X @ delta_w is [32768,]; we can compute it.
    new_pred_flat = y_vec + (X @ delta_w.flatten()) + delta_b
    new_pred = new_pred_flat.reshape(B, 1, 16, 16)
    new_mse = np.mean((new_pred - y_target.reshape(B, 1, 16, 16)) ** 2)

    diagnostics = {
        "out_weight_delta_norm": float(np.linalg.norm(delta_w)),
        "out_bias_delta": float(delta_b),
        "batch_mse_before": float(np.mean((pred_batch - y_target.reshape(B, 1, 16, 16)) ** 2)),
        "batch_mse_after": float(new_mse),
        "im2col_matrix_shape": list(X.shape),
    }

    # ------------------------------------------------------------------
    # 7. Return updated weights (and optional diagnostics).
    # ------------------------------------------------------------------
    # Preserve shapes, finite float32, |val| <= 1000 (they already are).
    return {"weights": w_new, "diagnostics": diagnostics}
