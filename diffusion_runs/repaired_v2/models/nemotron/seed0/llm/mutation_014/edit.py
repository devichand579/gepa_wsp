import numpy as np

def edit(w, f):
    """
    Closed-form least-squares update of the output layer (out.weight, out.bias)
    to minimize MSE between epsilon prediction and target Gaussian epsilon y.
    Uses only permitted NumPy linear algebra; no autograd, backprop, or iterative optimizers.
    """
    # ----------------------------------------------------------------------
    # Extract permitted feedback arrays (shapes as documented)
    # ----------------------------------------------------------------------
    features = f['features']          # [2000, 64, 16, 16]  final hidden feature maps
    prediction = f['prediction']     # [2000, 1, 16, 16]  current epsilon prediction
    y = f['y']                       # [2000, 1, 16, 16]  target Gaussian epsilon
    # ----------------------------------------------------------------------
    # We will update only the output layer weights/biases.
    # ----------------------------------------------------------------------
    # Output layer: out.weight [1, 64, 3, 3], out.bias [1]
    # Forward: conv2d(pad1) on 64-channel 16x16 input -> 1-channel 16x16 output.
    # ----------------------------------------------------------------------
    # Build normal-equation matrices by accumulating outer products of im2col patches.
    # Each output pixel at (u,v) is a dot product of 576 weights (64 channels × 3×3)
    # with the corresponding 3×3×64 patch (zero-padded input by 1 pixel).
    # ----------------------------------------------------------------------
    n_ex = features.shape[0]          # 2000
    n_c, h, w = features.shape[1], features.shape[2], features.shape[3]  # 64,16,16
    patch_size = 3 * n_c              # 576
    # Augmented dimension for bias (column of ones)
    d = patch_size + 1                # 577
    A = np.zeros((d, d), dtype=np.float32)
    c = np.zeros(d, dtype=np.float32)

    # Pre-allocate error map for vectorization
    # e[i,0,u,v] = prediction[i,0,u,v] - y[i,0,u,v]
    # We'll loop over examples and spatial positions.
    for i in range(n_ex):
        feat_i = features[i]                     # [64,16,16]
        # Zero-pad by 1 on height and width -> [64,18,18]
        padded = np.pad(feat_i, ((0,0),(1,1),(1,1)), mode='constant')
        # Iterate over all 16×16 output positions
        for u in range(h):
            for v in range(w):
                # Extract 3×3 patch from padded at (u,v) -> flatten to 576
                patch = padded[:, u:u+3, v:v+3].reshape(-1)  # [576]
                e_val = prediction[i, 0, u, v] - y[i, 0, u, v]
                # Augment with 1 for bias term
                patch_a = np.append(patch, 1.0)  # [577]
                # Accumulate A += patch_a ⊗ patch_a  (outer product)
                A += np.outer(patch_a, patch_a)
                # Accumulate c += patch_a * e_val
                c += patch_a * e_val

    # ----------------------------------------------------------------------
    # Solve normal equations: delta = A^{-1} c  (or use lstsable solve)
    # ----------------------------------------------------------------------
    # A is 577×577, well-conditioned for this scale; add tiny ridge if needed
    ridge = 1e-6 * np.eye(d, dtype=np.float32)
    A_reg = A + ridge
    delta = np.linalg.solve(A_reg, c)  # [577]

    # Split into weight delta and bias delta
    delta_weight = delta[:patch_size].reshape(1, n_c, 3, 3)  # [1,64,3,3]
    delta_bias   = delta[patch_size]                            # scalar

    # ----------------------------------------------------------------------
    # Apply update to the weight dictionary (preserve keys/shapes, clip)
    # ----------------------------------------------------------------------
    w = {k: v.copy() for k, v in w.items()}  # shallow copy of all arrays
    w['out.weight'] = np.clip(w['out.weight'] + delta_weight, -1000, 1000).astype(np.float32)
    w['out.bias']   = np.clip(w['out.bias']   + delta_bias,   -1000, 1000).astype(np.float32)

    # ----------------------------------------------------------------------
    # Diagnostics (small JSON‑compatible stats)
    # ----------------------------------------------------------------------
    # Compute achieved MSE on the full feedback batch before/after (optional)
    # Here we report the norm of the update and the residual MSE reduction proxy.
    pred = prediction.copy()
    # Re‑compute prediction with updated output layer (forward pass only, no backprop)
    # We implement the same conv2d pad1 logic using the updated weights.
    for i in range(n_ex):
        feat_i = features[i]
        padded = np.pad(feat_i, ((0,0),(1,1),(1,1)), mode='constant')
        # vectorized conv for this example
        # output [1,16,16] = sum_c sum_{kh,kw} w[c,kh,kw] * padded[c, u+kh-1, v+kw-1] + b
        # We'll use numpy stride/tricks for efficiency, but keep it simple:
        out_i = np.zeros((1, h, w), dtype=np.float32)
        for u in range(h):
            for v in range(w):
                # extract patch (same as above) and dot with weight + bias
                patch = padded[:, u:u+3, v:v+3].reshape(-1)  # [576]
                out_i[0, u, v] = np.dot(patch, w['out.weight'].reshape(-1)) + w['out.bias']
        pred[i] = out_i  # broadcast back to [1,16,16] -> we'll just store slice

    mse_before = np.mean((prediction - y) ** 2)
    mse_after  = np.mean((pred - y) ** 2)
    diag = {
        "update_weight_norm": float(np.linalg.norm(delta_weight)),
        "update_bias_delta":  float(delta_bias),
        "mse_before": float(mse_before),
        "mse_after":  float(mse_after),
        "mse_reduction": float(mse_before - mse_after)
    }

    # Return updated weights and diagnostics
    return {"weights": w, "diagnostics": diag}
