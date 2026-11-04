import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    """
    Closed‑form edit of the DDPM epsilon‑prediction network.
    Only the output convolutional layer (out.weight, out.bias) is modified.
    No autograd, back‑propagation, or iterative optimisation is used.
    """

    # ------- extract permitted feedback arrays -------
    features = f['features']          # [N, 64, 16, 16]
    y        = f['y']                 # [N, 1, 16, 16]   target Gaussian ε
    N        = features.shape[0]

    # ------- constants -------
    chunk_size = 100                  # keep working sets small
    lambda_reg = 1e-6                 # ridge regularisation (prevents singularity)

    # ------- first pass: accumulate sufficient statistics -------
    # XtX  = Σ (patchesᵀ patches)   (576 × 576)
    # Xty  = Σ (patchesᵀ y_flat)    (576,)
    # total_y = Σ y over all pixels
    # total_pixels = number of output pixels (N * 256)
    XtX = np.zeros((576, 576), dtype=np.float32)
    Xty = np.zeros(576,      dtype=np.float32)
    total_y   = 0.0
    total_pixels = 0

    for i in range(0, N, chunk_size):
        chunk_features = features[i:i+chunk_size]
        chunk_y        = y[i:i+chunk_size]
        cN = chunk_features.shape[0]

        for j in range(cN):
            feat = chunk_features[j]                # (64, 16, 16)

            # emulate padding=1 used by the network
            feat_pad = np.pad(feat, ((1,1),(1,1)),
                              mode='constant', constant_values=0.0)   # (64,18,18)

            # extract every 3×3 patch from each input channel
            patches = sliding_window_view(feat_pad, (3,3),
                                          axis=(1,2))           # (16,16,3,3,64)
            patches = patches.transpose(0,1,4,2,3)   # (16,16,64,3,3)
            patches_flat = patches.reshape(-1, 64*3*3)   # (256,576)

            # accumulate normal‑equation terms
            XtX += patches_flat.T @ patches_flat
            y_flat = chunk_y[j].reshape(-1)           # (256,)
            Xty += patches_flat.T @ y_flat

            total_y   += np.sum(y_flat)
            total_pixels += patches_flat.shape[0]     # 256 per example

    # ------- ridge regularisation & closed‑form solution -------
    XtX += lambda_reg * np.eye(576, dtype=np.float32)
    w_vec = np.linalg.solve(XtX, Xty)          # (576,) optimal weight vector

    # ------- reshape to the layer's weight tensor -------
    w_new_weight = w_vec.reshape(1, 64, 3, 3).astype(np.float32)   # [1,64,3,3]

    # ------- second pass: compute the optimal bias -------
    total_Xw = 0.0
    for i in range(0, N, chunk_size):
        chunk_features = features[i:i+chunk_size]
        cN = chunk_features.shape[0]
        for j in range(cN):
            feat = chunk_features[j]
            feat_pad = np.pad(feat, ((1,1),(1,1)),
                              mode='constant', constant_values=0.0)
            patches = sliding_window_view(feat_pad, (3,3),
                                          axis=(1,2))
            patches = patches.transpose(0,1,4,2,3)
            patches_flat = patches.reshape(-1, 64*3*3)

            y_flat = f['y'][i+j].reshape(-1)      # (256,)
            total_Xw += np.sum(patches_flat @ w_vec)

    bias_new = (total_y - total_Xw) / total_pixels
    bias_new = np.array([bias_new], dtype=np.float32)   # shape [1]

    # ------- copy original weights and overwrite the output layer -------
    w_new = {k: v.copy() for k, v in w.items()}   # shallow copy of all arrays
    w_new['out.weight'] = w_new_weight
    w_new['out.bias']   = bias_new

    # ------- diagnostics -------
    diagnostics = {
        "solved_weight_norm": float(np.linalg.norm(w_vec)),
        "bias": float(bias_new[0]),
        "regularization": lambda_reg,
        "examples_used": N,
        "pixels_used": total_pixels
    }

    return {"weights": w_new, "diagnostics": diagnostics}
