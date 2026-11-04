import numpy as np

def edit(w, f):
    """
    Update only the output‑layer weights (out.weight, out.bias) using
    statistical closed‑form fitting on the provided feedback arrays.
    No autograd, back‑prop, or iterative optimiser is used.
    """
    # ----- permitted feedback arrays -----
    features = f['features']          # shape (2000, 64, 16, 16)
    prediction = f['prediction']     # shape (2000, 1, 16, 16)
    y = f['y']                       # shape (2000, 1, 16, 16)

    e = prediction - y               # prediction error, shape (2000, 1, 16, 16)
    e_squeeze = e[:, 0]             # (2000, 16, 16)

    N = features.shape[0]            # 2000 examples
    # ----- accumulate per‑weight gradient & second moment -----
    g = np.zeros(576, dtype=np.float32)   # gradient w.r.t. out.weight (1*64*3*3)
    h_diag = np.zeros(576, dtype=np.float32)  # per‑weight ⟨patch²⟩

    for n in range(N):
        feat = features[n]            # (64, 16, 16)
        err = e_squeeze[n]            # (16, 16)

        # pad with 1 zero pixel so that 3×3 windows keep 16×16 output size
        feat_pad = np.pad(feat, ((0, 0), (1, 1), (1, 1)),
                          mode='constant', constant_values=0.0)  # (64, 18, 18)

        # extract all 16×16 = 256 patches, flatten each to 64*9 = 576 elements
        patches = []
        for i in range(16):
            for j in range(16):
                patch = feat_pad[:, i:i+3, j:j+3].reshape(64, -1)  # (64, 9)
                patches.append(patch)
        patches_flat = np.stack(patches, axis=0)          # (256, 64, 9)
        patches_flat = patches_flat.reshape(256, -1)      # (256, 576)

        err_flat = err.ravel()                           # (256,)

        # accumulate outer‑product / sum‑of‑squares
        g += patches_flat.T @ err_flat                  # (576,)
        h_diag += (patches_flat ** 2).sum(axis=0)       # (576,)

    total = N * 256
    g /= total
    h_diag /= total

    # diagonal‑Fisher step size per weight
    lr = np.where(h_diag > 1e-8, 1.0 / h_diag, 0.0)
    delta_w = -lr * g                                   # (576,)

    # ----- update output weight -----
    out_weight = w['out.weight'].copy().astype(np.float32)
    out_weight += delta_w.reshape(out_weight.shape)
    out_weight = np.clip(out_weight, -1000, 1000).astype(np.float32)

    # ----- update output bias -----
    bias_grad = float(np.mean(e_squeeze))               # scalar mean error
    delta_b = -bias_grad                                 # lr = 1 for bias
    out_bias = w['out.bias'].copy().astype(np.float32)
    out_bias += delta_b
    out_bias = np.clip(out_bias, -1000, 1000).astype(np.float32)

    # ----- return updated weights (other layers untouched) -----
    w_out = {k: v for k, v in w.items()}
    w_out['out.weight'] = out_weight
    w_out['out.bias'] = out_bias

    diagnostics = {
        'weight_update_norm': float(np.linalg.norm(delta_w)),
        'bias_update': delta_b,
        'mean_MSE_before': float(np.mean((prediction - y) ** 2)),
    }

    return w_out, diagnostics
