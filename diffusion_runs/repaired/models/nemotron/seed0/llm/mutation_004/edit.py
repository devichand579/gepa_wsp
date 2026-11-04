import numpy as np

def edit(w, f):
    """
    Closed‑form weight update for the output convolution of a DDPM epsilon‑prediction network.
    Only `out.weight` and `out.bias` are changed; all earlier layers remain untouched.
    """
    # make a full‑precision copy of the current weights
    w_new = {k: v.astype(np.float32, copy=True) for k, v in w.items()}

    # ----- feedback arrays we are allowed to use -----
    features = f['features']          # [N, 64, 16, 16]   final hidden feature maps
    prediction = f['prediction']      # [N, 1, 16, 16]   current ε‑prediction
    y = f['y']                      # [N, 1, 16, 16]   target Gaussian ε

    N = features.shape[0]            # number of training examples (2000)

    # per‑pixel error e = prediction - y
    e = prediction - y                # [N, 1, 16, 16]

    # ----- pad features so that a 3×3 convolution with pad=1 is equivalent
    #         to extracting centred 3×3 windows with zero‑padding -----
    F_pad = np.pad(features, ((0,0),(0,0),(1,1),(1,1)),
                   mode='constant', constant_values=0)   # [N,64,18,18]

    # ----- build the design matrix X (im2col) -----
    # each row corresponds to one spatial pixel (there are 16×16 = 256 per example)
    # and contains the 64·3·3 = 576 values from the receptive field.
    X_rows = []
    for i in range(16):
        for j in range(16):
            # extract the 3×3 window at position (i,j) from the padded feature maps
            patch = F_pad[:, :, i:i+3, j:j+3]          # [N,64,3,3]
            patch_flat = patch.reshape(N, -1)          # [N, 576]
            X_rows.append(patch_flat)
    X = np.vstack(X_rows)                # [N*256, 576]

    e_flat = e.reshape(-1)               # [N*256,]   stacked errors

    # ----- closed‑form least‑squares solution for a single gradient step -----
    # we want Δw, Δb such that  conv(F, Δw) + Δb ≈ -e  (first‑order correction)
    # the normal equations give Δw = -(Xᵀ e) / (N·256)
    delta_w = -(X.T @ e_flat) / (N * 256)          # [576,]
    delta_w = delta_w.reshape(w_new['out.weight'].shape)   # [1,64,3,3]

    # update output weight tensor
    w_new['out.weight'] = w_new['out.weight'] + delta_w

    # update output bias (scalar) – subtract the mean error
    delta_b = -float(np.mean(e_flat))
    w_new['out.bias'] = w_new['out.bias'] + delta_b

    # ----- enforce finite float32 and absolute bound ≤ 1000 -----
    for k in w_new:
        arr = w_new[k]
        # replace NaN / Inf with 0
        arr = np.where(np.isnan(arr) | np.isinf(arr), 0.0, arr)
        # clip to [-1000, 1000]
        arr = np.clip(arr, -1000, 1000)
        # be sure the dtype stays float32
        w_new[k] = arr.astype(np.float32)

    # return the updated weight dictionary (no further training occurs)
    return w_new
