import numpy as np

def edit(w, f):
    """
    One‑shot weight update for the DDPM epsilon‑prediction network.
    Uses only the permitted feedback arrays (x, t, y, features, prediction,
    alphabar) and current weights w.  No autograd / backprop / iterative
    optimiser is employed; the gradient is formed analytically from the
    provided activations and error.

    Parameters
    ----------
    w : dict
        Current network parameters with keys
        'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias',
        'conv3.weight', 'conv3.bias', 'out.weight', 'out.bias'.
    f : dict
        Feedback arrays:
        - x: noisy x_t [N,1,16,16]
        - t: timesteps [N]
        - y: target Gaussian epsilon [N,1,16,16]
        - features: final hidden feature maps [N,64,16,16]
        - prediction: current epsilon prediction [N,1,16,16]
        - alphabar: 32‑step schedule coefficients [32]
        plus batch‑ids etc. (ignored for the update).

    Returns
    -------
    {"weights": new_w, "diagnostics": {...}}
    """
    # ----- unpack feedback ----------------------------------------------------
    N = f['prediction'].shape[0]                # 2000 examples
    y = f['y']                                  # [N,1,16,16]
    pred = f['prediction']                      # [N,1,16,16]
    features = f['features']                    # [N,64,16,16]

    # ----- current output‑layer parameters -----------------------------------
    out_w = w['out.weight']                     # shape (1,64,3,3)
    out_b = w['out.bias']                       # shape (1,)

    # ----- error signal -------------------------------------------------------
    err = pred - y                              # [N,1,16,16]
    err_sq = err.squeeze(1)                     # [N,16,16]

    # ----- learning rate (fixed small value) ---------------------------------
    lr = 0.01

    # ----- update output bias -------------------------------------------------
    out_b_new = out_b - lr * np.mean(err)

    # ----- prepare padded features so that 3×3 windows exist at every
    #       of the 16×16 spatial positions (same‑padding convention) -----------
    feat_pad = np.pad(features, ((0,0),(0,0),(1,1),(1,1)),
                     mode='constant', constant_values=0)
    # feat_pad shape: (N,64,18,18)

    # ----- accumulate the gradient of MSE w.r.t. out.weight -------------------
    #    dMSE/dout_w = (1/(N*H*W)) * Σ_n Σ_i,j err[n,i,j] * patch[n,:,i:i+3,j:j+3]
    grad_num = np.zeros((64,3,3), dtype=np.float32)

    for i in range(16):
        for j in range(16):
            # extract the 3×3 patch at (i,j) for all examples and all input channels
            patch = feat_pad[:, :, i:i+3, j:j+3]          # (N,64,3,3)
            err_ij = err_sq[:, i, j]                     # (N,)
            # broadcast error over the patch and sum over the batch dimension
            weighted = err_ij[:, None, None, None] * patch   # (N,64,3,3)
            grad_num += weighted.sum(axis=0)                # (64,3,3)

    N_total = N * 16 * 16          # 2000 * 256 = 512 000
    grad = grad_num / N_total      # (64,3,3)
    grad_reshaped = grad.reshape(1,64,3,3)   # (1,64,3,3) → matches out.weight

    # ----- update output weight ------------------------------------------------
    out_w_new = out_w - lr * grad_reshaped
    out_w_new = np.clip(out_w_new, -1000, 1000)
    out_b_new = np.clip(out_b_new, -1000, 1000)

    # ----- keep all other weights unchanged ------------------------------------
    new_w = {
        'conv1.weight': w['conv1.weight'],
        'conv1.bias':   w['conv1.bias'],
        'conv2.weight': w['conv2.weight'],
        'conv2.bias':   w['conv2.bias'],
        'conv3.weight': w['conv3.weight'],
        'conv3.bias':   w['conv3.bias'],
        'out.weight':   out_w_new,
        'out.bias':     out_b_new,
    }

    # ----- diagnostics --------------------------------------------------------
    mse_before = float(np.mean(err**2))
    diagnostics = {
        'mse_before': mse_before,
        'lr': float(lr),
        'grad_norm': float(np.linalg.norm(grad)),
    }

    return {"weights": new_w, "diagnostics": diagnostics}
