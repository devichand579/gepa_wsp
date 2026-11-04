import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def edit(w, f):
    """
    Update the network weights using only permitted feedback arrays.
    No autograd, back‑propagation or iterative optimisers are used.
    A single closed‑form gradient step is computed from the correlation
    between the epsilon‑prediction error and the final hidden features.

    Parameters
    ----------
    w : dict
        Current network parameters (all weight tensors and biases).
    f : dict
        Permitted feedback arrays:
            - clean, x, t, y, features, prediction, alphabar
            - layer.* traces (only for diagnostics; not needed for the update)

    Returns
    -------
    {"weights": w, "diagnostics": {...}}
    """
    # ------------------------------------------------------------------
    # 1. Pull the relevant feedback arrays (shapes are fixed by the task)
    # ------------------------------------------------------------------
    # training batch size N = 2000 (full‑set) or 128 (paired batch);
    # we use the full‑set arrays that are always present.
    N = f['prediction'].shape[0]                # e.g. 2000
    e = f['prediction'] - f['y']               # [N,1,16,16] error image

    # final hidden feature maps for all training examples
    features = f['features']                    # [N,64,16,16]

    # ------------------------------------------------------------------
    # 2. Compute a per‑pixel error map (remove the channel dimension)
    # ------------------------------------------------------------------
    e_s = e[:, 0]                              # [N,16,16]

    # ------------------------------------------------------------------
    # 3. Extract all 3×3 spatial patches from the 64‑channel feature maps.
    #    sliding_window_view returns shape (N, 64, 16, 16, 3, 3).
    # ------------------------------------------------------------------
    patches = sliding_window_view(features, (3, 3), axis=(2, 3))
    # sanity: ensure correct axis ordering (N, C, H, W, kH, kW)
    # numpy>=1.20 guarantees the output is (N, C, H, W, kH, kW)

    # ------------------------------------------------------------------
    # 4. Build the gradient of the MSE w.r.t. out.weight.
    #    For each output pixel the weight gradient is the sum over examples
    #    of  error * corresponding input patch.
    # ------------------------------------------------------------------
    # broadcast error over channels and spatial dims:
    e_exp = e_s[:, None, None, :, :]          # [N,1,1,16,16]
    # element‑wise product (broadcasting over the 64 channels and 3×3)
    prod = patches * e_exp                     # [N,64,16,16,3,3]

    # sum over the batch (axis 0) and over the output spatial locations
    # (axes 3 and 4) → gradient per (channel, kH, kW) = [64,3,3]
    grad_weight_raw = prod.sum(axis=(0, 3, 4))   # [64,3,3]

    # average over the N examples → true gradient of the mean MSE
    grad_weight = grad_weight_raw / N           # [64,3,3]

    # reshape to the network's expected shape [1,64,3,3]
    grad_weight = grad_weight[np.newaxis, ...]   # [1,64,3,3]

    # ------------------------------------------------------------------
    # 5. Gradient for the output bias (scalar)
    # ------------------------------------------------------------------
    grad_bias = e_s.mean()                       # scalar

    # ------------------------------------------------------------------
    # 6. Closed‑form step size (inverse of mean activation energy)
    # ------------------------------------------------------------------
    act_var = (features ** 2).mean()             # scalar, E[‖a‖²]
    lr = 1.0 / (act_var + 1e-8)                  # small positive float

    # ------------------------------------------------------------------
    # 7. Apply the update (preserve shapes, clip to finite float32)
    # ------------------------------------------------------------------
    w['out.weight'] = w['out.weight'] - lr * grad_weight
    w['out.bias']   = w['out.bias']   - lr * grad_bias

    # Clip to the allowed range [‑1000, 1000] and guarantee finite32
    for key in w:
        w[key] = np.clip(w[key], -1000, 1000).astype(np.float32)

    # ------------------------------------------------------------------
    # 8. Diagnostics (tiny JSON‑compatible dict)
    # ------------------------------------------------------------------
    diagnostics = {
        "mse_before": float((e ** 2).mean()),
        "grad_bias": float(grad_bias),
        "grad_weight_norm": float(np.linalg.norm(grad_weight)),
        "lr": float(lr),
        "activation_var": float(act_var),
    }

    return {"weights": w, "diagnostics": diagnostics}
