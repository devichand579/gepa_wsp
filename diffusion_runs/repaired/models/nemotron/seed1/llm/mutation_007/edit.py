import numpy as np

def edit(w, f):
    """
    Update the network's output‑layer weights (and bias) with a single
    gradient‑descent step that minimises the per‑example MSE between the
    current epsilon prediction and the target Gaussian epsilon.

    Parameters
    ----------
    w : dict
        Current full‑precision parameters.  Expected keys:
        "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
        "conv3.weight", "conv3.bias", "out.weight", "out.bias".
    f : dict
        Permitted feedback arrays.  Must contain at least:
        - "prediction": current epsilon prediction [N,1,16,16]
        - "y": target Gaussian epsilon [N,1,16,16]
        - "layer.out.input0": final‑hidden feature maps [N,64,16,16]

    Returns
    -------
    {"weights": w_updated, "diagnostics": {...}}
    """
    # ------------------------------------------------------------------
    # 1.  Gather feedback and build the error signal
    # ------------------------------------------------------------------
    N = 2000                              # batch size (inferred from shapes)
    pred = f["prediction"]                # [N,1,16,16]
    target = f["y"]                       # [N,1,16,16]
    e = pred - target                     # raw error [N,1,16,16]
    e_flat = e.reshape(-1)                # [N*256]

    # final‑layer input (before the output conv) – shape [N,64,16,16]
    inp0 = f["layer.out.input0"]

    # ------------------------------------------------------------------
    # 2.  Pad the input so that the 3×3 convolution with stride 1 and
    #     padding=1 keeps the 16×16 spatial size.
    # ------------------------------------------------------------------
    padded = np.pad(inp0, ((0, 0), (0, 0), (1, 1), (1, 1)),
                    mode="constant")          # [N,64,18,18]

    # ------------------------------------------------------------------
    # 3.  Extract all 3×3 patches from the padded input.
    #     sliding_window_view gives a view of shape (N,64,16,16,3,3).
    # ------------------------------------------------------------------
    from numpy.lib.stride_tricks import sliding_window_view
    patches = sliding_window_view(padded, (3, 3), axis=(-2, -1))
    # reorder to (N,16,16,64,3,3) then flatten each patch to a 576‑dim vector
    patches = np.transpose(patches, (0, 2, 3, 1, 4, 5))   # [N,16,16,64,3,3]
    patches_flat = patches.reshape(-1, 64 * 3 * 3)       # [N*256, 576]

    # ------------------------------------------------------------------
    # 4.  Closed‑form gradient of the MSE w.r.t. the output weight tensor.
    #     MSE = (1/(N*256)) * sum e^2 ;  dMSE/dw = (1/(N*256)) * e · patches
    # ------------------------------------------------------------------
    grad_weight_raw = np.dot(e_flat, patches_flat)       # [576]
    grad_weight = (grad_weight_raw / (N * 256)).reshape(
        w["out.weight"].shape)                           # [1,64,3,3]

    # gradient for the output bias (scalar)
    grad_bias = np.mean(e_flat)                           # scalar

    # ------------------------------------------------------------------
    # 5.  Apply a single gradient‑descent step (lr = 0.01).
    # ------------------------------------------------------------------
    lr = 0.01
    w_new = {k: v.copy() for k, v in w.items()}
    w_new["out.weight"] = w["out.weight"] - lr * grad_weight
    w_new["out.bias"] = w["out.bias"] - lr * grad_bias

    # ------------------------------------------------------------------
    # 6.  Enforce finite float32 and |value| ≤ 1000.
    # ------------------------------------------------------------------
    for k in w_new:
        arr = w_new[k]
        # replace inf/nan with 0
        arr = np.where(np.isinf(arr) | np.isnan(arr), 0.0, arr)
        # clip absolute value
        arr = np.clip(arr, -1000, 1000)
        # cast back to float32
        w_new[k] = arr.astype(np.float32)

    # ------------------------------------------------------------------
    # 7.  Diagnostics: before‑MSE, gradient norm, learning rate, updated keys.
    # ------------------------------------------------------------------
    before_mse = float(np.mean(e ** 2))
    grad_norm = float(np.sqrt(np.sum(grad_weight ** 2) + grad_bias ** 2))
    diagnostics = {
        "before_mse": before_mse,
        "grad_norm": grad_norm,
        "lr": float(lr),
        "updated_layers": ["out.weight", "out.bias"]
    }

    return {"weights": w_new, "diagnostics": diagnostics}
