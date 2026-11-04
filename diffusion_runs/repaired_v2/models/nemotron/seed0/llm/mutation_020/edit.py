import numpy as np

def edit(w, f):
    """
    Edit the neural‑network weights using only permitted training feedback.
    Closed‑form least‑squares fit of the output layer to the target epsilon.
    Returns updated weights and a small diagnostics dictionary.
    """
    # ----- extract permitted feedback arrays -----
    prediction = f['prediction']          # shape (2000, 1, 16, 16)
    y          = f['y']                    # shape (2000, 1, 16, 16)   # target ε
    features   = f['features']             # shape (2000, 64, 16, 16)

    # ----- current output‑layer parameters -----
    out_weight = w['out.weight']          # shape (1, 64, 3, 3)
    out_bias   = w['out.bias']            # shape (1,)

    # ----- error signal -----
    e = prediction - y                     # (2000, 1, 16, 16)

    # ----- flatten spatial dimensions for a single linear regression -----
    B, C, H, W = features.shape               # B=2000, C=64, H=W=16
    F = features.transpose(0, 2, 3, 1).reshape(-1, C)   # (2000*256, 64)
    t = y.reshape(-1)                         # (2000*256,)

    # ----- closed‑form least‑squares: w_opt minimizes ||F w - t||² -----
    w_opt, _, _, _ = np.linalg.lstsq(F, t, rcond=None)
    w_opt = np.asarray(w_opt, dtype=np.float32)       # size‑64 vector

    # ----- build new out.weight (centre the 3×3 kernel) -----
    new_weight = np.zeros((1, 64, 3, 3), dtype=np.float32)
    new_weight[0, :, 1, 1] = w_opt      # place the optimal vector at the centre tap

    # ----- bias correction: mean(target) - mean(predicted without bias) -----
    pred_t = F @ w_opt                    # predicted ε per pixel (no bias)
    new_bias = np.float32(np.mean(t) - np.mean(pred_t))

    # ----- update weights (keep all other layers untouched) -----
    w_new = dict(w)                        # shallow copy
    w_new['out.weight'] = new_weight
    w_new['out.bias']   = new_bias

    # ----- diagnostics -----
    mse_before = float(np.mean(e**2))
    diagnostics = {
        "mse_before": mse_before,
        "weight_norm": float(np.linalg.norm(w_opt)),
        "bias": float(new_bias),
        "updated": True
    }

    return {"weights": w_new, "diagnostics": diagnostics}
