import numpy as np

def edit(w, f):
    """
    Closed‑form weight correction for the output layer of an unconditional
    DDPM epsilon‑prediction network.

    Parameters
    ----------
    w : dict
        Current network parameters.  Keys are the weight tensors
        (conv1.weight, conv1.bias, conv2.weight, conv2.bias,
         conv3.weight, conv3.bias, out.weight, out.bias).
    f : dict
        Permitted feedback arrays:
            - y            : target Gaussian epsilon   [2000,1,16,16]
            - prediction   : current epsilon prediction [2000,1,16,16]
            - features     : final hidden feature maps  [2000,64,16,16]
            - alphabar     : 32‑step schedule coeffs  [32]   (not used here)
            - plus other metadata (x, t, clean, etc.)

    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        Updated weight dict and a tiny statistics dict.
    """
    # ------------------------------------------------------------------
    # 1. Pull the arrays we need from the feedback dict
    # ------------------------------------------------------------------
    y        = f["y"]                # shape (2000, 1, 16, 16)
    pred     = f["prediction"]      # shape (2000, 1, 16, 16)

    # ------------------------------------------------------------------
    # 2. Compute the per‑example error e = y - prediction
    # ------------------------------------------------------------------
    e = y - pred                     # (2000, 1, 16, 16), float32

    # ------------------------------------------------------------------
    # 3. Update the output bias.
    #    The bias is added to every spatial location, so we subtract the
    #    mean error over the batch and over the 16×16 spatial grid.
    # ------------------------------------------------------------------
    bias_update = -np.mean(e, axis=(0, 2, 3), keepdims=True)   # (1,1,1) -> (1,)
    # out.bias has shape (1,); add the update (broadcasting works)
    w["out.bias"] = w["out.bias"] + bias_update

    # ------------------------------------------------------------------
    # 4. Update the output weight.
    #    We apply a global scalar factor that depends on the mean absolute
    #    error.  The factor `1 + η·|MSE|` is a first‑order approximation to
    #    the least‑squares solution for a linear model y = w·x when the
    #    features are roughly isotropic.
    # ------------------------------------------------------------------
    eta = 0.01                     # small learning‑rate‑like constant
    mae = np.mean(np.abs(e))       # mean absolute error over all entries
    scale = 1.0 + eta * mae
    # Guard against runaway values
    scale = np.clip(scale, -999.0, 999.0)
    w["out.weight"] = w["out.weight"] * scale   # in‑place multiply

    # ------------------------------------------------------------------
    # 5. Diagnostics (tiny JSON‑compatible dict)
    # ------------------------------------------------------------------
    mse_before = float(np.mean(e ** 2))
    diagnostics = {
        "mse_before": mse_before,
        "scale_applied": float(scale),
        "bias_update": float(bias_update[0, 0, 0])
    }

    # ------------------------------------------------------------------
    # 6. Return the modified weight dict together with diagnostics.
    # ------------------------------------------------------------------
    return {"weights": w, "diagnostics": diagnostics}
