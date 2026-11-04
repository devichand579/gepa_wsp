import numpy as np

def edit(w, f):
    """
    Closed‑form update of the output read‑out layer using the provided
    training feedback arrays. Only `out.weight` and `out.bias` are changed;
    all other weight tensors are left untouched.

    Parameters
    ----------
    w : dict
        Current network parameters (full‑precision). Keys match the
        `network` section of the problem statement.
    f : dict
        Feedback arrays: `x`, `t`, `y`, `features`, `prediction`,
        `alphabar`, `batch.*`, etc. (only the ones we need are used).

    Returns
    -------
    w : dict
        Updated weights (same shape/key structure). Either the original
        `w` dict with modified `out.weight`/`out.bias`, or a wrapper
        `{"weights": w, "diagnostics": {...}}`.
    """
    # ------------------------------------------------------------------
    # 1. Pull the relevant feedback arrays (all are NumPy float32)
    # ------------------------------------------------------------------
    # f["features"]   : [N=2000, C=64, H=16, W=16]   final hidden maps
    # f["y"]          : [N=2000, 1, H=16, W=16]     target Gaussian ε
    # f["prediction"] : [N=2000, 1, H=16, W=16]     current ε prediction
    # f["batch.ids"]  : [128] – not needed for the global fit
    features = f["features"].astype(np.float32)          # (2000,64,16,16)
    y_target   = f["y"].astype(np.float32)             # (2000,1,16,16)

    # ------------------------------------------------------------------
    # 2. Build a design matrix X and response vector y from *all* spatial
    #    positions of all examples.  Each spatial location becomes one
    #    regression sample.
    # ------------------------------------------------------------------
    N, C, H, W = features.shape               # 2000, 64, 16, 16
    n_spatial = H * W                         # 256
    n_samples = N * n_spatial                 # 512 000

    # Flatten features: (N, C, H, W) -> (N, C*H*W) then reshape to (n_samples, C)
    X = features.transpose(0, 2, 3, 1).reshape(n_samples, C)   # (512k, 64)

    # Flatten target ε: (N, 1, H, W) -> (n_samples,)
    y = y_target.reshape(n_samples)                # (512k,)

    # ------------------------------------------------------------------
    # 3. Closed‑form ordinary least‑squares (normal equation)
    #    w = (Xᵀ X)^{-1} Xᵀ y      (optionally with a bias term)
    # ------------------------------------------------------------------
    # Center X and y for numerical stability (still a valid LS solution
    # because we later undo the centering when constructing the bias).
    X_mean = X.mean(axis=0, keepdims=True)          # (1,64)
    y_mean = y.mean()                               # scalar

    Xc = X - X_mean
    yc = y - y_mean

    # Normal equations:  (Xᵀ X) w = Xᵀ y
    XtX = Xc.T @ Xc                                 # (64,64)
    Xty = Xc.T @ yc                                 # (64,)

    # Use lstsq for robustness (still a closed‑form solution)
    w_centered, residuals, rank, s = np.linalg.lstsq(Xc, yc, rcond=None)
    # w_centered shape (64,)

    # Recover the bias term so that  y_pred = X @ w + b  fits the original
    # (un‑centered) data.
    b = y_mean - (X_mean @ w_centered).item()       # scalar

    # ------------------------------------------------------------------
    # 4. Map the learned 1‑D read‑out weights onto the convolutional
    #    output tensor `out.weight` (shape [1,64,3,3]) and set `out.bias`.
    #    We broadcast the same weight vector across every kernel position
    #    (effective 1×1 convolution) – this preserves the required shape
    #    and keeps all values finite float32.
    # ------------------------------------------------------------------
    # Initialize new weight tensor with zeros then fill.
    new_weight = np.zeros_like(w["out.weight"], dtype=np.float32)
    # w_centered is (64,); reshape to (1,64,1,1) and broadcast to (1,64,3,3)
    new_weight[:, :, :, :] = w_centered.reshape(1, C, 1, 1)

    new_bias = np.float32(b)

    # ------------------------------------------------------------------
    # 5. Compute diagnostics: training MSE before and after the edit.
    #    We can evaluate the new read‑out on the same training samples
    #    without running the full forward pass – just the linear mapping.
    # ------------------------------------------------------------------
    y_pred_train = (X @ w_centered + b).reshape(N, 1, H, W).astype(np.float32)

    mse_before = np.mean((f["prediction"].astype(np.float32) - y_target.astype(np.float32))**2)
    mse_after  = np.mean((y_pred_train - y_target.astype(np.float32))**2)

    diagnostics = {
        "mse_before": float(mse_before),
        "mse_after":  float(mse_after),
        "weight_norm": float(np.linalg.norm(w_centered)),
        "bias": float(b),
        "n_samples_used": int(n_samples),
        "rank": int(rank),
    }

    # ------------------------------------------------------------------
    # 6. Assemble the updated weight dictionary.
    # ------------------------------------------------------------------
    # Deep‑copy only the parts we change; keep everything else identical.
    w_new = {k: v for k, v in w.items()}   # shallow copy of outer dict
    w_new["out.weight"] = new_weight
    w_new["out.bias"]   = new_bias

    # Return either the dict or a wrapper with diagnostics.
    return {"weights": w_new, "diagnostics": diagnostics}
