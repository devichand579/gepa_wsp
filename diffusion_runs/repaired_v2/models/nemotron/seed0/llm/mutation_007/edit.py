import numpy as np

def edit(w, f):
    """
    Closed-form improvement of the DDPM epsilon‑prediction network weights.
    Permitted operations: NumPy statistics, covariance, closed‑form least‑squares/ridge normal equations.
    No autograd, backprop, gradient estimation, or iterative optimizers.
    
    Parameters
    ----------
    w : dict
        Full current parameter arrays (keys/ shapes as in the network definition).
    f : dict
        Permitted feedback arrays: 'features', 'prediction', 'y', 'alphabar',
        'batch_ids', 'batch_x', 'batch_t', 'batch_y', 'generated_samples',
        and layer traces (only 'features', 'prediction', 'y' are needed here).
    
    Returns
    -------
    w : dict
        Updated weights with 'out.weight' and 'out.bias' modified.
    {"weights": w, "diagnostics": {...}}   (handled by caller; we return w and a diagnostics dict)
    """
    # --- pull core feedback arrays ---
    features = f['features']                # [N, 64, 16, 16]  final hidden feature maps
    pred   = f['prediction']                # [N, 1, 16, 16]  current ε‑prediction
    y      = f['y']                         # [N, 1, 16, 16]  target Gaussian ε
    N      = features.shape[0]

    # --- residual between target and current prediction ---
    r = y - pred                           # [N, 1, 16, 16]

    # flatten residual per pixel → design target vector
    y_vec = r.reshape(-1).astype(np.float32)   # [N*256]

    # --- build design matrix X via im2col (64 channels × 3×3 = 576 weights) ---
    # 256 spatial positions, each gives one row of 576 patch values
    X_rows = []
    for i in range(16):
        for j in range(16):
            patch = features[:, :, i:i+3, j:i+3]   # [N, 64, 3, 3]
            X_rows.append(patch.reshape(N, -1))    # [N, 576]
    X = np.concatenate(X_rows, axis=0).astype(np.float32)   # [N*256, 576]

    # --- ridge normal‑equation fit (closed‑form, no iterative optimizer) ---
    λ = 1e-6
    XtX = X.T @ X + λ * np.eye(576, dtype=np.float32)
    Xty = X.T @ y_vec
    w_opt = np.linalg.solve(XtX, Xty).astype(np.float32)    # [576]

    # reshape to output convolution weight [1, 64, 3, 3]
    new_weight = w_opt.reshape(1, 64, 3, 3)

    # --- update output weight (preserve all other keys / shapes) ---
    w = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in w.items()}
    w['out.weight'] = new_weight

    # --- update output bias via mean residual (closed‑form moment shift) ---
    mean_r = float(np.mean(r).astype(np.float32))
    w['out.bias'] = (w['out.bias'].astype(np.float32) + np.float32(mean_r)).astype(np.float32)

    # --- preserve constraints: finite float32, |val| ≤ 1000 ---
    for k in w:
        if isinstance(w[k], np.ndarray):
            np.clip(w[k], -1000, 1000, out=w[k])

    # --- diagnostics (small JSON‑compatible statistics) ---
    diagnostics = {
        "bias_shift": mean_r,
        "ridge_lambda": float(λ),
        "weight_update_l2": float(np.linalg.norm(w_opt)),
        "residual_mean": mean_r,
        "X_rows": 256,
        "weights_fit": 576,
        "status": "closed_form_ridge_bias_shift"
    }

    return {"weights": w, "diagnostics": diagnostics}
