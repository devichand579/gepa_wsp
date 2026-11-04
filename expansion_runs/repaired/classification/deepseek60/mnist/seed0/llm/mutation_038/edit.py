import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    W0 = w["fc2.weight"].astype(np.float64)       # [10, 128]
    b0 = w["fc2.bias"].astype(np.float64)         # [10]

    n, d = X.shape
    K = W0.shape[0]

    # One-hot targets
    Y = np.zeros((n, K), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Augment features with bias column
    Xa = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)  # [n, d+1]
    W0a = np.concatenate([W0, b0[:, None]], axis=1)                      # [K, d+1]

    # Ridge fit with prior toward current fc2.
    # lambda is chosen relative to the feature scale.
    XtX = Xa.T @ Xa
    scale = np.trace(XtX) / (d + 1)
    lam = 1e-2 * scale + 1.0

    A = XtX + lam * np.eye(d + 1, dtype=np.float64)
    B = Xa.T @ Y + lam * W0a.T
    W_aug = np.linalg.solve(A, B).T   # [K, d+1]

    W_new = W_aug[:, :d]
    b_new = W_aug[:, d]

    # Keep logit scale close to the original to avoid destroying calibration.
    Z_old = X @ W0.T + b0
    Z_new = X @ W_new.T + b_new
    s_old = np.std(Z_old)
    s_new = np.std(Z_new)
    if s_new > 1e-8:
        scale_factor = s_old / s_new
        # Limit the scale factor to avoid extreme confidence.
        scale_factor = float(np.clip(scale_factor, 0.25, 4.0))
        W_new *= scale_factor
        b_new *= scale_factor

    w["fc2.weight"] = W_new.astype(np.float32)
    w["fc2.bias"] = b_new.astype(np.float32)

    # Diagnostics
    Z_final = X @ W_new.T + b_new
    pred = np.argmax(Z_final, axis=1)
    acc = float(np.mean(pred == y))
    ce = float(np.mean(-np.log(np.clip(
        np.exp(Z_final - Z_final.max(axis=1, keepdims=True)) /
        np.sum(np.exp(Z_final - Z_final.max(axis=1, keepdims=True)), axis=1, keepdims=True),
        1e-12, 1.0
    )[np.arange(n), y])))
    diag = {
        "lambda": float(lam),
        "scale_factor": float(scale_factor) if s_new > 1e-8 else 1.0,
        "accuracy": acc,
        "cross_entropy": ce,
        "fc2_weight_norm": float(np.linalg.norm(W_new)),
        "fc2_bias_norm": float(np.linalg.norm(b_new)),
    }
    return {"weights": w, "diagnostics": diag}
