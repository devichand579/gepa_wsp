import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    n, d = X.shape
    Y = np.zeros((n, 10), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    Xb = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)
    lam = 1e-4
    A = Xb.T @ Xb
    reg = np.eye(d + 1) * lam
    reg[-1, -1] = 0.0
    A += reg
    B = Xb.T @ Y
    Wb = np.linalg.solve(A, B)
    W = Wb[:-1, :].T
    b = Wb[-1, :]
    max_abs = np.max(np.abs(W))
    if max_abs > 0:
        s = 1000.0 / max_abs
        W *= s
        b *= s
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    diagnostics = {
        "lambda": lam,
        "scale": float(s) if max_abs > 0 else 1.0,
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b))),
        "head_weight_mean": float(np.mean(W)),
        "head_weight_std": float(np.std(W)),
        "head_bias_mean": float(np.mean(b)),
    }
    return {"weights": w, "diagnostics": diagnostics}
