import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # [4000, 32]
    y = f["y"].astype(np.float64)                 # [4000]
    
    # Centre features for numerical stability
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    
    # Ridge regression: solve (Xc^T Xc + lambda I) w = Xc^T (y - mean(y))
    lam = 1e-3
    n, d = Xc.shape
    A = Xc.T @ Xc + lam * np.eye(d)
    b = Xc.T @ (y - y.mean())
    w_head = np.linalg.solve(A, b)
    bias = y.mean() - mu.ravel() @ w_head
    
    # Set head weights: logit1 - logit0 = 2 * (w_head @ pooled + bias)
    # So head.weight[1] = w_head, head.weight[0] = -w_head, head.bias[1] = bias, head.bias[0] = -bias
    w["head.weight"][1] = w_head.astype(np.float32)
    w["head.weight"][0] = (-w_head).astype(np.float32)
    w["head.bias"][1] = np.float32(bias)
    w["head.bias"][0] = np.float32(-bias)
    
    # Ensure all weights are finite and within bounds
    for k in w:
        v = w[k]
        if not np.isfinite(v).all():
            v[:] = 0.0
        np.clip(v, -1000.0, 1000.0, out=v)
    
    diagnostics = {
        "head_norm": float(np.linalg.norm(w_head)),
        "bias": float(bias),
        "lambda": lam,
        "train_acc": float(np.mean((X @ w_head + bias > 0) == (y > 0.5)))
    }
    return {"weights": w, "diagnostics": diagnostics}
