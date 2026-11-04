import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    
    # Current final classifier parameters
    W = w["fc2.weight"].astype(np.float64)  # (10, 128)
    b = w["fc2.bias"].astype(np.float64)    # (10,)
    
    # Current logits and predictions
    Z = X @ W.T + b                         # (2000, 10)
    pred = np.argmax(Z, axis=1)             # (2000,)
    errors = np.where(pred != y)[0]
    
    if len(errors) == 0:
        return w
    
    # Target margin for the true class over the predicted class
    target_margin = 0.5
    margins = Z[errors, y[errors]] - Z[errors, pred[errors]]
    delta = target_margin - margins  # positive for all errors
    
    m = len(errors)
    n = 10 * 128 + 10  # 1290 parameters (weights + biases)
    A = np.zeros((m, n), dtype=np.float64)
    
    # Build linear system: for each error, increase (W_t - W_p) @ x + (b_t - b_p) by delta
    for i, idx in enumerate(errors):
        t = y[idx]
        p = pred[idx]
        A[i, t*128:(t+1)*128] = X[idx]
        A[i, p*128:(p+1)*128] = -X[idx]
        A[i, 1280 + t] = 1.0
        A[i, 1280 + p] = -1.0
    
    # Minimum-norm solution: d = A.T @ (A A.T + lambda I)^{-1} @ delta
    lambda_reg = 1e-3
    AAT = A @ A.T
    AAT_reg = AAT + lambda_reg * np.eye(m)
    sol = np.linalg.solve(AAT_reg, delta)
    d = A.T @ sol
    
    dW = d[:1280].reshape(10, 128)
    db = d[1280:]
    
    # Apply the adjustment
    W_new = W + dW
    b_new = b + db
    
    w["fc2.weight"] = W_new.astype(np.float32)
    w["fc2.bias"] = b_new.astype(np.float32)
    
    # Diagnostics: evaluate on the full feedback pool
    Z_new = X @ W_new.T + b_new
    pred_new = np.argmax(Z_new, axis=1)
    acc = np.mean(pred_new == y)
    Z_new_shifted = Z_new - np.max(Z_new, axis=1, keepdims=True)
    expZ = np.exp(Z_new_shifted)
    probs = expZ / np.sum(expZ, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
    
    diagnostics = {
        "num_errors": int(m),
        "target_margin": target_margin,
        "dW_norm": float(np.linalg.norm(dW)),
        "db_norm": float(np.linalg.norm(db)),
        "new_acc": float(acc),
        "new_ce": float(ce),
    }
    return {"weights": w, "diagnostics": diagnostics}
