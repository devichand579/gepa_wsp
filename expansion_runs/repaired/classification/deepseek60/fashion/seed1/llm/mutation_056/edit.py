import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    
    # Current head weights
    W = w["head.weight"].astype(np.float64)  # (10, 1568)
    b = w["head.bias"].astype(np.float64)    # (10,)
    
    # Compute current logits
    L = X @ W.T + b  # (N, 10)
    pred = np.argmax(L, axis=1)
    mis = np.where(pred != y)[0]
    
    # If there are misclassifications, apply targeted correction
    if len(mis) > 0:
        Xb_mis = np.hstack([X[mis], np.ones((len(mis), 1))])  # (M, 1569)
        L_mis = L[mis]  # (M, 10)
        # Target: correct class = 10, others = -10 (strong margin)
        T_mis = -10.0 * np.ones((len(mis), 10))
        T_mis[np.arange(len(mis)), y[mis]] = 10.0
        dL = T_mis - L_mis  # (M, 10)
        # Solve for dW (10, 1569) such that Xb_mis @ dW.T = dL.T
        pinv_Xb = np.linalg.pinv(Xb_mis)  # (1569, M)
        dW = dL @ pinv_Xb.T  # (10, 1569)
        W += dW[:, :D]
        b += dW[:, D]
        
        # Recompute logits and check accuracy
        L_new = X @ W.T + b
        pred_new = np.argmax(L_new, axis=1)
        acc_new = np.mean(pred_new == y)
        
        # If still not perfect, fall back to full least-squares
        if acc_new < 1.0:
            Xb = np.hstack([X, np.ones((N, 1))])
            T = -1.0 * np.ones((N, 10))
            T[np.arange(N), y] = 1.0
            lam = 1e-3
            A = Xb.T @ Xb + lam * np.eye(D + 1)
            B = Xb.T @ T
            Wb = np.linalg.solve(A, B)  # (1569, 10)
            W = Wb[:D].T
            b = Wb[D]
    else:
        # No misclassifications, just scale current head to max 1000
        pass
    
    # Scale to max absolute value 1000
    max_val = max(np.max(np.abs(W)), np.max(np.abs(b)))
    if max_val > 0:
        alpha = 1000.0 / max_val
        W *= alpha
        b *= alpha
    else:
        alpha = 1.0
    
    # Clip to be safe
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)
    
    # Update weights
    w["head.weight"] = W
    w["head.bias"] = b
    
    # Diagnostics
    L_final = X @ W.T + b
    pred_final = np.argmax(L_final, axis=1)
    acc_final = np.mean(pred_final == y)
    # Cross-entropy
    L_final_shifted = L_final - np.max(L_final, axis=1, keepdims=True)
    exp_L = np.exp(L_final_shifted)
    probs = exp_L / np.sum(exp_L, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
    
    diagnostics = {
        "accuracy": float(acc_final),
        "cross_entropy": float(ce),
        "num_misclassified_before": int(len(mis)),
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b))),
        "scale_alpha": float(alpha)
    }
    
    return {"weights": w, "diagnostics": diagnostics}
