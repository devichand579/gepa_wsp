import numpy as np

def edit(w, f):
    # Extract current head parameters
    W = w["head.weight"].copy()  # shape (10, 1568)
    b = w["head.bias"].copy()    # shape (10,)
    X = f["features"]            # shape (2000, 1568)
    y = f["y"]                   # shape (2000,)
    
    # Current logits and predictions
    logits = X @ W.T + b
    pred = np.argmax(logits, axis=1)
    mis = np.where(pred != y)[0]
    
    # If there are misclassifications, apply a targeted correction
    if len(mis) > 0:
        # For each misclassified sample, we want to increase the margin for the true class.
        # We solve a least-squares problem to find a correction to W and b that
        # exactly fixes the misclassified samples while minimally affecting others.
        # Let M be the features of misclassified samples (m x 1568).
        M = X[mis]                # (m, 1568)
        m = M.shape[0]
        
        # Desired change in logits for misclassified samples:
        # For each sample i, we want to increase logit_y by delta and decrease logit_p by delta.
        # We'll set delta to be large enough to flip the prediction, e.g., 10.
        delta = 10.0
        D = np.zeros((m, 10))
        for idx, i in enumerate(mis):
            true_y = y[i]
            pred_p = pred[i]
            D[idx, true_y] = delta
            D[idx, pred_p] = -delta
        
        # We seek a correction dW (10, 1568) and db (10,) such that:
        # M @ dW.T + db = D
        # We restrict dW to be a linear combination of the misclassified features:
        # dW = A @ M, where A is (10, m). Then M @ dW.T = M @ M.T @ A.T.
        # So we need (M @ M.T) @ A.T + db = D.
        # We can also absorb db into the correction by appending a column of ones to M.
        # Let M_aug = [M, ones] (m, 1569). Then we want M_aug @ C.T = D, where C is (10, 1569)
        # and C = [A, db]. We can solve for C using least squares.
        M_aug = np.hstack([M, np.ones((m, 1))])  # (m, 1569)
        # Solve C @ M_aug.T = D.T  =>  C = D.T @ pinv(M_aug.T)
        # Actually, we want M_aug @ C.T = D  =>  C.T = pinv(M_aug) @ D  =>  C = D.T @ pinv(M_aug).T
        # Since m is small, we can solve directly.
        # Use least squares: C = D.T @ np.linalg.pinv(M_aug).T
        # But pinv(M_aug) is (1569, m). So C is (10, 1569).
        C = D.T @ np.linalg.pinv(M_aug).T  # (10, 1569)
        dW = C[:, :1568]  # (10, 1568)
        db = C[:, 1568]   # (10,)
        
        # Apply correction
        W = W + dW
        b = b + db
        
        # Recompute logits and check if all misclassifications are fixed
        logits = X @ W.T + b
        pred = np.argmax(logits, axis=1)
        mis = np.where(pred != y)[0]
        
        # If still misclassified, we can increase delta and repeat once (optional, but we avoid loops)
        # For simplicity, we just proceed.
    
    # Now, if training accuracy is 100%, scale up to reduce cross-entropy.
    # Check accuracy
    logits = X @ W.T + b
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    
    if acc == 1.0:
        # Scale the head to make max abs weight = 1000 (or slightly less)
        max_abs = np.max(np.abs(W))
        if max_abs > 0:
            scale = 1000.0 / max_abs
            W = W * scale
            b = b * scale
        # Also ensure max abs bias <= 1000
        max_abs_b = np.max(np.abs(b))
        if max_abs_b > 1000.0:
            scale_b = 1000.0 / max_abs_b
            W = W * scale_b
            b = b * scale_b
    else:
        # If not perfect, we can still try to scale moderately to reduce CE without increasing errors too much.
        # We'll just leave as is or apply a small scale.
        pass
    
    # Ensure all values are finite and within [-1000, 1000]
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)
    
    # Update weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Diagnostics
    logits_new = X @ W.T + b
    pred_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(pred_new == y)
    # Compute cross-entropy
    probs = np.exp(logits_new - np.max(logits_new, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
    
    diagnostics = {
        "accuracy": float(acc_new),
        "cross_entropy": float(ce),
        "num_misclassified_before": int(len(mis)),
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b)))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
