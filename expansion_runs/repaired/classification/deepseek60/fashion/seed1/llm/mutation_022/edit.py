import numpy as np

def edit(w, f):
    # Extract cached features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    batch_ids = f["batch.ids"].astype(np.int64)   # (128,)
    
    # Add bias column
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    X_batch_aug = X_aug[batch_ids]
    y_batch = y[batch_ids]
    
    # One-hot encoding
    n_classes = 10
    Y = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    Y[np.arange(y.shape[0]), y] = 1.0
    
    # Ridge regression with a small grid of lambdas
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    best_lambda = None
    best_W = None
    best_acc = -1.0
    best_ce = np.inf
    
    for lam in lambdas:
        A = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1])
        B = X_aug.T @ Y
        W = np.linalg.solve(A, B).T  # (10, 1569)
        
        # Evaluate on paired batch
        logits_batch = X_batch_aug @ W.T
        pred = np.argmax(logits_batch, axis=1)
        acc = np.mean(pred == y_batch)
        
        # Cross-entropy
        logits_shifted = logits_batch - np.max(logits_batch, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-12))
        
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc = acc
            best_ce = ce
            best_lambda = lam
            best_W = W
    
    # Optimize logit scale on the paired batch (accuracy is invariant to positive scaling)
    W = best_W
    logits_batch = X_batch_aug @ W.T
    
    scales = np.linspace(0.1, 100.0, 2000)
    best_scale = 1.0
    best_scale_ce = np.inf
    for s in scales:
        scaled_logits = logits_batch * s
        logits_shifted = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-12))
        if ce < best_scale_ce:
            best_scale_ce = ce
            best_scale = s
    
    # Apply scale to the linear head
    W_scaled = W * best_scale
    
    # Update only the head weights
    w["head.weight"] = W_scaled[:, :1568].astype(np.float32)
    w["head.bias"] = W_scaled[:, 1568].astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "best_scale": float(best_scale),
        "batch_accuracy": float(best_acc),
        "batch_ce": float(best_scale_ce),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias_mean": float(np.mean(w["head.bias"])),
        "head_weight_min": float(np.min(w["head.weight"])),
        "head_weight_max": float(np.max(w["head.weight"])),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
