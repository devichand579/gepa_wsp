import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)  # (N, 1568)
    y = f["y"].astype(np.int64)           # (N,)
    N, D = X.shape
    C = 10
    
    # Target encoding: +1 for correct class, -1 for others
    Y = -np.ones((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Augment with bias term
    X_aug = np.hstack([X, np.ones((N, 1))])  # (N, D+1)
    
    # Compute current errors to set sample weights
    W_cur = w["head.weight"].astype(np.float64)  # (10, 1568)
    b_cur = w["head.bias"].astype(np.float64)    # (10,)
    logits_cur = X @ W_cur.T + b_cur
    pred_cur = np.argmax(logits_cur, axis=1)
    errors = (pred_cur != y)
    
    # Sample weights: upweight errors
    sample_weights = np.ones(N, dtype=np.float64)
    sample_weights[errors] = 100.0  # high weight to force correct classification
    
    # Weighted least squares
    W_sqrt = np.sqrt(sample_weights)
    Xw = X_aug * W_sqrt[:, None]
    Yw = Y * W_sqrt[:, None]
    
    # Ridge regularization
    λ = 1e-6
    Xw_aug = np.vstack([Xw, np.sqrt(λ) * np.eye(D+1)])
    Yw_aug = np.vstack([Yw, np.zeros((D+1, C))])
    
    # Solve least squares
    coef, _, _, _ = np.linalg.lstsq(Xw_aug, Yw_aug, rcond=None)
    
    W_new = coef[:D, :].T  # (C, D)
    b_new = coef[D, :]     # (C,)
    
    # Check accuracy on full pool
    logits_new = X @ W_new.T + b_new
    pred_new = np.argmax(logits_new, axis=1)
    acc_full = np.mean(pred_new == y)
    
    # Scale to max abs weight <= 1000
    max_abs = np.max(np.abs(W_new))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W_new *= scale
        b_new *= scale
    
    # Clip to ensure bounds
    W_new = np.clip(W_new, -1000.0, 1000.0)
    b_new = np.clip(b_new, -1000.0, 1000.0)
    
    # Update weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)
    
    # Diagnostics
    # Paired batch
    batch_ids = f["batch.ids"]  # (128,)
    X_batch = X[batch_ids]
    y_batch = y[batch_ids]
    logits_batch = X_batch @ W_new.T + b_new
    
    def softmax_ce(logits, labels):
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12))
        return ce
    
    ce_batch = softmax_ce(logits_batch, y_batch)
    pred_batch = np.argmax(logits_batch, axis=1)
    acc_batch = np.mean(pred_batch == y_batch)
    
    ce_full = softmax_ce(logits_new, y)
    
    diagnostics = {
        "full_pool_accuracy": float(acc_full),
        "full_pool_cross_entropy": float(ce_full),
        "paired_batch_accuracy": float(acc_batch),
        "paired_batch_cross_entropy": float(ce_batch),
        "max_abs_weight": float(np.max(np.abs(W_new))),
        "max_abs_bias": float(np.max(np.abs(b_new))),
        "num_errors_before": int(np.sum(errors)),
        "num_errors_after": int(np.sum(pred_new != y)),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
