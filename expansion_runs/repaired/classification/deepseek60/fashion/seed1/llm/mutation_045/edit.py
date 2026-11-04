def edit(w, f):
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    
    # Create targets: +1 for correct class, -1 for others
    T = -np.ones((N, 10), dtype=np.float64)
    T[np.arange(N), y] = 1.0
    
    # Augment with bias column
    X_aug = np.hstack([X, np.ones((N, 1))])
    
    # Solve least squares: X_aug @ W_aug.T = T
    W_aug_T, residuals, rank, s = np.linalg.lstsq(X_aug, T, rcond=None)
    W_aug = W_aug_T.T  # shape (10, D+1)
    
    W_new = W_aug[:, :D]
    b_new = W_aug[:, D]
    
    # Scale to max absolute value 1000
    max_abs = max(np.max(np.abs(W_new)), np.max(np.abs(b_new)))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        W_new *= scale
        b_new *= scale
    
    # Clip to ensure within bounds
    W_new = np.clip(W_new, -1000, 1000)
    b_new = np.clip(b_new, -1000, 1000)
    
    # Update weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)
    
    # Diagnostics on full pool
    logits = X @ W_new.T + b_new
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
    
    # Diagnostics on paired batch
    batch_ids = f["batch.ids"]
    batch_y = f["batch.y"]
    X_batch = X[batch_ids]
    logits_batch = X_batch @ W_new.T + b_new
    logits_max_b = np.max(logits_batch, axis=1, keepdims=True)
    exp_logits_b = np.exp(logits_batch - logits_max_b)
    probs_batch = exp_logits_b / np.sum(exp_logits_b, axis=1, keepdims=True)
    preds_batch = np.argmax(logits_batch, axis=1)
    acc_batch = np.mean(preds_batch == batch_y)
    ce_batch = -np.mean(np.log(probs_batch[np.arange(len(batch_y)), batch_y] + 1e-12))
    
    diagnostics = {
        "full_pool_accuracy": float(acc),
        "full_pool_cross_entropy": float(ce),
        "paired_batch_accuracy": float(acc_batch),
        "paired_batch_cross_entropy": float(ce_batch),
        "max_abs_weight": float(np.max(np.abs(W_new))),
        "max_abs_bias": float(np.max(np.abs(b_new))),
        "num_errors_full": int(np.sum(preds != y)),
        "num_errors_batch": int(np.sum(preds_batch != batch_y))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
