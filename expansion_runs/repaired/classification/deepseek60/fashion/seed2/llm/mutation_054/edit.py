import numpy as np

def edit(w, f):
    # Extract current head parameters
    W = w["head.weight"].copy()  # (10, 1568)
    b = w["head.bias"].copy()    # (10,)
    
    # Feedback features and labels
    X = f["features"]            # (2000, 1568)
    y = f["y"]                   # (2000,)
    
    # Current predictions
    logits = X @ W.T + b
    pred = np.argmax(logits, axis=1)
    errors = np.where(pred != y)[0]
    
    # If no errors, return original weights
    if len(errors) == 0:
        return w
    
    # Try to fix errors with a single rank-1 update per error, but check overall accuracy.
    # We'll accumulate updates and apply them, then evaluate.
    W_new = W.copy()
    b_new = b.copy()
    
    # Use a small margin to ensure correct class wins
    margin = 1.0
    
    for i in errors:
        x = X[i]                 # (1568,)
        true_c = y[i]
        pred_c = pred[i]
        # Current logit difference
        d = logits[i, true_c] - logits[i, pred_c]
        # We want new difference >= margin
        # Adding alpha * x to W[true_c] and subtracting alpha * x from W[pred_c]
        # changes difference by 2 * alpha * (x · x)
        norm2 = np.dot(x, x)
        if norm2 < 1e-12:
            continue
        alpha = (margin - d) / (2.0 * norm2)
        # Apply update
        W_new[true_c] += alpha * x
        W_new[pred_c] -= alpha * x
    
    # Evaluate new head on full feedback pool
    logits_new = X @ W_new.T + b_new
    pred_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(pred_new == y)
    acc_old = np.mean(pred == y)
    
    # Only accept if accuracy improves or stays same with lower CE (we'll just check accuracy)
    if acc_new > acc_old:
        # Check weight bound
        max_abs = np.max(np.abs(W_new))
        if max_abs > 1000.0:
            # Scale down to satisfy bound
            scale = 1000.0 / max_abs
            W_new *= scale
            b_new *= scale
            # Re-evaluate after scaling
            logits_new = X @ W_new.T + b_new
            pred_new = np.argmax(logits_new, axis=1)
            acc_new = np.mean(pred_new == y)
            if acc_new <= acc_old:
                return w  # scaling broke it, revert
        w["head.weight"] = W_new.astype(np.float32)
        w["head.bias"] = b_new.astype(np.float32)
        return w
    else:
        return w
