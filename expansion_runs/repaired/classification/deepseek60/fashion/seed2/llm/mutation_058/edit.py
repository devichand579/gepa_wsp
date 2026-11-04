import numpy as np

def edit(w, f):
    # Work on copies to avoid mutating the input in place (though returning w is fine)
    W = w["head.weight"].copy()  # (10, 1568)
    b = w["head.bias"].copy()    # (10,)
    
    X = f["features"]            # (2000, 1568)
    y = f["y"]                   # (2000,)
    
    # Current logits
    logits = X @ W.T + b         # (2000, 10)
    preds = np.argmax(logits, axis=1)
    
    # Correct and max-wrong logits
    correct_logits = logits[np.arange(len(y)), y]
    logits_wrong = logits.copy()
    logits_wrong[np.arange(len(y)), y] = -np.inf
    max_wrong_logits = np.max(logits_wrong, axis=1)
    margins = correct_logits - max_wrong_logits
    
    # Select examples to update: all misclassified + the 200 smallest margins
    mis = np.where(preds != y)[0]
    N = 200
    sorted_idx = np.argsort(margins)
    selected = set(sorted_idx[:N].tolist())
    selected.update(mis.tolist())
    selected = np.array(list(selected))
    
    target_margin = 2.0
    
    for i in selected:
        c = y[i]
        # Find the best wrong class for this example
        wrong_logits = logits[i].copy()
        wrong_logits[c] = -np.inf
        j = int(np.argmax(wrong_logits))
        
        cur_margin = logits[i, c] - logits[i, j]
        if cur_margin >= target_margin:
            continue
        
        x = X[i]  # (1568,)
        # Change in margin if we add alpha*x to W[c] and subtract alpha*x from W[j],
        # and add alpha to b[c] and subtract alpha from b[j]:
        # delta_margin = 2 * alpha * (x·x + 1)
        denom = 2.0 * (np.dot(x, x) + 1.0)
        if denom <= 0:
            continue
        alpha = (target_margin - cur_margin) / denom
        
        W[c] += alpha * x
        b[c] += alpha
        W[j] -= alpha * x
        b[j] -= alpha
        
        # Update logits for this example so subsequent updates use fresh values
        logits[i, c] += alpha * (np.dot(x, x) + 1.0)
        logits[i, j] -= alpha * (np.dot(x, x) + 1.0)
    
    # Ensure all weights stay within [-1000, 1000]
    max_abs = max(np.max(np.abs(W)), np.max(np.abs(b)))
    if max_abs > 1000.0:
        scale = 1000.0 / max_abs
        W *= scale
        b *= scale
    
    # Write back
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Diagnostics
    new_logits = X @ W.T + b
    new_preds = np.argmax(new_logits, axis=1)
    acc = np.mean(new_preds == y)
    # Cross-entropy
    probs = np.exp(new_logits - np.max(new_logits, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
    
    diagnostics = {
        "train_accuracy": float(acc),
        "cross_entropy": float(ce),
        "num_updated": int(len(selected)),
        "max_abs_head_weight": float(np.max(np.abs(W))),
        "max_abs_head_bias": float(np.max(np.abs(b))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
