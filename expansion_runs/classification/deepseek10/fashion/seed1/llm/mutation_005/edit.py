import numpy as np
import math

def edit(w, f):
    # Extract full feedback logits and labels
    logits = f["logits"].astype(np.float64)  # [2000, 10]
    y = f["y"].astype(np.int64)              # [2000]
    
    # Compute current cross-entropy
    def cross_entropy(logits, y):
        # logits: [N, 10]
        # stable softmax
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # clip to avoid log(0)
        probs = np.clip(probs, 1e-12, 1.0)
        return -np.mean(np.log(probs[np.arange(len(y)), y]))
    
    current_ce = cross_entropy(logits, y)
    
    # Grid search for temperature T > 0
    # We want to minimize cross-entropy of softmax(T * logits)
    best_T = 1.0
    best_ce = current_ce
    # Try a range of temperatures
    for T in np.linspace(0.5, 20.0, 200):
        ce = cross_entropy(T * logits, y)
        if ce < best_ce:
            best_ce = ce
            best_T = T
    
    # Apply temperature scaling to head weights and bias
    # Since logits = head.weight @ features + head.bias,
    # scaling logits by T is equivalent to scaling head.weight and head.bias by T.
    w["head.weight"] = (w["head.weight"].astype(np.float32) * best_T).astype(np.float32)
    w["head.bias"] = (w["head.bias"].astype(np.float32) * best_T).astype(np.float32)
    
    # Ensure all weights are finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    diagnostics = {
        "temperature": float(best_T),
        "original_cross_entropy": float(current_ce),
        "new_cross_entropy": float(best_ce),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias": w["head.bias"].tolist()
    }
    
    return {"weights": w, "diagnostics": diagnostics}
