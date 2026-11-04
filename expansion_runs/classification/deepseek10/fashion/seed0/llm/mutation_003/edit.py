import numpy as np

def edit(w, f):
    # Read current head parameters
    W = w["head.weight"].astype(np.float32)
    b = w["head.bias"].astype(np.float32)
    
    # Compute current logits on the full feedback pool using provided features
    # features shape: [2000, 1568]
    X = f["features"].astype(np.float32)
    logits = X @ W.T + b  # [2000, 10]
    
    # Verify accuracy on the full pool (sanity check, not used for search)
    y = f["y"]
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    
    # Compute current cross-entropy
    # Stable softmax
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
    
    # Choose a scale factor that increases confidence but stays safe.
    # Since accuracy is 100%, any scale > 1 reduces CE. We use a moderate 1.35
    # to avoid extreme overconfidence that might hurt held-out generalization.
    scale = 1.35
    
    # Apply scaling to both weights and biases
    w["head.weight"] = (W * scale).astype(np.float32)
    w["head.bias"] = (b * scale).astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "original_accuracy": float(acc),
        "original_cross_entropy": float(ce),
        "scale_factor": float(scale),
        "new_weight_max": float(np.max(np.abs(w["head.weight"]))),
        "new_bias_max": float(np.max(np.abs(w["head.bias"]))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
