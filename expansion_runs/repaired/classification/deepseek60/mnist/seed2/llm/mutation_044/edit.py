import numpy as np

def edit(w, f):
    # Read current parameters
    fc2_bias = w["fc2.bias"].copy()
    logits = f["logits"]          # [2000, 10]
    y = f["y"]                    # [2000]
    
    # Current full-pool accuracy
    current_pred = np.argmax(logits, axis=1)
    current_acc = np.mean(current_pred == y)
    
    # Compute mean logit per class over the entire feedback pool
    mean_logits = logits.mean(axis=0)          # [10]
    target_mean = mean_logits.mean()
    adjustment = target_mean - mean_logits     # [10]
    
    # New logits after bias correction
    new_logits = logits + adjustment
    new_pred = np.argmax(new_logits, axis=1)
    new_acc = np.mean(new_pred == y)
    
    # Apply only if it improves full-pool accuracy
    if new_acc > current_acc:
        w["fc2.bias"] = (fc2_bias + adjustment).astype(np.float32)
        applied = True
    else:
        applied = False
    
    # Diagnostics
    diagnostics = {
        "current_acc": float(current_acc),
        "new_acc": float(new_acc),
        "applied": applied,
        "mean_logits_before": mean_logits.tolist(),
        "adjustment": adjustment.tolist(),
        "target_mean": float(target_mean),
        "fc2_bias_after": w["fc2.bias"].tolist()
    }
    
    return {"weights": w, "diagnostics": diagnostics}
