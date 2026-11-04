import numpy as np

def edit(w, f):
    # Extract current fc2 bias and logits
    fc2_bias = w["fc2.bias"].copy()
    logits = f["logits"]  # shape (2000, 10)
    
    # Compute mean logit per class over the entire feedback pool
    mean_logits = logits.mean(axis=0)  # shape (10,)
    target_mean = mean_logits.mean()
    
    # Compute adjustment to equalize mean logits
    adjustment = target_mean - mean_logits
    
    # Apply adjustment to fc2 bias
    new_bias = fc2_bias + adjustment.astype(np.float32)
    
    # Ensure finite and within bounds
    new_bias = np.clip(new_bias, -1000.0, 1000.0)
    
    # Update weights
    w["fc2.bias"] = new_bias.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "mean_logits_before": mean_logits.tolist(),
        "adjustment": adjustment.tolist(),
        "new_bias": new_bias.tolist(),
        "target_mean": float(target_mean),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
