import numpy as np

def edit(w, f):
    # Extract current head parameters
    W = w["head.weight"].astype(np.float64)  # [10, 1568]
    b = w["head.bias"].astype(np.float64)    # [10]
    
    # Get current logits and labels from feedback
    logits = f["logits"].astype(np.float64)  # [2000, 10]
    y = f["y"].astype(np.int64)              # [2000]
    
    # Compute current probabilities and cross-entropy
    # Use stable softmax
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)
    n = len(y)
    ce = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
    
    # Current average correct probability
    avg_correct = np.mean(probs[np.arange(n), y])
    
    # We want to increase confidence. A simple heuristic: scale by a factor
    # that reduces CE by roughly half. Since CE = -log(avg_correct) approximately,
    # we can target a new average correct probability.
    # Current CE = 1.5286, so avg_correct ≈ exp(-1.5286) ≈ 0.217.
    # We target avg_correct = 0.5 (CE ≈ 0.693). The required scale can be
    # estimated from the logit margins. For a two-class case, margin = logit_correct - logit_max_other.
    # We compute the average margin and set scale = log(target/(1-target)) / avg_margin.
    # This is a closed-form estimate.
    
    # Compute margins: correct logit minus max other logit
    correct_logits = logits[np.arange(n), y]
    # Set correct logit to -inf to find max other
    logits_other = logits.copy()
    logits_other[np.arange(n), y] = -np.inf
    max_other = logits_other.max(axis=1)
    margins = correct_logits - max_other
    
    # Average margin
    avg_margin = np.mean(margins)
    if avg_margin <= 0:
        # Should not happen since accuracy is 1.0, but just in case
        scale = 1.0
    else:
        # Target correct probability for the binary approximation
        target_p = 0.5
        # For binary logistic, p = sigmoid(scale * margin). We want p = target_p.
        # scale = logit(target_p) / margin = log(target_p/(1-target_p)) / margin
        # logit(0.5) = 0
        # So scale = 0. This is not useful. Let's target a higher p, e.g., 0.8.
        target_p = 0.8
        logit_target = np.log(target_p / (1 - target_p))  # ≈ 1.386
        scale = logit_target / avg_margin
    
    # Clamp scale to a reasonable range to avoid extreme values
    scale = np.clip(scale, 1.0, 10.0)
    
    # Apply scaling to head weights and bias
    w["head.weight"] = (W * scale).astype(np.float32)
    w["head.bias"] = (b * scale).astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "scale": float(scale),
        "avg_margin": float(avg_margin),
        "original_ce": float(ce),
        "original_avg_correct": float(avg_correct),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
