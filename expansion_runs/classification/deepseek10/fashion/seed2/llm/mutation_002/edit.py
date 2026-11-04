import numpy as np

def edit(w, f):
    # Copy weights to avoid mutating input
    w = {k: v.copy() for k, v in w.items()}
    
    # Only adjust the head with a very small shrinkage factor
    # This is a closed-form ridge-like regularization (shrinkage toward zero)
    # The factor is chosen to be extremely small to avoid harming the already good head.
    shrink = 0.9995
    
    # Apply shrinkage to head weights
    w['head.weight'] = w['head.weight'] * shrink
    
    # Adjust head bias to compensate for the shrinkage in weights
    # We want to preserve the average logit magnitude. Since features are non-negative,
    # the mean logit shift is approximately (1 - shrink) * mean(features) @ head.weight.T
    # We can compute this shift using the provided features and current head weights.
    features = f['features']  # [2000, 1568]
    head_w = w['head.weight']  # [10, 1568] (already shrunk)
    # Compute the mean feature vector
    mean_feat = np.mean(features, axis=0)  # [1568]
    # The original head weights before shrinkage
    orig_head_w = w['head.weight'] / shrink
    # The shift in logits due to shrinkage
    shift = (1.0 - shrink) * (orig_head_w @ mean_feat)  # [10]
    # Add the shift to the bias to compensate
    w['head.bias'] = w['head.bias'] + shift
    
    # Ensure all weights are finite float32 and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "shrink_factor": shrink,
        "head_weight_norm": float(np.linalg.norm(w['head.weight'])),
        "head_bias_norm": float(np.linalg.norm(w['head.bias'])),
        "mean_logit_shift": float(np.mean(shift))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
