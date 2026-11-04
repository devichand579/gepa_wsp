import numpy as np

def edit(w, f):
    # Compute maximum absolute values for parameters that will be scaled
    max_abs_conv1_w = float(np.max(np.abs(w["conv1.weight"])))
    max_abs_conv1_b = float(np.max(np.abs(w["conv1.bias"])))
    max_abs_conv2_w = float(np.max(np.abs(w["conv2.weight"])))
    max_abs_conv2_b = float(np.max(np.abs(w["conv2.bias"])))
    max_abs_head_b = float(np.max(np.abs(w["head.bias"])))
    
    # Compute maximum allowed scale s such that all scaled params stay <= 1000
    s1 = 1000.0 / max_abs_conv1_w if max_abs_conv1_w > 0 else np.inf
    s2 = 1000.0 / max_abs_conv1_b if max_abs_conv1_b > 0 else np.inf
    s3 = 1000.0 / max_abs_conv2_w if max_abs_conv2_w > 0 else np.inf
    s4 = np.sqrt(1000.0 / max_abs_conv2_b) if max_abs_conv2_b > 0 else np.inf
    s5 = np.sqrt(1000.0 / max_abs_head_b) if max_abs_head_b > 0 else np.inf
    
    s = min(s1, s2, s3, s4, s5)
    # Safety margin to ensure strictly within bounds
    s = s * 0.9999
    s2_val = s * s
    
    # Apply scaling
    w["conv1.weight"] = (w["conv1.weight"] * s).astype(np.float32)
    w["conv1.bias"] = (w["conv1.bias"] * s).astype(np.float32)
    w["conv2.weight"] = (w["conv2.weight"] * s).astype(np.float32)
    w["conv2.bias"] = (w["conv2.bias"] * s2_val).astype(np.float32)
    w["head.bias"] = (w["head.bias"] * s2_val).astype(np.float32)
    # head.weight is unchanged
    
    # Diagnostics: compute new logits and metrics on feedback pool
    old_logits = f["logits"]
    y = f["y"]
    new_logits = old_logits * s2_val
    # stable softmax
    max_logits = np.max(new_logits, axis=1, keepdims=True)
    exp_logits = np.exp(new_logits - max_logits)
    new_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    new_ce = -np.mean(np.log(new_probs[np.arange(len(y)), y] + 1e-12))
    new_acc = np.mean(np.argmax(new_logits, axis=1) == y)
    
    # Check max abs after scaling
    max_abs_after = max(
        float(np.max(np.abs(w["conv1.weight"]))),
        float(np.max(np.abs(w["conv1.bias"]))),
        float(np.max(np.abs(w["conv2.weight"]))),
        float(np.max(np.abs(w["conv2.bias"]))),
        float(np.max(np.abs(w["head.weight"]))),
        float(np.max(np.abs(w["head.bias"])))
    )
    
    diagnostics = {
        "scale_s": float(s),
        "scale_s2": float(s2_val),
        "max_abs_conv1_weight": max_abs_conv1_w,
        "max_abs_conv1_bias": max_abs_conv1_b,
        "max_abs_conv2_weight": max_abs_conv2_w,
        "max_abs_conv2_bias": max_abs_conv2_b,
        "max_abs_head_bias": max_abs_head_b,
        "max_abs_after": max_abs_after,
        "feedback_accuracy": float(new_acc),
        "feedback_cross_entropy": float(new_ce),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
