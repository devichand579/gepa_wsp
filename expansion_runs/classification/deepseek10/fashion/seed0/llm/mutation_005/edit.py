import numpy as np

def edit(w, f):
    # Read current parameters
    conv1_w = w["conv1.weight"]
    conv1_b = w["conv1.bias"]
    head_w = w["head.weight"]
    head_b = w["head.bias"]
    
    # Compute maximum absolute values to determine safe scaling
    max_head_b = np.max(np.abs(head_b))
    max_head_w = np.max(np.abs(head_w))
    max_conv1_w = np.max(np.abs(conv1_w))
    max_conv1_b = np.max(np.abs(conv1_b))
    
    # We want to scale logits by T. Constraint: T * max_head_b <= 1000
    T = 1000.0 / max_head_b * 0.99  # 0.99 safety margin
    # We need s such that (T/s) * max_head_w <= 1000  => s >= T * max_head_w / 1000
    s = T * max_head_w / 1000.0 * 1.01  # 1.01 safety margin
    # Also ensure conv1 weights and bias stay within bounds
    # s * max_conv1_w <= 1000 and s * max_conv1_b <= 1000
    s_max = min(1000.0 / max_conv1_w, 1000.0 / max_conv1_b) * 0.99
    if s > s_max:
        s = s_max
        # Recompute T to keep head weights within bounds
        T = s * 1000.0 / max_head_w * 0.99
        # Re-check head bias bound
        if T * max_head_b > 1000:
            T = 1000.0 / max_head_b * 0.99
    
    # Apply scaling
    w["conv1.weight"] = (conv1_w * s).astype(np.float32)
    w["conv1.bias"] = (conv1_b * s).astype(np.float32)
    w["head.weight"] = (head_w * (T / s)).astype(np.float32)
    w["head.bias"] = (head_b * T).astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "T": float(T),
        "s": float(s),
        "max_conv1_weight": float(np.max(np.abs(w["conv1.weight"]))),
        "max_conv1_bias": float(np.max(np.abs(w["conv1.bias"]))),
        "max_head_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_head_bias": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
