import numpy as np

def edit(w, f):
    # Work on a copy to avoid mutating the input directly if needed
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    
    # Current maximum absolute values (for safety)
    max_conv1_w = np.max(np.abs(w["conv1.weight"]))
    max_conv1_b = np.max(np.abs(w["conv1.bias"]))
    max_conv2_b = np.max(np.abs(w["conv2.bias"]))
    max_head_w = np.max(np.abs(w["head.weight"]))
    max_head_b = np.max(np.abs(w["head.bias"]))
    
    # We want to scale logits by T. Constraints:
    # a * max_conv1_w <= 1000
    # a * max_conv1_b <= 1000
    # a * max_conv2_b <= 1000
    # (T/a) * max_head_w <= 1000
    # T * max_head_b <= 1000
    # Choose T as large as possible given head.bias, and a to satisfy head.weight.
    T = min(1000.0 / max_head_b, 78.0)  # use 78 to stay safe
    # Need T/a <= 1000/max_head_w  => a >= T * max_head_w / 1000
    a_min = T * max_head_w / 1000.0
    # Also a <= 1000/max_conv1_w, etc.
    a_max = min(1000.0 / max_conv1_w, 1000.0 / max_conv1_b, 1000.0 / max_conv2_b)
    # Pick a safely in the middle, but ensure a >= a_min
    a = max(a_min * 1.05, 1.0)  # small margin
    if a > a_max:
        a = a_max
    # Recompute T if a is constrained (should not happen with our numbers)
    T = min(T, a * 1000.0 / max_head_w, 1000.0 / max_head_b)
    
    # Apply scaling
    w["conv1.weight"] *= a
    w["conv1.bias"] *= a
    w["conv2.bias"] *= a
    w["head.weight"] *= (T / a)
    w["head.bias"] *= T
    
    # Clip to ensure within bounds (safety)
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    # Diagnostics
    diag = {
        "T": float(T),
        "a": float(a),
        "max_conv1_w": float(np.max(np.abs(w["conv1.weight"]))),
        "max_conv1_b": float(np.max(np.abs(w["conv1.bias"]))),
        "max_conv2_b": float(np.max(np.abs(w["conv2.bias"]))),
        "max_head_w": float(np.max(np.abs(w["head.weight"]))),
        "max_head_b": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diag}
