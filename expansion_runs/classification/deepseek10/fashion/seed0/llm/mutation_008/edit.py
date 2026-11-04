import numpy as np

def edit(w, f):
    # Work on a copy to avoid mutating the input directly if needed
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    
    # Current maximum absolute values in each parameter group
    max_conv1_w = np.max(np.abs(w["conv1.weight"]))
    max_conv1_b = np.max(np.abs(w["conv1.bias"]))
    max_conv2_w = np.max(np.abs(w["conv2.weight"]))
    max_conv2_b = np.max(np.abs(w["conv2.bias"]))
    max_head_w  = np.max(np.abs(w["head.weight"]))
    max_head_b  = np.max(np.abs(w["head.bias"]))
    
    # We want to choose s (scale for conv layers) and T (scale for logits)
    # such that all parameters stay within [-1000, 1000].
    # Constraints:
    #   s * max_conv1_w <= 1000  => s <= 1000 / max_conv1_w
    #   s * max_conv1_b <= 1000  => s <= 1000 / max_conv1_b
    #   s * max_conv2_w <= 1000  => s <= 1000 / max_conv2_w
    #   s * max_conv2_b <= 1000  => s <= 1000 / max_conv2_b
    #   (T/s) * max_head_w <= 1000 => T <= 1000 * s / max_head_w
    #   T * max_head_b <= 1000 => T <= 1000 / max_head_b
    # We want to maximize T to reduce cross-entropy as much as possible.
    # So we should choose s as large as possible, then T as large as possible.
    
    # Maximum allowed s from conv layers
    s_max = min(
        1000.0 / max(max_conv1_w, 1e-12),
        1000.0 / max(max_conv1_b, 1e-12),
        1000.0 / max(max_conv2_w, 1e-12),
        1000.0 / max(max_conv2_b, 1e-12)
    )
    # Use a safety margin to avoid floating point issues
    s = s_max * 0.999
    
    # Maximum allowed T from head weights and bias
    T_max_w = 1000.0 * s / max(max_head_w, 1e-12)
    T_max_b = 1000.0 / max(max_head_b, 1e-12)
    T = min(T_max_w, T_max_b) * 0.999
    
    # Apply scaling
    w["conv1.weight"] *= s
    w["conv1.bias"]   *= s
    w["conv2.weight"] *= s
    w["conv2.bias"]   *= s
    w["head.weight"]  *= (T / s)
    w["head.bias"]    *= T
    
    # Ensure all values are finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    diagnostics = {
        "s": float(s),
        "T": float(T),
        "max_conv1_w": float(np.max(np.abs(w["conv1.weight"]))),
        "max_conv1_b": float(np.max(np.abs(w["conv1.bias"]))),
        "max_conv2_w": float(np.max(np.abs(w["conv2.weight"]))),
        "max_conv2_b": float(np.max(np.abs(w["conv2.bias"]))),
        "max_head_w": float(np.max(np.abs(w["head.weight"]))),
        "max_head_b": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
