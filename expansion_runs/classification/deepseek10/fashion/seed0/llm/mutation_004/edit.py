import numpy as np

def edit(w, f):
    # Copy weights to avoid mutating input
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    
    # Current maximum absolute values
    max_conv1_w = np.max(np.abs(w["conv1.weight"]))
    max_conv1_b = np.max(np.abs(w["conv1.bias"]))
    max_conv2_w = np.max(np.abs(w["conv2.weight"]))
    max_conv2_b = np.max(np.abs(w["conv2.bias"]))
    max_head_w = np.max(np.abs(w["head.weight"]))
    max_head_b = np.max(np.abs(w["head.bias"]))
    
    # We want new logits = T * old_logits.
    # Transformation: conv* -> s * conv*, head.w -> (T/s) * head.w, head.b -> T * head.b
    # Constraints: all new params <= 1000.
    # s * max_conv <= 1000  => s <= 1000 / max_conv
    # (T/s) * max_head_w <= 1000 => T <= 1000 * s / max_head_w
    # T * max_head_b <= 1000 => T <= 1000 / max_head_b
    
    # Maximum T from head bias
    T_max_bias = 1000.0 / max_head_b if max_head_b > 0 else 1000.0
    
    # We can choose s to allow T up to T_max_bias.
    # Need s >= T * max_head_w / 1000.
    # Also s <= 1000 / max_conv (for all conv params).
    # To maximize T, we want s as large as possible, but T is limited by bias.
    # Actually T is limited by bias regardless of s, as long as we can satisfy the head weight constraint.
    # So set T = T_max_bias, then choose s = T * max_head_w / 1000 (minimum s to satisfy head weight constraint).
    # But also s must be <= 1000 / max_conv. Check if feasible.
    
    T = T_max_bias
    # Minimum s to keep head weights <= 1000
    s_min = T * max_head_w / 1000.0
    # Maximum s allowed by conv weights
    s_max_conv1_w = 1000.0 / max_conv1_w if max_conv1_w > 0 else np.inf
    s_max_conv1_b = 1000.0 / max_conv1_b if max_conv1_b > 0 else np.inf
    s_max_conv2_w = 1000.0 / max_conv2_w if max_conv2_w > 0 else np.inf
    s_max_conv2_b = 1000.0 / max_conv2_b if max_conv2_b > 0 else np.inf
    s_max = min(s_max_conv1_w, s_max_conv1_b, s_max_conv2_w, s_max_conv2_b)
    
    if s_min <= s_max:
        s = s_min
    else:
        # If not feasible, reduce T to maximum feasible
        # Need T <= s_max * 1000 / max_head_w and T <= T_max_bias
        T = min(T_max_bias, s_max * 1000.0 / max_head_w)
        s = s_max  # use maximum s to allow largest T
    
    # Apply scaling
    w["conv1.weight"] *= s
    w["conv1.bias"] *= s
    w["conv2.weight"] *= s
    w["conv2.bias"] *= s
    w["head.weight"] *= (T / s)
    w["head.bias"] *= T
    
    # Ensure float32 and within bounds (clip just in case)
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "T": float(T),
        "s": float(s),
        "max_conv1_w": float(np.max(np.abs(w["conv1.weight"]))),
        "max_conv2_w": float(np.max(np.abs(w["conv2.weight"]))),
        "max_head_w": float(np.max(np.abs(w["head.weight"]))),
        "max_head_b": float(np.max(np.abs(w["head.bias"]))),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
