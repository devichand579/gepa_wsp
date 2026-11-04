import numpy as np

def edit(w, f):
    # Copy weights to avoid mutating input in place (though we return w)
    w = {k: v.copy() for k, v in w.items()}
    
    # Extract current head bias
    b = w['head.bias'].astype(np.float64)
    b_min = b.min()
    b_max = b.max()
    mid = (b_max + b_min) / 2.0
    M = max(abs(b_min - mid), abs(b_max - mid))  # = (b_max - b_min)/2
    
    # Maximum logit scale k allowed by head bias range
    k_max = 1000.0 / M if M > 0 else 1.0
    # We can also scale conv features, but k is limited by head bias.
    # Choose k = k_max, and set s^2 = k so that head.weight is unchanged.
    k = k_max
    s2 = k
    s = np.sqrt(s2)
    
    # Apply scaling
    w['conv1.weight'] *= s
    w['conv1.bias'] *= s
    w['conv2.weight'] *= s
    w['conv2.bias'] *= s2
    w['head.weight'] *= (k / s2)  # = 1.0, but keep general
    # Center and scale head bias: new_b = k * (b - mid)
    w['head.bias'] = (k * (b - mid)).astype(np.float32)
    
    # Ensure all parameters are finite and within [-1000, 1000]
    for key in w:
        w[key] = np.clip(w[key], -1000.0, 1000.0).astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "k": float(k),
        "s": float(s),
        "s2": float(s2),
        "head_bias_range": float(b_max - b_min),
        "max_abs_head_bias_after": float(np.max(np.abs(w['head.bias']))),
        "max_abs_head_weight": float(np.max(np.abs(w['head.weight']))),
        "max_abs_conv1_weight": float(np.max(np.abs(w['conv1.weight']))),
        "max_abs_conv2_bias": float(np.max(np.abs(w['conv2.bias']))),
    }
    return {"weights": w, "diagnostics": diagnostics}
