import numpy as np

def edit(w, f):
    # Extract head bias and center by midpoint to minimize max absolute value
    b = w['head.bias'].astype(np.float64)
    c = (b.max() + b.min()) / 2.0
    b_centered = b - c
    max_abs_b = np.max(np.abs(b_centered))
    
    # Maximum K such that |K * b_centered| <= 1000
    K = 1000.0 / max_abs_b * 0.999999  # small safety margin
    s = np.sqrt(K)
    
    # Scale conv1 weights and bias by s
    w['conv1.weight'] = (w['conv1.weight'].astype(np.float64) * s).astype(np.float32)
    w['conv1.bias']   = (w['conv1.bias'].astype(np.float64) * s).astype(np.float32)
    
    # Scale conv2 weights by s, bias by K
    w['conv2.weight'] = (w['conv2.weight'].astype(np.float64) * s).astype(np.float32)
    w['conv2.bias']   = (w['conv2.bias'].astype(np.float64) * K).astype(np.float32)
    
    # Scale centered head bias by K; head weights unchanged
    w['head.bias'] = (b_centered * K).astype(np.float32)
    
    # Safety clip to ensure all parameters are within [-1000, 1000]
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    diagnostics = {
        "K": float(K),
        "s": float(s),
        "max_abs_centered_bias": float(max_abs_b),
        "max_abs_conv1_weight": float(np.max(np.abs(w['conv1.weight']))),
        "max_abs_conv2_bias": float(np.max(np.abs(w['conv2.bias']))),
        "max_abs_head_bias": float(np.max(np.abs(w['head.bias']))),
        "max_abs_head_weight": float(np.max(np.abs(w['head.weight']))),
    }
    return {"weights": w, "diagnostics": diagnostics}
