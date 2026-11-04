import numpy as np

def edit(w, f):
    # Compute the maximum safe scaling factor s for the convolutional stack.
    # Scaling rules:
    #   conv1.weight, conv1.bias, conv2.weight  -> multiply by s
    #   conv2.bias, head.bias                   -> multiply by s^2
    #   head.weight                             -> unchanged
    # This multiplies all logits by s^2 exactly.
    max_abs = lambda a: float(np.max(np.abs(a)))
    
    m_w1 = max_abs(w["conv1.weight"])
    m_b1 = max_abs(w["conv1.bias"])
    m_w2 = max_abs(w["conv2.weight"])
    m_b2 = max_abs(w["conv2.bias"])
    m_b3 = max_abs(w["head.bias"])
    
    # s must satisfy:
    #   s * m_w1 <= 1000
    #   s * m_b1 <= 1000
    #   s * m_w2 <= 1000
    #   s^2 * m_b2 <= 1000
    #   s^2 * m_b3 <= 1000
    eps = 1e-12
    s_limits = [
        1000.0 / (m_w1 + eps),
        1000.0 / (m_b1 + eps),
        1000.0 / (m_w2 + eps),
        np.sqrt(1000.0 / (m_b2 + eps)),
        np.sqrt(1000.0 / (m_b3 + eps)),
    ]
    s = float(np.min(s_limits))
    # Leave a tiny safety margin.
    s = max(1.0, s * 0.999)
    s2 = s * s
    
    # Apply the scaling in-place on the float32 arrays.
    w["conv1.weight"] = (w["conv1.weight"] * s).astype(np.float32)
    w["conv1.bias"]   = (w["conv1.bias"]   * s).astype(np.float32)
    w["conv2.weight"] = (w["conv2.weight"] * s).astype(np.float32)
    w["conv2.bias"]   = (w["conv2.bias"]   * s2).astype(np.float32)
    w["head.bias"]    = (w["head.bias"]    * s2).astype(np.float32)
    # head.weight is intentionally left unchanged.
    
    # Clip to the allowed range as a final safety measure.
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    diagnostics = {
        "scale_s": s,
        "scale_s2": s2,
        "max_abs_conv1_weight": float(np.max(np.abs(w["conv1.weight"]))),
        "max_abs_conv1_bias": float(np.max(np.abs(w["conv1.bias"]))),
        "max_abs_conv2_weight": float(np.max(np.abs(w["conv2.weight"]))),
        "max_abs_conv2_bias": float(np.max(np.abs(w["conv2.bias"]))),
        "max_abs_head_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_head_bias": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
