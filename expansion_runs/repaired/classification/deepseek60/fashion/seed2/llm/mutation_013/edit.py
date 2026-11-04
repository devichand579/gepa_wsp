import numpy as np

def edit(w, f):
    # Scale the head to reduce cross-entropy while preserving argmax.
    # The maximum absolute weight must stay <= 1000.
    max_abs = max(
        np.abs(w["head.weight"]).max(),
        np.abs(w["head.bias"]).max()
    )
    # Use 99% of the maximum allowed scale for safety.
    s = 1000.0 / max_abs * 0.99
    # Apply scaling to both weight and bias.
    w["head.weight"] = (w["head.weight"] * s).astype(np.float32)
    w["head.bias"] = (w["head.bias"] * s).astype(np.float32)
    
    diagnostics = {
        "scale": float(s),
        "max_abs_before": float(max_abs),
        "max_abs_after": float(max(np.abs(w["head.weight"]).max(), np.abs(w["head.bias"]).max()))
    }
    return {"weights": w, "diagnostics": diagnostics}
