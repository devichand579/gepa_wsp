import numpy as np

def edit(w, f):
    # The current head appears to be the earlier good head multiplied by 40.
    # Rescale it back to restore the well-calibrated logits.
    scale = 40.0
    w["head.weight"] = (w["head.weight"] / scale).astype(np.float32)
    w["head.bias"] = (w["head.bias"] / scale).astype(np.float32)
    
    # Optional diagnostics: compute current paired-batch accuracy and CE
    # using the provided logits/probs is not necessary for the edit.
    diagnostics = {
        "scale_applied": 1.0 / scale,
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias_mean": float(np.mean(w["head.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
