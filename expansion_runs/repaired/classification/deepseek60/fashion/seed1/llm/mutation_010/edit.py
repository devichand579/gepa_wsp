import numpy as np
import math

def edit(w, f):
    # Compute a global scale factor for the head to improve calibration.
    # Use the feedback logits to estimate the current margin between top-1 and top-2.
    logits = f['logits']  # shape (2000, 10)
    sorted_logits = np.sort(logits, axis=1)
    top1 = sorted_logits[:, -1]
    top2 = sorted_logits[:, -2]
    margins = top1 - top2
    mean_margin = float(np.mean(margins))
    
    # Target margin for ~0.9 top probability: log(9) ≈ 2.197
    target_margin = math.log(9.0)
    alpha = target_margin / (mean_margin + 1e-8)
    # Clip to a safe range to avoid extreme values.
    alpha = float(np.clip(alpha, 1.0, 50.0))
    
    # Scale the head weights and bias.
    w['head.weight'] = (w['head.weight'] * alpha).astype(np.float32)
    w['head.bias'] = (w['head.bias'] * alpha).astype(np.float32)
    
    diagnostics = {
        'alpha': alpha,
        'mean_margin': mean_margin,
        'target_margin': target_margin,
        'head_weight_std': float(np.std(w['head.weight'])),
        'head_bias_mean': float(np.mean(w['head.bias']))
    }
    return {"weights": w, "diagnostics": diagnostics}
