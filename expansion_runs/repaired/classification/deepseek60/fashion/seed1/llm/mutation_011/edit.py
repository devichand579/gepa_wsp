import numpy as np

def edit(w, f):
    # Extract current head parameters
    W = w["head.weight"].astype(np.float64)  # [10, 1568]
    b = w["head.bias"].astype(np.float64)    # [10]

    # Use the full feedback logits to estimate the current mean margin
    logits = f["logits"].astype(np.float64)  # [N, 10]
    # Sort logits to get top-1 and top-2
    sorted_logits = np.sort(logits, axis=1)
    top1 = sorted_logits[:, -1]
    top2 = sorted_logits[:, -2]
    mean_margin = np.mean(top1 - top2)

    # Target margin: log(9) ~ 2.197, corresponding to ~0.9 top probability
    target_margin = np.log(9.0)

    # Avoid division by zero
    if mean_margin > 1e-8:
        alpha = target_margin / mean_margin
    else:
        alpha = 1.0

    # Clamp alpha to a reasonable range to avoid extreme values
    alpha = float(np.clip(alpha, 0.5, 10.0))

    # Apply scaling to head weights and bias
    w["head.weight"] = (W * alpha).astype(np.float32)
    w["head.bias"] = (b * alpha).astype(np.float32)

    # Ensure all weights are finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "alpha": alpha,
        "mean_margin": float(mean_margin),
        "target_margin": float(target_margin),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias_mean": float(np.mean(w["head.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
