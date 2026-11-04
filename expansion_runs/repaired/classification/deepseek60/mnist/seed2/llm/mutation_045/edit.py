import numpy as np

def edit(w, f):
    # Read current fc2 bias and full-pool logits/labels
    fc2_bias = w["fc2.bias"].copy()
    logits = f["logits"]          # [2000, 10]
    y = f["y"]                    # [2000]

    # Current full-pool accuracy
    pred = np.argmax(logits, axis=1)
    current_acc = float(np.mean(pred == y))

    # Class-wise mean logits over the balanced pool
    mean_logits = logits.mean(axis=0)          # [10]
    target = float(mean_logits.mean())
    adjustment = target - mean_logits          # [10]

    # Evaluate the bias-only correction
    new_logits = logits + adjustment
    new_pred = np.argmax(new_logits, axis=1)
    new_acc = float(np.mean(new_pred == y))

    # Apply only if it helps
    applied = new_acc > current_acc
    if applied:
        w["fc2.bias"] = (fc2_bias + adjustment).astype(np.float32)

    diagnostics = {
        "current_acc": current_acc,
        "new_acc": new_acc,
        "applied": applied,
        "mean_logits_before": mean_logits.tolist(),
        "adjustment": adjustment.tolist(),
        "target_mean": target,
        "fc2_bias_after": w["fc2.bias"].tolist(),
    }
    return {"weights": w, "diagnostics": diagnostics}
