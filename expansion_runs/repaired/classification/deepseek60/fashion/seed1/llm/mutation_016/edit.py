import numpy as np

def edit(w, f):
    # Scale the final linear layer to sharpen probabilities without changing argmax.
    scale = 40.0
    w["head.weight"] = (w["head.weight"].astype(np.float32) * scale).astype(np.float32)
    w["head.bias"]   = (w["head.bias"].astype(np.float32) * scale).astype(np.float32)

    # Diagnostics: report the scaling and the resulting paired-batch metrics.
    logits = f["layer.head.output"].astype(np.float32) * scale
    y = f["batch.y"].astype(np.int64)
    # Softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    acc = float((preds == y).mean())
    # Cross-entropy
    eps = 1e-12
    ce = float(-np.log(probs[np.arange(len(y)), y] + eps).mean())

    diagnostics = {
        "scale": scale,
        "paired_accuracy": acc,
        "paired_cross_entropy": ce,
        "head_weight_max_abs": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_max_abs": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
