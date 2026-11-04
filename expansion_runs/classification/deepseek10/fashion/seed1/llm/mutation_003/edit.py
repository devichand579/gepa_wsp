import numpy as np

def edit(w, f):
    # Temperature scaling of the final linear head.
    # This is a closed-form, parameter-free sharpening of the existing classifier.
    T = 4.0

    # Read current head parameters
    head_w = w["head.weight"].astype(np.float32)
    head_b = w["head.bias"].astype(np.float32)

    # Scale both weight and bias by T
    w["head.weight"] = (head_w * T).astype(np.float32)
    w["head.bias"] = (head_b * T).astype(np.float32)

    # Optional diagnostics: compute training accuracy and cross-entropy after scaling
    # using the provided logits (which are the pre-scaling logits).
    logits = f["logits"].astype(np.float32)  # [N, 10]
    y = f["y"].astype(np.int64)              # [N]
    N = logits.shape[0]

    scaled_logits = logits * T
    # stable softmax
    max_logits = np.max(scaled_logits, axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "temperature": T,
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "head_weight_mean": float(np.mean(w["head.weight"])),
        "head_weight_std": float(np.std(w["head.weight"])),
        "head_bias": w["head.bias"].tolist()
    }

    return {"weights": w, "diagnostics": diagnostics}
