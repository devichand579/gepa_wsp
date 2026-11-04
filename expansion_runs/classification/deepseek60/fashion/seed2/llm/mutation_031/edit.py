import numpy as np

def edit(w, f):
    logits = f["logits"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N = logits.shape[0]

    def softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def compute_ce(alpha):
        L = alpha * logits
        p = softmax(L)
        return -np.mean(np.log(p[np.arange(N), y] + 1e-12))

    def compute_derivs(alpha):
        L = alpha * logits
        p = softmax(L)
        m = np.sum(p * L, axis=1)
        d1 = -np.sum(L[np.arange(N), y]) + np.sum(m)
        d2 = np.sum(np.sum(p * L**2, axis=1) - m**2)
        return d1, d2

    ce_before = compute_ce(1.0)

    # Two Newton steps to find the optimal scaling factor (no iterative loop)
    alpha = 1.0
    d1, d2 = compute_derivs(alpha)
    if d2 > 1e-12:
        alpha = alpha - d1 / d2
    d1, d2 = compute_derivs(alpha)
    if d2 > 1e-12:
        alpha = alpha - d1 / d2

    alpha = float(np.clip(alpha, 0.1, 100.0))
    ce_after = compute_ce(alpha)

    # Apply temperature scaling to the head only
    w["head.weight"] = (w["head.weight"] * alpha).astype(np.float32)
    w["head.bias"] = (w["head.bias"] * alpha).astype(np.float32)

    diagnostics = {
        "alpha": alpha,
        "ce_before": float(ce_before),
        "ce_after": float(ce_after),
        "accuracy_before": float(np.mean(np.argmax(logits, axis=1) == y)),
        "accuracy_after": float(np.mean(np.argmax(alpha * logits, axis=1) == y)),
    }
    return {"weights": w, "diagnostics": diagnostics}
