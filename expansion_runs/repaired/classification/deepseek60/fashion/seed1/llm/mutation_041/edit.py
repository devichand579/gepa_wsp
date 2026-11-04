import numpy as np

def edit(w, f):
    # Extract cached final hidden features and labels for the entire feedback pool
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    num_classes = 10

    # Build target matrix: +1 for correct class, -1 for others.
    # This is a standard least-squares classification target.
    T = np.full((N, num_classes), -1.0, dtype=np.float64)
    T[np.arange(N), y] = 1.0

    # Identify current errors to upweight them in the least-squares fit.
    logits_cur = f["logits"].astype(np.float64)
    pred_cur = np.argmax(logits_cur, axis=1)
    errors = pred_cur != y

    # Sample weights: heavily upweight the currently misclassified examples.
    w_sample = np.ones(N, dtype=np.float64)
    w_sample[errors] = 100.0

    # Augment features with a bias column.
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)

    # Weighted least-squares: solve for W_aug (1569, 10)
    sw = np.sqrt(w_sample)[:, None]
    Xw = X_aug * sw
    Tw = T * sw
    W_aug, _, _, _ = np.linalg.lstsq(Xw, Tw, rcond=None)

    # Split into weight and bias.
    W_new = W_aug[:-1, :].T   # (10, 1568)
    b_new = W_aug[-1, :]      # (10,)

    # Check training accuracy of the new head.
    logits_new = X_aug @ W_aug
    pred_new = np.argmax(logits_new, axis=1)
    acc = np.mean(pred_new == y)

    # Scale the head to use the allowed magnitude, but only if we achieved
    # perfect separation (otherwise scaling amplifies errors).
    if acc >= 0.9999:
        max_val = max(np.max(np.abs(W_new)), np.max(np.abs(b_new)))
        if max_val > 0:
            scale = 1000.0 / max_val
            W_new *= scale
            b_new *= scale
    else:
        # Moderate scaling to reduce cross-entropy without over-amplifying errors.
        max_val = max(np.max(np.abs(W_new)), np.max(np.abs(b_new)))
        if max_val > 0:
            scale = 100.0 / max_val
            W_new *= scale
            b_new *= scale

    # Ensure float32 and respect the absolute value bound.
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    # Write back to the parameter dictionary.
    w["head.weight"] = W_new
    w["head.bias"] = b_new

    diagnostics = {
        "train_accuracy": float(acc),
        "num_errors_before": int(np.sum(errors)),
        "num_errors_after": int(np.sum(pred_new != y)),
        "max_abs_weight": float(np.max(np.abs(W_new))),
        "max_abs_bias": float(np.max(np.abs(b_new))),
    }

    return {"weights": w, "diagnostics": diagnostics}
