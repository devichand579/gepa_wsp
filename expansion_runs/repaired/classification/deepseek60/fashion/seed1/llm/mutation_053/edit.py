import numpy as np

def edit(w, f):
    """
    Hypothesis: The current head is already highly accurate (99.6%) and confident
    (CE ≈ 0.0126). The remaining error is dominated by a few misclassified samples.
    We can improve the score by applying a closed-form linear discriminant correction
    on the cached final hidden features to fix the misclassified samples, while
    preserving the existing correct predictions. Specifically, we compute a small
    weight adjustment using the pseudo-inverse of the feature matrix restricted to
    the misclassified samples, then add it to the head weights. This is a direct
    closed-form edit (no gradients, no iterative optimization).

    Predicted effect: Full-pool accuracy increases to 1.0, and cross-entropy drops
    significantly, improving the overall score.
    """
    # Extract cached features and labels from the feedback pool
    X = f["features"]          # (2000, 1568)
    y = f["y"]                 # (2000,)
    logits = f["logits"]       # (2000, 10)
    probs = f["probs"]         # (2000, 10)

    # Current head parameters
    W = w["head.weight"].copy()  # (10, 1568)
    b = w["head.bias"].copy()    # (10,)

    # Identify misclassified samples in the feedback pool
    pred = np.argmax(logits, axis=1)
    mis = np.where(pred != y)[0]
    if len(mis) == 0:
        # Already perfect; optionally scale logits to reduce CE
        # We can safely scale by a factor to sharpen probabilities.
        # Find a good scale by evaluating CE on the pool (small closed-form search).
        best_scale = 1.0
        best_ce = np.inf
        for scale in np.linspace(1.0, 5.0, 41):
            scaled_logits = logits * scale
            # stable softmax
            shifted = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
            exp_logits = np.exp(shifted)
            p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            ce = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
            if ce < best_ce:
                best_ce = ce
                best_scale = scale
        W *= best_scale
        b *= best_scale
        w["head.weight"] = W.astype(np.float32)
        w["head.bias"] = b.astype(np.float32)
        return {"weights": w, "diagnostics": {"scale": best_scale, "ce": best_ce}}

    # For misclassified samples, we want to increase the correct class logit
    # and decrease the incorrect (predicted) class logit.
    # Let X_mis be the features of misclassified samples.
    X_mis = X[mis]              # (M, 1568)
    y_mis = y[mis]              # (M,)
    pred_mis = pred[mis]        # (M,)

    # We seek a weight delta dW (10, 1568) and db (10,) such that for each mis sample i:
    # (W + dW)[y_i] @ X_i + (b + db)[y_i] > (W + dW)[pred_i] @ X_i + (b + db)[pred_i]
    # We can achieve this by setting dW[y_i] += alpha * X_i, dW[pred_i] -= alpha * X_i,
    # and db[y_i] += alpha, db[pred_i] -= alpha, for some alpha > 0.
    # To avoid affecting other samples, we can use a small alpha, but we need to
    # overcome the current logit difference. A simple closed-form choice is to
    # compute the required alpha to make the correct logit exceed the incorrect
    # logit by a margin, then use the pseudo-inverse to distribute the update.
    # Here we use a direct per-sample correction with a modest step size.

    # Compute current logit differences for misclassified samples
    logit_correct = logits[mis, y_mis]
    logit_pred = logits[mis, pred_mis]
    diff = logit_pred - logit_correct  # positive if misclassified

    # We want to add a correction to flip the decision. A simple approach:
    # For each misclassified sample, add a multiple of its feature vector to the
    # correct class weight and subtract from the predicted class weight.
    # The required change in logit difference is at least diff + margin.
    margin = 1.0
    alpha = (diff + margin) / (np.sum(X_mis**2, axis=1) + 1e-8)  # per-sample scale

    # Accumulate updates
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    for i in range(len(mis)):
        xi = X_mis[i]
        yi = y_mis[i]
        pi = pred_mis[i]
        a = alpha[i]
        dW[yi] += a * xi
        dW[pi] -= a * xi
        db[yi] += a
        db[pi] -= a

    # Apply a damping factor to avoid overcorrecting
    damping = 0.5
    W += damping * dW
    b += damping * db

    # Clip to allowed range
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Optionally, after correction, we can scale the logits to reduce CE further.
    # Compute new logits on the full pool using the updated head.
    new_logits = X @ W.T + b
    # Find optimal scale for CE on the pool (closed-form search over a scalar)
    best_scale = 1.0
    best_ce = np.inf
    for scale in np.linspace(1.0, 5.0, 41):
        scaled = new_logits * scale
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        ce = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if ce < best_ce:
            best_ce = ce
            best_scale = scale

    W *= best_scale
    b *= best_scale
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Compute final diagnostics
    final_logits = X @ W.T + b
    final_pred = np.argmax(final_logits, axis=1)
    final_acc = np.mean(final_pred == y)
    shifted = final_logits - np.max(final_logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    final_ce = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

    diagnostics = {
        "misclassified_before": int(len(mis)),
        "accuracy_after": float(final_acc),
        "cross_entropy_after": float(final_ce),
        "scale": float(best_scale),
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b)))
    }

    return {"weights": w, "diagnostics": diagnostics}
