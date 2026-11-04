import numpy as np

def edit(w, f):
    # Read current head and cached final features.
    W = w["head.weight"].astype(np.float64)   # [10, 1568]
    b = w["head.bias"].astype(np.float64)     # [10]
    X = f["features"].astype(np.float64)      # [2000, 1568]
    y = f["y"].astype(np.int64)               # [2000]

    # Current logits and predictions.
    logits = X @ W.T + b[None, :]
    pred = np.argmax(logits, axis=1)

    # Build a target adjustment: for misclassified samples, increase correct-class
    # logit and decrease the strongest wrong-class logit. For correct samples,
    # target zero adjustment.
    delta = np.zeros_like(logits)
    mis = np.where(pred != y)[0]
    for i in mis:
        c = y[i]
        wrong = np.argmax(np.where(np.arange(10) == c, -np.inf, logits[i]))
        # Desired margin increase: push correct class up and wrong class down.
        delta[i, c] = 1.0
        delta[i, wrong] = -1.0

    # Solve for a small head correction (dW, db) that best produces delta.
    # We augment features with a bias column and solve ridge-regularized least squares.
    # Minimize || [X, 1] @ [dW.T; db] - delta ||^2 + lambda * ||dW||^2
    # This is a closed-form normal-equation fit, not a gradient step.
    n, d = X.shape
    Xa = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)  # [n, d+1]
    # We solve for each output class independently.
    # Regularization keeps the correction small.
    lam = 1e-2
    A = Xa.T @ Xa + lam * np.eye(d + 1)
    # Solve A @ Z = Xa.T @ delta  => Z is [d+1, 10]
    Z = np.linalg.solve(A, Xa.T @ delta)
    dW = Z[:d, :].T   # [10, d]
    db = Z[d, :]      # [10]

    # Apply a conservative step: only a fraction of the correction to avoid
    # disturbing already-correct samples.
    alpha = 0.25
    W_new = W + alpha * dW
    b_new = b + alpha * db

    # Clip to allowed bounds.
    W_new = np.clip(W_new, -1000.0, 1000.0)
    b_new = np.clip(b_new, -1000.0, 1000.0)

    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)

    # Diagnostics.
    new_logits = X @ W_new.T + b_new[None, :]
    new_pred = np.argmax(new_logits, axis=1)
    acc = float(np.mean(new_pred == y))
    # Stable cross-entropy.
    shifted = new_logits - np.max(new_logits, axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    ce = float(-np.mean(np.log(probs[np.arange(n), y] + 1e-12)))

    diagnostics = {
        "train_accuracy": acc,
        "train_cross_entropy": ce,
        "num_misclassified": int(np.sum(new_pred != y)),
        "max_abs_head_weight": float(np.max(np.abs(W_new))),
        "max_abs_head_bias": float(np.max(np.abs(b_new))),
    }
    return {"weights": w, "diagnostics": diagnostics}
