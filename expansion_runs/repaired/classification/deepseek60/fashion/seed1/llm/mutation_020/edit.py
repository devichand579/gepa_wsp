import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # Augment with bias term
    Xb = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # [2000, 1569]

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Ridge regression: W = (X^T X + lambda I)^-1 X^T Y
    # Use a small lambda to avoid overfitting while keeping high accuracy.
    # We'll try a few lambdas and pick the one with best full-pool accuracy.
    best_acc = -1.0
    best_W = None
    best_b = None
    best_lambda = None

    # Candidate lambdas (log-spaced)
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    for lam in lambdas:
        # Solve (Xb^T Xb + lam I) W = Xb^T Y
        A = Xb.T @ Xb + lam * np.eye(D + 1, dtype=np.float64)
        B = Xb.T @ Y
        W_full = np.linalg.solve(A, B)  # [1569, 10]
        W_candidate = W_full[:D, :].T   # [10, 1568]
        b_candidate = W_full[D, :]      # [10]

        # Evaluate on full pool
        logits = X @ W_candidate.T + b_candidate
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        if acc > best_acc:
            best_acc = acc
            best_W = W_candidate
            best_b = b_candidate
            best_lambda = lam

    # If for some reason nothing found, fall back to current head
    if best_W is None:
        return w

    # Optional: rescale logits to reduce cross-entropy without changing argmax.
    # We can scale W and b by a positive factor s. We'll choose s to minimize
    # cross-entropy on the full pool (or just use s=1 if it's already good).
    # Compute logits with best_W, best_b
    logits = X @ best_W.T + best_b
    # Softmax cross-entropy
    def ce(logits, y):
        # logits: [N, C]
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    base_ce = ce(logits, y)
    # Try a few scales
    best_scale = 1.0
    best_score = -np.inf
    # We don't know the exact score formula, but we can try to minimize CE while keeping accuracy.
    # Since scaling doesn't change argmax, accuracy is constant. We'll pick scale that minimizes CE.
    # Actually, scaling logits by s > 1 reduces CE if accuracy is high, but can increase CE if there are errors.
    # We'll search over s in a reasonable range.
    for s in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]:
        scaled_logits = logits * s
        c = ce(scaled_logits, y)
        if c < best_score or best_score == -np.inf:
            best_score = c
            best_scale = s

    # Apply best scale
    final_W = best_W * best_scale
    final_b = best_b * best_scale

    # Clip to allowed range
    final_W = np.clip(final_W, -1000.0, 1000.0).astype(np.float32)
    final_b = np.clip(final_b, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["head.weight"] = final_W
    w["head.bias"] = final_b

    # Diagnostics
    diagnostics = {
        "best_lambda": best_lambda,
        "full_accuracy": float(best_acc),
        "base_ce": float(base_ce),
        "best_scale": float(best_scale),
        "scaled_ce": float(best_score),
        "head_weight_mean": float(np.mean(final_W)),
        "head_weight_std": float(np.std(final_W)),
        "head_bias_mean": float(np.mean(final_b)),
        "head_bias_std": float(np.std(final_b)),
    }

    return {"weights": w, "diagnostics": diagnostics}
