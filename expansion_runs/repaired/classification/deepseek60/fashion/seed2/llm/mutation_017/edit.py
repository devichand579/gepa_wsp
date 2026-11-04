import numpy as np

def edit(w, f):
    # Extract current head parameters
    W = w["head.weight"].astype(np.float64)  # (10, 1568)
    b = w["head.bias"].astype(np.float64)    # (10,)
    X = f["features"].astype(np.float64)     # (2000, 1568)
    y = f["y"].astype(np.int64)              # (2000,)
    L = f["logits"].astype(np.float64)       # (2000, 10)

    # Find misclassified samples
    pred = np.argmax(L, axis=1)
    mis = np.where(pred != y)[0]
    if len(mis) == 0:
        return w  # already perfect

    # We'll fix the first misclassified sample (usually only one)
    i = mis[0]
    y_true = y[i]
    y_pred = pred[i]
    x_i = X[i]

    # Compute a direction v that is small on all other samples.
    # v = (X^T X + lambda I)^-1 x_i, then scale so x_i @ v = 1.
    lam = 1e-4
    A = X.T @ X + lam * np.eye(X.shape[1])
    v = np.linalg.solve(A, x_i)
    v = v / (x_i @ v)

    # Current margin for the misclassified sample
    m = L[i, y_true] - L[i, y_pred]

    # Compute c_j = X_j @ v for all samples
    c = X @ v

    # Required alpha to make margin zero: m + 2*alpha = 0 -> alpha = -m/2
    alpha_req = -m / 2.0

    # We'll search for alpha in a range that fixes the sample and keeps others correct.
    # Bounds from weight limits
    max_abs_W_true = np.max(np.abs(W[y_true]))
    max_abs_W_pred = np.max(np.abs(W[y_pred]))
    max_abs_v = np.max(np.abs(v))
    alpha_max_true = (1000.0 - max_abs_W_true) / max_abs_v if max_abs_v > 0 else 1e9
    alpha_max_pred = (1000.0 - max_abs_W_pred) / max_abs_v if max_abs_v > 0 else 1e9
    alpha_max = min(alpha_max_true, alpha_max_pred)

    # Grid search over alpha
    best_alpha = None
    best_acc = -1.0
    best_ce = np.inf
    # Start from alpha_req and go up to alpha_max, but also try a bit below if needed.
    # We want to fix the sample, so alpha must be > alpha_req.
    start = max(0.0, alpha_req)
    end = min(alpha_max, alpha_req + 50.0)
    if start > end:
        # Cannot fix within bounds; fall back to bias adjustment
        delta_b = alpha_req
        b_new = b.copy()
        b_new[y_true] += delta_b
        b_new[y_pred] -= delta_b
        # Check if this fixes without breaking others
        L_new = L + delta_b * (np.eye(10)[y_true] - np.eye(10)[y_pred])
        pred_new = np.argmax(L_new, axis=1)
        acc = np.mean(pred_new == y)
        if acc > np.mean(pred == y):
            w["head.bias"] = b_new.astype(np.float32)
        return w

    for alpha in np.linspace(start, end, 201):
        if alpha <= alpha_req:
            continue
        # Update weights
        W_new = W.copy()
        W_new[y_true] += alpha * v
        W_new[y_pred] -= alpha * v
        # Compute new logits
        L_new = L + alpha * c[:, None] * (np.eye(10)[y_true] - np.eye(10)[y_pred])
        pred_new = np.argmax(L_new, axis=1)
        acc = np.mean(pred_new == y)
        # Cross-entropy
        probs = np.exp(L_new - np.max(L_new, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc = acc
            best_ce = ce
            best_alpha = alpha

    if best_alpha is not None and best_acc >= np.mean(pred == y):
        W_new = W.copy()
        W_new[y_true] += best_alpha * v
        W_new[y_pred] -= best_alpha * v
        w["head.weight"] = W_new.astype(np.float32)
        # Optionally update bias if it helps, but weight update is usually enough.
        # We can also try a small bias adjustment to further reduce CE.
        # For simplicity, just return the weight update.
        return w

    # Fallback: bias adjustment
    delta_b = alpha_req
    b_new = b.copy()
    b_new[y_true] += delta_b
    b_new[y_pred] -= delta_b
    L_new = L + delta_b * (np.eye(10)[y_true] - np.eye(10)[y_pred])
    pred_new = np.argmax(L_new, axis=1)
    acc = np.mean(pred_new == y)
    if acc > np.mean(pred == y):
        w["head.bias"] = b_new.astype(np.float32)
    return w
