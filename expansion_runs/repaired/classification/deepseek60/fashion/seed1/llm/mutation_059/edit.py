import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # Compute class means and global mean
    means = np.zeros((C, D))
    counts = np.zeros(C)
    for c in range(C):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            means[c] = X[mask].mean(axis=0)
    global_mean = X.mean(axis=0)

    # Compute within-class scatter matrix S_w
    XtX = X.T @ X
    S_w = XtX - N * np.outer(global_mean, global_mean)
    for c in range(C):
        diff = means[c] - global_mean
        S_w -= counts[c] * np.outer(diff, diff)

    # Try several ridge values and pick the best LDA solution
    best_acc = -1.0
    best_ce = 1e9
    best_W = None
    best_b = None

    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        S_w_reg = S_w + lam * np.eye(D)
        try:
            W_lda = np.linalg.solve(S_w_reg, (means - global_mean).T).T  # (C, D)
        except np.linalg.LinAlgError:
            continue
        b_lda = -W_lda @ global_mean
        logits = X @ W_lda.T + b_lda
        pred = logits.argmax(axis=1)
        acc = (pred == y).mean()
        # stable cross-entropy
        logits_stable = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_stable)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        ce = -np.log(probs[np.arange(N), y] + 1e-12).mean()
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc = acc
            best_ce = ce
            best_W = W_lda
            best_b = b_lda

    # If no LDA solution worked, fall back to current head
    if best_W is None:
        best_W = w["head.weight"].astype(np.float64)
        best_b = w["head.bias"].astype(np.float64)

    # One-step closed-form correction for misclassified samples
    # We want to enforce margin >= 1 for all training samples.
    # Build constraints for currently misclassified samples.
    logits = X @ best_W.T + best_b
    pred = logits.argmax(axis=1)
    mis = np.where(pred != y)[0]
    if len(mis) > 0:
        # Build constraint matrix A and target vector
        # Each constraint: (W_c - W_j)·x_i + (b_c - b_j) = 1
        # where c = y[i], j = pred[i]
        A_rows = []
        for i in mis:
            c = y[i]
            j = pred[i]
            row = np.zeros(C * D + C)
            # W_c coefficients
            row[c*D:(c+1)*D] = X[i]
            # -W_j coefficients
            row[j*D:(j+1)*D] = -X[i]
            # bias c
            row[C*D + c] = 1.0
            # bias j
            row[C*D + j] = -1.0
            A_rows.append(row)
        A = np.array(A_rows)
        target = np.ones(len(mis))
        # Current vector
        w0 = np.concatenate([best_W.flatten(), best_b])
        # Projection onto affine subspace A w = target
        # w = w0 + A^T (A A^T)^{-1} (target - A w0)
        AAT = A @ A.T
        # Add small ridge for numerical stability
        AAT += 1e-8 * np.eye(len(mis))
        delta = target - A @ w0
        w_new = w0 + A.T @ np.linalg.solve(AAT, delta)
        best_W = w_new[:C*D].reshape(C, D)
        best_b = w_new[C*D:]

    # Scale weights and biases to max absolute value 1000
    max_abs = max(np.abs(best_W).max(), np.abs(best_b).max())
    if max_abs > 0:
        scale = 1000.0 / max_abs
        best_W *= scale
        best_b *= scale

    # Ensure float32 and within bounds
    best_W = np.clip(best_W, -1000.0, 1000.0).astype(np.float32)
    best_b = np.clip(best_b, -1000.0, 1000.0).astype(np.float32)

    # Update the head weights
    w["head.weight"] = best_W
    w["head.bias"] = best_b

    # Diagnostics
    logits_final = X @ best_W.T + best_b
    pred_final = logits_final.argmax(axis=1)
    acc_final = (pred_final == y).mean()
    logits_stable = logits_final - logits_final.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    probs_final = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce_final = -np.log(probs_final[np.arange(N), y] + 1e-12).mean()

    diagnostics = {
        "final_accuracy": float(acc_final),
        "final_cross_entropy": float(ce_final),
        "max_abs_weight": float(np.abs(best_W).max()),
        "max_abs_bias": float(np.abs(best_b).max()),
        "num_misclassified": int((pred_final != y).sum())
    }

    return {"weights": w, "diagnostics": diagnostics}
