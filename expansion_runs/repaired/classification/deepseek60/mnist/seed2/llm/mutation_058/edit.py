import numpy as np

def edit(w, f):
    # Extract full feedback pool
    H = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = H.shape
    C = 10

    # Current fc2 parameters
    W_cur = w["fc2.weight"].astype(np.float64)  # (10, 128)
    b_cur = w["fc2.bias"].astype(np.float64)    # (10,)

    # Helper: compute accuracy and cross-entropy for a given (W, b)
    def evaluate(W, b):
        logits = H @ W.T + b
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == y)
        # stable softmax cross-entropy
        logits_shift = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_shift)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        return acc, ce

    acc_cur, ce_cur = evaluate(W_cur, b_cur)

    # Candidate 1: Ridge regression on raw features
    best_W, best_b = W_cur.copy(), b_cur.copy()
    best_acc, best_ce = acc_cur, ce_cur

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Try a range of ridge penalties
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        # Solve (H^T H + alpha I) W^T = H^T Y  => W = Y^T H (H^T H + alpha I)^{-1}
        # More stable: use lstsq on augmented system or solve normal equations
        A = H.T @ H + alpha * np.eye(D)
        B = H.T @ Y  # (D, C)
        W_ridge = np.linalg.solve(A, B).T  # (C, D)
        b_ridge = np.mean(Y - H @ W_ridge.T, axis=0)  # (C,)
        acc, ce = evaluate(W_ridge, b_ridge)
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc, best_ce = acc, ce
            best_W, best_b = W_ridge, b_ridge

    # Candidate 2: Ridge on standardized features (often better)
    mean_H = np.mean(H, axis=0)
    std_H = np.std(H, axis=0) + 1e-6
    Z = (H - mean_H) / std_H

    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        A = Z.T @ Z + alpha * np.eye(D)
        B = Z.T @ Y
        W_z = np.linalg.solve(A, B).T  # (C, D)
        b_z = np.mean(Y - Z @ W_z.T, axis=0)
        # Convert back to original feature space
        W_orig = W_z / std_H
        b_orig = b_z - W_orig @ mean_H
        acc, ce = evaluate(W_orig, b_orig)
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc, best_ce = acc, ce
            best_W, best_b = W_orig, b_orig

    # Candidate 3: Diagonal LDA (naive Bayes) - very robust
    # Compute class means and pooled variance
    means = np.zeros((C, D))
    vars_ = np.zeros((C, D))
    counts = np.zeros(C)
    for c in range(C):
        mask = (y == c)
        counts[c] = np.sum(mask)
        if counts[c] > 1:
            means[c] = np.mean(H[mask], axis=0)
            vars_[c] = np.var(H[mask], axis=0)
        else:
            means[c] = np.mean(H, axis=0)
            vars_[c] = np.var(H, axis=0)
    # Pooled variance
    pooled_var = np.sum((counts[:, None] - 1) * vars_, axis=0) / (N - C)
    pooled_var = np.maximum(pooled_var, 1e-6)
    W_diag = means / pooled_var  # (C, D)
    b_diag = -0.5 * np.sum(means**2 / pooled_var, axis=1) + np.log(counts / N)
    acc, ce = evaluate(W_diag, b_diag)
    if acc > best_acc or (acc == best_acc and ce < best_ce):
        best_acc, best_ce = acc, ce
        best_W, best_b = W_diag, b_diag

    # Candidate 4: LDA with shrinkage
    mu = np.mean(H, axis=0)
    Sw = np.zeros((D, D))
    for c in range(C):
        mask = (y == c)
        if np.sum(mask) > 1:
            Hc = H[mask] - means[c]
            Sw += Hc.T @ Hc
    Sw /= N
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        Sw_shrunk = Sw + alpha * np.trace(Sw) / D * np.eye(D)
        try:
            Sw_inv = np.linalg.inv(Sw_shrunk)
        except np.linalg.LinAlgError:
            continue
        W_lda = (Sw_inv @ (means - mu).T).T  # (C, D)
        b_lda = -W_lda @ mu + np.log(counts / N)
        acc, ce = evaluate(W_lda, b_lda)
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc, best_ce = acc, ce
            best_W, best_b = W_lda, b_lda

    # Install the best candidate if it improves or matches current
    if best_acc > acc_cur or (best_acc == acc_cur and best_ce < ce_cur):
        w["fc2.weight"] = best_W.astype(np.float32)
        w["fc2.bias"] = best_b.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "acc_current": float(acc_cur),
        "ce_current": float(ce_cur),
        "acc_best": float(best_acc),
        "ce_best": float(best_ce),
        "chosen": "best" if (best_acc > acc_cur or (best_acc == acc_cur and best_ce < ce_cur)) else "current",
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
