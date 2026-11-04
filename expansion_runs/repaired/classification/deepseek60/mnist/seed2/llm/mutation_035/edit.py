import numpy as np

def edit(w, f):
    # Read current parameters
    fc1_w = w["fc1.weight"]
    fc1_b = w["fc1.bias"]
    fc2_w = w["fc2.weight"]
    fc2_b = w["fc2.bias"]

    # Feedback data
    L = f["logits"].astype(np.float64)          # (N, 10)
    y = f["y"].astype(np.int64)                 # (N,)
    N, C = L.shape

    # Current accuracy on feedback pool
    current_pred = np.argmax(L, axis=1)
    current_acc = float(np.mean(current_pred == y))

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment logits with a bias column
    L_aug = np.hstack([L, np.ones((N, 1), dtype=np.float64)])  # (N, 11)

    best_acc = current_acc
    best_fc2_w = fc2_w.copy()
    best_fc2_b = fc2_b.copy()
    best_method = "current"
    best_alpha = None

    # Try a small set of closed-form ridge solutions on the logits.
    # This is not an iterative optimizer; it is a finite set of normal-equation solves.
    for alpha in [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        # Solve (L_aug^T L_aug + alpha I) B = L_aug^T Y
        A_mat = L_aug.T @ L_aug
        if alpha > 0:
            A_mat = A_mat + alpha * np.eye(A_mat.shape[0], dtype=np.float64)
        B = np.linalg.solve(A_mat, L_aug.T @ Y)   # (11, 10)

        # New logits = L_aug @ B
        new_logits = L_aug @ B
        new_pred = np.argmax(new_logits, axis=1)
        new_acc = float(np.mean(new_pred == y))

        if new_acc > best_acc:
            best_acc = new_acc
            # Compose with current fc2: new_logits = (L @ A.T + c)
            # L = features @ fc2_w.T + fc2_b
            # new_logits = (features @ fc2_w.T + fc2_b) @ A.T + c
            #            = features @ (A @ fc2_w).T + (A @ fc2_b + c)
            A = B[:C, :].T          # (10, 10)
            c = B[C, :]             # (10,)
            best_fc2_w = (A @ fc2_w).astype(np.float32)
            best_fc2_b = (A @ fc2_b + c).astype(np.float32)
            best_method = "ridge_logits"
            best_alpha = alpha

    # Optional: also try LDA on the logits as a second closed-form candidate.
    # Compute class means and pooled covariance.
    means = np.zeros((C, C), dtype=np.float64)
    for k in range(C):
        means[k] = L[y == k].mean(axis=0)
    # Pooled within-class covariance
    S = np.zeros((C, C), dtype=np.float64)
    for k in range(C):
        diff = L[y == k] - means[k]
        S += diff.T @ diff
    S /= N
    # Small ridge for stability
    S_reg = S + 1e-3 * np.eye(C, dtype=np.float64)
    S_inv = np.linalg.inv(S_reg)
    # LDA weights: w_k = S_inv @ mu_k
    W_lda = (S_inv @ means.T).T          # (10, 10)
    # Bias: -0.5 * mu_k^T S_inv mu_k + log prior (priors equal)
    b_lda = -0.5 * np.sum(means * (S_inv @ means.T).T, axis=1)
    new_logits_lda = L @ W_lda.T + b_lda
    new_pred_lda = np.argmax(new_logits_lda, axis=1)
    new_acc_lda = float(np.mean(new_pred_lda == y))

    if new_acc_lda > best_acc:
        best_acc = new_acc_lda
        best_fc2_w = (W_lda @ fc2_w).astype(np.float32)
        best_fc2_b = (W_lda @ fc2_b + b_lda).astype(np.float32)
        best_method = "lda_logits"
        best_alpha = None

    # Build output weights
    w_out = dict(w)
    w_out["fc1.weight"] = fc1_w
    w_out["fc1.bias"] = fc1_b
    w_out["fc2.weight"] = best_fc2_w
    w_out["fc2.bias"] = best_fc2_b

    diagnostics = {
        "current_acc": current_acc,
        "best_acc": best_acc,
        "best_method": best_method,
        "best_alpha": best_alpha,
        "fc2_weight_norm": float(np.linalg.norm(best_fc2_w)),
        "fc2_bias_norm": float(np.linalg.norm(best_fc2_b)),
        "fc2_weight_min": float(best_fc2_w.min()),
        "fc2_weight_max": float(best_fc2_w.max()),
        "fc2_bias_min": float(best_fc2_b.min()),
        "fc2_bias_max": float(best_fc2_b.max()),
    }

    return {"weights": w_out, "diagnostics": diagnostics}
