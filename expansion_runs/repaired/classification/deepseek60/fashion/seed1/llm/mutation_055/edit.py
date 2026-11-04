import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    num_classes = 10

    # --- LDA with ridge regularization ---
    classes = np.arange(num_classes)
    mean_c = np.zeros((num_classes, D))
    for c in classes:
        mean_c[c] = X[y == c].mean(axis=0)

    # Within-class scatter matrix
    S = np.zeros((D, D))
    for c in classes:
        Xc = X[y == c] - mean_c[c]
        S += Xc.T @ Xc

    # Ridge regularization
    lam = 1e-3 * np.trace(S) / D
    S_reg = S + lam * np.eye(D)

    # LDA weights and bias
    W_lda = mean_c @ np.linalg.inv(S_reg)          # (10, 1568)
    b_lda = -0.5 * np.sum(mean_c * W_lda, axis=1)  # (10,)

    # Evaluate LDA on full pool
    logits_lda = X @ W_lda.T + b_lda
    pred_lda = np.argmax(logits_lda, axis=1)
    acc_lda = np.mean(pred_lda == y)

    # --- Ridge regression with one-hot targets ---
    Y_onehot = np.zeros((N, num_classes))
    Y_onehot[np.arange(N), y] = 1.0
    X_aug = np.hstack([X, np.ones((N, 1))])
    lam_reg = 1e-3
    A = X_aug.T @ X_aug + lam_reg * np.eye(D + 1)
    B = X_aug.T @ Y_onehot
    W_aug = np.linalg.solve(A, B)
    W_ridge = W_aug[:D, :].T          # (10, 1568)
    b_ridge = W_aug[D, :]             # (10,)

    logits_ridge = X @ W_ridge.T + b_ridge
    pred_ridge = np.argmax(logits_ridge, axis=1)
    acc_ridge = np.mean(pred_ridge == y)

    # Choose the better classifier
    if acc_lda >= acc_ridge:
        W = W_lda
        b = b_lda
        acc = acc_lda
        logits = logits_lda
        pred = pred_lda
    else:
        W = W_ridge
        b = b_ridge
        acc = acc_ridge
        logits = logits_ridge
        pred = pred_ridge

    # --- Targeted least-squares correction for misclassified samples ---
    mis_idx = np.where(pred != y)[0]
    if len(mis_idx) > 0:
        # We want to adjust W and b so that for each misclassified sample i,
        # the correct class score exceeds the wrong class score by a margin.
        # We can solve a small least-squares problem on the misclassified samples.
        # Let delta_W (10, D) and delta_b (10,) be adjustments.
        # For each misclassified sample i, we want:
        #   X_i @ (W + delta_W)_{y_i} + (b + delta_b)_{y_i} > X_i @ (W + delta_W)_c + (b + delta_b)_c + 1
        # for all c != y_i. This is a set of linear inequalities.
        # Instead of solving inequalities, we can use a least-squares correction:
        # We want to increase the score of the correct class and decrease the score of the wrong class.
        # A simple closed-form approach: solve for delta_W, delta_b that minimizes
        # || X_mis @ delta_W.T + delta_b - target ||^2, where target is a matrix that
        # pushes the correct class up and wrong classes down.
        X_mis = X[mis_idx]  # (M, D)
        y_mis = y[mis_idx]  # (M,)
        M = len(mis_idx)

        # Target: for each misclassified sample, we want the correct class to have a high score,
        # and the current highest wrong class to have a low score.
        # We can set target such that correct class = current_score + 1, wrong class = current_score - 1.
        # But we can simply use a one-hot target with a large margin.
        # Let's create a target matrix T (M, 10) where T[i, y_mis[i]] = 1, others = -1.
        T = -np.ones((M, num_classes))
        T[np.arange(M), y_mis] = 1.0

        # We want to find delta_W, delta_b such that X_mis @ delta_W.T + delta_b ≈ T * scale
        # We can solve the least-squares problem: [X_mis, ones] @ [delta_W.T; delta_b] = T
        X_mis_aug = np.hstack([X_mis, np.ones((M, 1))])
        # Solve for delta_W_aug (D+1, 10)
        # Use ridge regularization to avoid singularity
        lam_corr = 1e-3
        A_corr = X_mis_aug.T @ X_mis_aug + lam_corr * np.eye(D + 1)
        B_corr = X_mis_aug.T @ T
        delta_W_aug = np.linalg.solve(A_corr, B_corr)
        delta_W = delta_W_aug[:D, :].T  # (10, D)
        delta_b = delta_W_aug[D, :]     # (10,)

        # Apply correction with a small step size to avoid overshooting
        # We can scale the correction to ensure it doesn't ruin correct samples.
        # A simple approach: add the correction directly, then check accuracy.
        # If accuracy improves, keep it; otherwise, scale down.
        step = 1.0
        W_new = W + step * delta_W
        b_new = b + step * delta_b
        logits_new = X @ W_new.T + b_new
        pred_new = np.argmax(logits_new, axis=1)
        acc_new = np.mean(pred_new == y)

        # If the correction improves accuracy, use it. Otherwise, try smaller steps.
        if acc_new > acc:
            W = W_new
            b = b_new
            acc = acc_new
            logits = logits_new
            pred = pred_new
        else:
            # Try a few step sizes
            for step in [0.5, 0.2, 0.1, 0.05, 0.01]:
                W_new = W + step * delta_W
                b_new = b + step * delta_b
                logits_new = X @ W_new.T + b_new
                pred_new = np.argmax(logits_new, axis=1)
                acc_new = np.mean(pred_new == y)
                if acc_new > acc:
                    W = W_new
                    b = b_new
                    acc = acc_new
                    logits = logits_new
                    pred = pred_new
                    break

    # --- Final scaling to max abs 1000 ---
    # Only scale if we have perfect training accuracy, to avoid huge cross-entropy on errors.
    if acc == 1.0:
        max_abs = max(np.abs(W).max(), np.abs(b).max())
        if max_abs > 0:
            scale = 1000.0 / max_abs
            W = W * scale
            b = b * scale
    else:
        # If not perfect, scale to a moderate value to keep cross-entropy reasonable.
        # But we can still scale to 1000 if we want, but it might hurt if there are errors.
        # Let's scale to a value that keeps max abs around 100.
        max_abs = max(np.abs(W).max(), np.abs(b).max())
        if max_abs > 0:
            scale = 100.0 / max_abs
            W = W * scale
            b = b * scale

    # Convert to float32 and ensure within bounds
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)

    # Update the head weights and biases
    w["head.weight"] = W
    w["head.bias"] = b

    # Diagnostics
    diagnostics = {
        "lda_accuracy": float(acc_lda),
        "ridge_accuracy": float(acc_ridge),
        "final_accuracy": float(acc),
        "num_misclassified": int(np.sum(pred != y)),
        "max_abs_weight": float(np.abs(W).max()),
        "max_abs_bias": float(np.abs(b).max()),
    }

    return {"weights": w, "diagnostics": diagnostics}
