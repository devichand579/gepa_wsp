import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Center features and targets for ridge with unregularized bias
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Precompute Gram matrix and cross term
    G = Xc.T @ Xc          # (D, D)
    B = Xc.T @ Yc          # (D, C)

    # Candidate regularization strengths (lambda)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]

    # 5-fold cross-validation to choose lambda
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    n_folds = 5
    fold_sizes = [N // n_folds] * n_folds
    for i in range(N % n_folds):
        fold_sizes[i] += 1
    fold_indices = []
    start = 0
    for fs in fold_sizes:
        fold_indices.append(idx[start:start+fs])
        start += fs

    best_lambda = None
    best_cv_acc = -1.0

    for lam in lambdas:
        accs = []
        for k in range(n_folds):
            val_idx = fold_indices[k]
            train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != k])
            X_tr = X[train_idx]
            Y_tr = Y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

            # Center using training statistics
            X_tr_mean = X_tr.mean(axis=0)
            Y_tr_mean = Y_tr.mean(axis=0)
            X_tr_c = X_tr - X_tr_mean
            Y_tr_c = Y_tr - Y_tr_mean

            G_tr = X_tr_c.T @ X_tr_c
            B_tr = X_tr_c.T @ Y_tr_c

            # Ridge solution: W = (G + lam*I)^{-1} B
            # Use solve for stability
            A = G_tr + lam * np.eye(D)
            W = np.linalg.solve(A, B_tr)  # (D, C)
            b = Y_tr_mean - X_tr_mean @ W

            # Predict on validation
            logits = X_val @ W + b
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_val)
            accs.append(acc)

        mean_acc = np.mean(accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # If no lambda improved over a simple default, use a moderate value
    if best_lambda is None:
        best_lambda = 1.0

    # Train final model on all data with chosen lambda
    A = G + best_lambda * np.eye(D)
    W = np.linalg.solve(A, B)  # (D, C)
    b = Y_mean - X_mean @ W

    # head.weight is [out, in] = [10, 1568], so transpose W
    head_w = W.T.astype(np.float32)  # (10, 1568)
    head_b = b.astype(np.float32)    # (10,)

    # Scale weights and bias to maximize magnitude while keeping argmax unchanged.
    # Find max absolute value in head_w and head_b, then scale so max abs is 1000.
    max_abs = max(np.max(np.abs(head_w)), np.max(np.abs(head_b)))
    if max_abs > 0:
        scale = 1000.0 / max_abs
        head_w *= scale
        head_b *= scale

    # Ensure finite and within bounds
    head_w = np.clip(head_w, -1000.0, 1000.0).astype(np.float32)
    head_b = np.clip(head_b, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["head.weight"] = head_w
    w["head.bias"] = head_b

    diagnostics = {
        "chosen_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "max_abs_head_weight": float(np.max(np.abs(head_w))),
        "max_abs_head_bias": float(np.max(np.abs(head_b))),
    }

    return {"weights": w, "diagnostics": diagnostics}
