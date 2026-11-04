import numpy as np

def edit(w, f):
    # Extract current weights
    W1 = w["fc1.weight"].astype(np.float64)
    b1 = w["fc1.bias"].astype(np.float64)
    W2 = w["fc2.weight"].astype(np.float64)
    b2 = w["fc2.bias"].astype(np.float64)

    # Feedback data
    X = f["x"].astype(np.float64)          # [2000,1,28,28]
    y = f["y"].astype(np.int64)            # [2000]
    H = f["features"].astype(np.float64)   # [2000,128] current hidden features

    N, D = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Add bias column to features
    Hb = np.concatenate([H, np.ones((N, 1), dtype=np.float64)], axis=1)  # [N,129]

    # Ridge regression with several lambdas, selected by 5-fold CV on the feedback pool
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_lambda = None
    best_cv_acc = -1.0

    # 5-fold cross-validation
    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for lam in lambdas:
        accs = []
        for k in range(n_folds):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            Hb_tr = Hb[tr_idx]
            Y_tr = Y[tr_idx]
            Hb_val = Hb[val_idx]
            Y_val = Y[val_idx]

            # Solve (Hb^T Hb + lam I) W = Hb^T Y
            A = Hb_tr.T @ Hb_tr + lam * np.eye(Hb.shape[1])
            B = Hb_tr.T @ Y_tr
            W_fit = np.linalg.solve(A, B)  # [129,10]

            # Predict
            logits_val = Hb_val @ W_fit
            pred = np.argmax(logits_val, axis=1)
            acc = np.mean(pred == y[val_idx])
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Fit final classifier on all data with best lambda
    A = Hb.T @ Hb + best_lambda * np.eye(Hb.shape[1])
    B = Hb.T @ Y
    W_fit = np.linalg.solve(A, B)  # [129,10]

    # W_fit is [129,10]; we need fc2.weight [10,128] and fc2.bias [10]
    new_W2 = W_fit[:128, :].T   # [10,128]
    new_b2 = W_fit[128, :]      # [10]

    # Optional: shrink toward current fc2 to preserve calibration
    # Use a small shrinkage factor based on CV improvement
    # If CV accuracy is not better than current, keep current weights.
    # Compute current CV accuracy for comparison
    current_cv_accs = []
    for k in range(n_folds):
        val_idx = folds[k]
        tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        # Fit current fc2 on training fold? Actually we just evaluate current fc2 on validation fold.
        # But current fc2 was trained on all data, so evaluating on validation is optimistic.
        # Instead, we can just use the best_lambda solution.
        pass

    # To be safe, we can blend the new weights with the old ones.
    # A simple blend: new = alpha * new + (1-alpha) * old, with alpha chosen by CV.
    # We can try a few alphas and pick the best by CV.
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_alpha = 1.0
    best_blend_cv = -1.0

    for alpha in alphas:
        accs = []
        for k in range(n_folds):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            Hb_tr = Hb[tr_idx]
            Y_tr = Y[tr_idx]
            Hb_val = Hb[val_idx]
            Y_val = Y[val_idx]

            # Fit on training fold
            A = Hb_tr.T @ Hb_tr + best_lambda * np.eye(Hb.shape[1])
            B = Hb_tr.T @ Y_tr
            W_fit_tr = np.linalg.solve(A, B)  # [129,10]

            # Blend with current weights on training fold? Actually we blend the final weights.
            # For CV, we need to blend the fitted weights with the current weights.
            # But current weights are fixed. We can compute the blended weights on the training fold.
            # However, the current weights were trained on all data, so blending on training fold is not perfectly clean.
            # For simplicity, we just use the fitted weights without blending in CV, and then apply blending at the end.
            # Let's just use alpha=1.0 for CV.
            logits_val = Hb_val @ W_fit_tr
            pred = np.argmax(logits_val, axis=1)
            acc = np.mean(pred == y[val_idx])
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_blend_cv:
            best_blend_cv = mean_acc
            best_alpha = alpha

    # Actually, let's just use the best_lambda solution without blending, as blending might not help.
    # But we can do a simple shrinkage: new_W2 = new_W2, new_b2 = new_b2.
    # To be conservative, we can check if the new classifier improves CV accuracy over the current one.
    # Compute current CV accuracy using current fc2 on the same folds.
    current_accs = []
    for k in range(n_folds):
        val_idx = folds[k]
        Hb_val = Hb[val_idx]
        logits_val = Hb_val[:, :128] @ W2.T + b2
        pred = np.argmax(logits_val, axis=1)
        acc = np.mean(pred == y[val_idx])
        current_accs.append(acc)
    current_cv_acc = np.mean(current_accs)

    if best_cv_acc <= current_cv_acc:
        # No improvement, keep current weights
        new_W2 = W2
        new_b2 = b2
    else:
        # Use the fitted weights
        new_W2 = W_fit[:128, :].T
        new_b2 = W_fit[128, :]

    # Ensure float32 and finite
    new_W2 = np.clip(new_W2, -1000, 1000).astype(np.float32)
    new_b2 = np.clip(new_b2, -1000, 1000).astype(np.float32)

    # Update weights
    w["fc2.weight"] = new_W2
    w["fc2.bias"] = new_b2

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "best_cv_accuracy": float(best_cv_acc),
        "current_cv_accuracy": float(current_cv_acc),
        "fc2_weight_norm": float(np.linalg.norm(new_W2)),
        "fc2_bias_norm": float(np.linalg.norm(new_b2)),
        "used_new_weights": bool(best_cv_acc > current_cv_acc)
    }

    return {"weights": w, "diagnostics": diagnostics}
