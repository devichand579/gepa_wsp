import numpy as np

def edit(w, f):
    # Read current weights
    W1 = w["fc1.weight"].astype(np.float64)  # [128,784]
    b1 = w["fc1.bias"].astype(np.float64)    # [128]
    W2 = w["fc2.weight"].astype(np.float64)  # [10,128]
    b2 = w["fc2.bias"].astype(np.float64)    # [10]

    # Feedback data
    X = f["x"].astype(np.float64)            # [2000,1,28,28]
    y = f["y"].astype(np.int64)              # [2000]
    N = X.shape[0]
    X_flat = X.reshape(N, -1)                # [2000,784]

    # Compute current hidden features (ReLU) from raw X using current fc1
    # This is allowed: we are just doing a forward pass, not training fc1.
    Z = X_flat @ W1.T + b1                   # [2000,128]
    H = np.maximum(Z, 0.0)                   # [2000,128]

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with bias
    H_aug = np.concatenate([H, np.ones((N, 1))], axis=1)  # [2000,129]
    W2_aug = np.concatenate([W2, b2[:, None]], axis=1)    # [10,129]

    # Cross-validation to select prior strength lambda
    # We use 5-fold CV on the 2000 examples.
    # For each lambda, we solve the ridge regression with prior centered at W2_aug.
    # The solution is:
    # W_new = (H^T H + lambda I)^{-1} (H^T Y + lambda W0)
    # where W0 is the current W2_aug (10,129).
    # We evaluate accuracy on the validation fold.
    # We do not use any gradient or iterative optimizer; it's a direct linear solve.

    n_folds = 5
    indices = np.arange(N)
    # Deterministic shuffle for reproducibility
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    lambdas = np.logspace(-3, 4, 15)  # 0.001 to 10000
    best_lambda = None
    best_cv_acc = -1.0

    # Precompute H^T H and H^T Y for the full data? For CV we need per-fold.
    # We'll just compute per fold.
    for lam in lambdas:
        cv_acc = 0.0
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            H_train = H_aug[train_idx]
            Y_train = Y[train_idx]
            H_val = H_aug[val_idx]
            Y_val = Y[val_idx]

            # Solve ridge with prior
            # (H^T H + lam I) W^T = H^T Y + lam W0^T
            # W is [10,129], so W^T is [129,10]
            A = H_train.T @ H_train + lam * np.eye(H_train.shape[1])
            B = H_train.T @ Y_train + lam * W2_aug.T
            W_sol = np.linalg.solve(A, B).T  # [10,129]

            # Predict on validation
            logits_val = H_val @ W_sol[:, :-1].T + W_sol[:, -1]
            preds = np.argmax(logits_val, axis=1)
            cv_acc += np.mean(preds == y[val_idx])
        cv_acc /= n_folds
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Refit on all data with best lambda
    lam = best_lambda
    A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1])
    B = H_aug.T @ Y + lam * W2_aug.T
    W_new_aug = np.linalg.solve(A, B).T  # [10,129]

    W2_new = W_new_aug[:, :-1]
    b2_new = W_new_aug[:, -1]

    # Clip to safe range (absolute value <= 1000)
    W2_new = np.clip(W2_new, -1000.0, 1000.0).astype(np.float32)
    b2_new = np.clip(b2_new, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new

    # Diagnostics
    diagnostics = {
        "best_lambda": float(lam),
        "best_cv_acc": float(best_cv_acc),
        "fc2_weight_norm": float(np.linalg.norm(W2_new)),
        "fc2_bias_norm": float(np.linalg.norm(b2_new)),
        "fc2_weight_mean": float(np.mean(W2_new)),
        "fc2_bias_mean": float(np.mean(b2_new)),
    }

    return {"weights": w, "diagnostics": diagnostics}
