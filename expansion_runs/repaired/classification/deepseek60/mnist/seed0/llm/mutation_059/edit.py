import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # [2000,784]
    y = f["y"].astype(np.int64)                                 # [2000]
    H_orig = f["features"].astype(np.float32)                   # [2000,128]
    W1 = w["fc1.weight"].copy().astype(np.float32)              # [128,784]
    b1 = w["fc1.bias"].copy().astype(np.float32)                # [128]
    W2 = w["fc2.weight"].copy().astype(np.float32)              # [10,128]
    b2 = w["fc2.bias"].copy().astype(np.float32)                # [10]

    # Identify dead units (all zero on the full feedback pool)
    dead_mask = np.all(H_orig == 0, axis=0)
    dead_idx = np.where(dead_mask)[0]
    n_dead = len(dead_idx)

    # If no dead units, just do a safe ridge refit with prior toward original
    if n_dead == 0:
        # Use original weights unchanged
        return w

    # Compute class means and global mean for template features
    classes = np.arange(10)
    mu = np.zeros((10, X.shape[1]), dtype=np.float32)
    for c in classes:
        mu[c] = X[y == c].mean(axis=0)
    global_mean = X.mean(axis=0)

    # Create new weights for dead units
    W1_new = W1.copy()
    b1_new = b1.copy()
    new_indices = []
    for k, idx in enumerate(dead_idx):
        c = k % 10
        direction = mu[c] - global_mean
        # Template discriminant: positive when closer to class mean than global mean
        W1_new[idx] = direction
        b1_new[idx] = -0.5 * np.dot(direction, direction)
        new_indices.append(idx)

    # Compute new hidden features
    H_new = np.maximum(0.0, X @ W1_new.T + b1_new)  # [2000,128]

    # Scale new units to unit variance (using only the new units)
    for idx in new_indices:
        col = H_new[:, idx]
        std = col.std()
        if std > 1e-8:
            W1_new[idx] /= std
            b1_new[idx] /= std
            H_new[:, idx] /= std

    # Prepare ridge regression with prior toward original fc2
    # Augment H_new with bias column
    H_aug = np.hstack([H_new, np.ones((H_new.shape[0], 1), dtype=np.float32)])  # [2000,129]
    # One-hot targets
    Y_onehot = np.zeros((len(y), 10), dtype=np.float32)
    Y_onehot[np.arange(len(y)), y] = 1.0

    # Prior: original W2 and b2, but zero for new units
    W_aug_prior = np.hstack([W2, b2.reshape(-1, 1)])  # [10,129]
    for idx in new_indices:
        W_aug_prior[:, idx] = 0.0

    # Cross-validation to select lambda
    # We'll use 5-fold CV on the 2000 examples
    n_folds = 5
    indices = np.arange(len(y))
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lambda = 1.0
    best_cv_acc = -1.0

    for lam in lambdas:
        cv_accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            H_train = H_aug[train_idx]
            Y_train = Y_onehot[train_idx]
            H_val = H_aug[val_idx]
            Y_val = Y_onehot[val_idx]
            # Solve ridge with prior
            # W_aug = (H^T H + lam I)^-1 (H^T Y + lam W_prior)
            # Note: H is [N,129], W_aug is [10,129]
            A = H_train.T @ H_train + lam * np.eye(H_aug.shape[1], dtype=np.float32)
            B = H_train.T @ Y_train + lam * W_aug_prior.T  # [129,10]
            W_aug = np.linalg.solve(A, B).T  # [10,129]
            # Predict
            logits_val = H_val @ W_aug.T
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y[val_idx])
            cv_accs.append(acc)
        mean_acc = np.mean(cv_accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Final fit on all data with best lambda
    lam = best_lambda
    A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1], dtype=np.float32)
    B = H_aug.T @ Y_onehot + lam * W_aug_prior.T
    W_aug_final = np.linalg.solve(A, B).T  # [10,129]

    W2_new = W_aug_final[:, :128].astype(np.float32)
    b2_new = W_aug_final[:, 128].astype(np.float32)

    # Clip to finite range
    W1_new = np.clip(W1_new, -1000, 1000).astype(np.float32)
    b1_new = np.clip(b1_new, -1000, 1000).astype(np.float32)
    W2_new = np.clip(W2_new, -1000, 1000).astype(np.float32)
    b2_new = np.clip(b2_new, -1000, 1000).astype(np.float32)

    # Update weights
    w["fc1.weight"] = W1_new
    w["fc1.bias"] = b1_new
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new

    diagnostics = {
        "n_dead_revived": int(n_dead),
        "best_lambda": float(best_lambda),
        "best_cv_acc": float(best_cv_acc),
        "new_fc1_weight_norm": float(np.linalg.norm(W1_new)),
        "new_fc2_weight_norm": float(np.linalg.norm(W2_new)),
        "new_fc2_bias_norm": float(np.linalg.norm(b2_new)),
    }
    return {"weights": w, "diagnostics": diagnostics}
