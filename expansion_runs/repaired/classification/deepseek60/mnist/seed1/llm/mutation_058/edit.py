import numpy as np

def edit(w, f):
    # Extract current fc1 parameters
    W1 = w["fc1.weight"].astype(np.float32)  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float32)    # (128,)

    # Full feedback pool
    x = f["x"].astype(np.float32)            # (2000, 1, 28, 28)
    y = f["y"].astype(np.int64)              # (2000,)
    H = f["features"].astype(np.float32)     # (2000, 128)

    # Helper: compute ReLU features for a batch of images
    def compute_features(images):
        N = images.shape[0]
        flat = images.reshape(N, -1)          # (N, 784)
        return np.maximum(0.0, flat @ W1.T + b1)  # (N, 128)

    # Translation augmentation: shifts of -1, 0, 1 in both dimensions
    shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    H_aug_list = []
    y_aug_list = []
    for dy, dx in shifts:
        if dy == 0 and dx == 0:
            H_shift = H
        else:
            # Shift images with zero padding
            shifted = np.zeros_like(x)
            if dy >= 0:
                src_y = slice(0, 28 - dy)
                dst_y = slice(dy, 28)
            else:
                src_y = slice(-dy, 28)
                dst_y = slice(0, 28 + dy)
            if dx >= 0:
                src_x = slice(0, 28 - dx)
                dst_x = slice(dx, 28)
            else:
                src_x = slice(-dx, 28)
                dst_x = slice(0, 28 + dx)
            shifted[:, :, dst_y, dst_x] = x[:, :, src_y, src_x]
            H_shift = compute_features(shifted)
        H_aug_list.append(H_shift)
        y_aug_list.append(y)
    H_aug = np.vstack(H_aug_list)            # (18000, 128)
    y_aug = np.concatenate(y_aug_list)       # (18000,)

    # One-hot targets
    Y_aug = np.zeros((H_aug.shape[0], 10), dtype=np.float32)
    Y_aug[np.arange(H_aug.shape[0]), y_aug] = 1.0

    # Add bias column
    H_aug_b = np.hstack([H_aug, np.ones((H_aug.shape[0], 1), dtype=np.float32)])  # (18000, 129)

    # 5-fold cross-validation to select ridge lambda
    n = H_aug.shape[0]
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    best_lambda = 1.0
    best_cv_acc = -1.0

    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_tr = H_aug_b[train_idx]
            Y_tr = Y_aug[train_idx]
            H_va = H_aug_b[val_idx]
            Y_va = Y_aug[val_idx]

            # Ridge solution: (H^T H + lam I) W = H^T Y
            A = H_tr.T @ H_tr + lam * np.eye(H_tr.shape[1], dtype=np.float32)
            B = H_tr.T @ Y_tr
            W = np.linalg.solve(A, B)  # (129, 10)

            logits_va = H_va @ W
            preds = np.argmax(logits_va, axis=1)
            acc = np.mean(preds == np.argmax(Y_va, axis=1))
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Final fit on all augmented data with best lambda
    A = H_aug_b.T @ H_aug_b + best_lambda * np.eye(H_aug_b.shape[1], dtype=np.float32)
    B = H_aug_b.T @ Y_aug
    W_final = np.linalg.solve(A, B)  # (129, 10)

    # Extract fc2 weights and bias
    new_fc2_weight = W_final[:128, :].T.astype(np.float32)  # (10, 128)
    new_fc2_bias = W_final[128, :].astype(np.float32)       # (10,)

    # Update weights
    w["fc2.weight"] = new_fc2_weight
    w["fc2.bias"] = new_fc2_bias

    # Diagnostics
    # Evaluate on original full pool
    H_b = np.hstack([H, np.ones((H.shape[0], 1), dtype=np.float32)])
    logits_new = H_b @ W_final
    preds_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(preds_new == y)

    # Cross-entropy
    probs_new = np.exp(logits_new - np.max(logits_new, axis=1, keepdims=True))
    probs_new /= np.sum(probs_new, axis=1, keepdims=True)
    ce_new = -np.mean(np.log(probs_new[np.arange(len(y)), y] + 1e-12))

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "train_accuracy": float(acc_new),
        "train_cross_entropy": float(ce_new),
        "fc2_weight_mean": float(np.mean(new_fc2_weight)),
        "fc2_weight_std": float(np.std(new_fc2_weight)),
        "fc2_bias_mean": float(np.mean(new_fc2_bias)),
        "fc2_bias_std": float(np.std(new_fc2_bias)),
        "augmented_size": int(H_aug.shape[0]),
    }

    return {"weights": w, "diagnostics": diagnostics}
