import numpy as np

def edit(w, f):
    # Extract full feedback data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    num_classes = 10

    # Global and class means
    mu = X.mean(axis=0)
    class_means = np.zeros((num_classes, X.shape[1]))
    class_counts = np.zeros(num_classes)
    for c in range(num_classes):
        mask = (y == c)
        class_counts[c] = mask.sum()
        if class_counts[c] > 0:
            class_means[c] = X[mask].mean(axis=0)

    # Within-class scatter Sw
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for c in range(num_classes):
        mask = (y == c)
        if class_counts[c] > 0:
            Xc = X[mask] - class_means[c]
            Sw += Xc.T @ Xc
    Sw /= N

    # Between-class scatter Sb
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for c in range(num_classes):
        if class_counts[c] > 0:
            diff = class_means[c] - mu
            Sb += class_counts[c] * np.outer(diff, diff)
    Sb /= N

    # Regularize Sw and solve generalized eigenproblem
    reg = 1e-3 * np.trace(Sw) / X.shape[1]
    Sw_reg = Sw + reg * np.eye(X.shape[1])
    Sw_inv = np.linalg.inv(Sw_reg)
    M = Sw_inv @ Sb

    eigvals, eigvecs = np.linalg.eig(M)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Top 128 LDA directions
    W = eigvecs[:, :128].T  # (128, 784)

    # Center and scale hidden projections
    Z = (X - mu) @ W.T  # (2000, 128)
    std = Z.std(axis=0)
    std = np.maximum(std, 1e-6)
    W_scaled = W / std[:, None]
    bias = -W_scaled @ mu

    # New hidden features with ReLU
    H = X @ W_scaled.T + bias
    H = np.maximum(H, 0.0)

    # One-hot targets
    Y = np.zeros((N, num_classes))
    Y[np.arange(N), y] = 1.0

    # Center features and targets for ridge regression
    H_mean = H.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    H_c = H - H_mean
    Y_c = Y - Y_mean

    # Cross-validate ridge penalty
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_lambda = 1e-2
    best_acc = -1.0

    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])

            H_train = H_c[train_idx]
            Y_train = Y_c[train_idx]
            H_val = H_c[val_idx]

            A = H_train.T @ H_train + lam * np.eye(H.shape[1])
            B = H_train.T @ Y_train
            W_ridge = np.linalg.solve(A, B)

            bias_ridge = Y_mean - H_mean @ W_ridge
            logits_val = H[val_idx] @ W_ridge + bias_ridge
            preds = np.argmax(logits_val, axis=1)
            acc = (preds == y[val_idx]).mean()
            accs.append(acc)

        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Final ridge fit on all data
    A = H_c.T @ H_c + best_lambda * np.eye(H.shape[1])
    B = H_c.T @ Y_c
    W_ridge = np.linalg.solve(A, B)  # (128, 10)
    bias_ridge = Y_mean - H_mean @ W_ridge

    # Update weights
    w["fc1.weight"] = W_scaled.astype(np.float32)
    w["fc1.bias"] = bias.astype(np.float32)
    w["fc2.weight"] = W_ridge.T.astype(np.float32)
    w["fc2.bias"] = bias_ridge.astype(np.float32)

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc1_bias_mean": float(w["fc1.bias"].mean()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
    }
    return {"weights": w, "diagnostics": diagnostics}
