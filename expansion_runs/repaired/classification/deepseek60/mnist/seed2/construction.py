import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (N, 784)
    y = f["y"].astype(np.int64)  # (N,)
    N = X.shape[0]
    num_classes = 10
    hidden_dim = 128

    # One-hot targets
    Y = np.zeros((N, num_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Global mean and centered data
    mu = X.mean(axis=0)
    Xc = X - mu

    # Global PCA (top components)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_global_pca = 8
    V_global = Vt[:n_global_pca].T  # (784, n_global_pca)
    W_global = V_global.T  # (n_global_pca, 784)
    b_global = -W_global @ mu  # (n_global_pca,)

    # Class-specific filters: mean + top PCA per class
    filters = []
    biases = []
    n_pca_per_class = 11  # 1 mean + 11 PCA = 12 per class -> 120 total
    for c in range(num_classes):
        X_c = X[y == c]  # (N_c, 784)
        mu_c = X_c.mean(axis=0)
        Xc_c = X_c - mu_c
        # Mean filter
        filters.append(mu_c)
        biases.append(-0.5 * np.dot(mu_c, mu_c))  # nearest-centroid style bias
        # PCA filters for this class
        if Xc_c.shape[0] > 1:
            Uc, Sc, Vtc = np.linalg.svd(Xc_c, full_matrices=False)
            k = min(n_pca_per_class, Vtc.shape[0])
            V_c = Vtc[:k].T  # (784, k)
            for i in range(k):
                filters.append(V_c[:, i])
                biases.append(-np.dot(V_c[:, i], mu_c))
        else:
            # fallback: zeros
            for i in range(n_pca_per_class):
                filters.append(np.zeros(784))
                biases.append(0.0)

    # Combine class-specific and global PCA filters
    W_class = np.array(filters, dtype=np.float64)  # (120, 784)
    b_class = np.array(biases, dtype=np.float64)  # (120,)

    # Stack all filters to exactly hidden_dim
    W1 = np.vstack([W_class, W_global])  # (128, 784)
    b1 = np.concatenate([b_class, b_global])  # (128,)

    # Compute hidden features
    H = X @ W1.T + b1  # (N, 128)
    H = np.maximum(H, 0.0)  # ReLU

    # Fit output layer via ridge regression
    H_aug = np.hstack([H, np.ones((N, 1))])  # (N, 129)
    lambda_reg = 1e-2
    A = H_aug.T @ H_aug + lambda_reg * np.eye(H_aug.shape[1])
    B = H_aug.T @ Y
    W2_aug = np.linalg.solve(A, B)  # (129, 10)
    W2 = W2_aug[:hidden_dim].T  # (10, 128)
    b2 = W2_aug[hidden_dim]  # (10,)

    # Update weights
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    logits = H @ W2.T + b2
    preds = np.argmax(logits, axis=1)
    train_acc = float(np.mean(preds == y))

    diagnostics = {
        "train_accuracy": train_acc,
        "n_global_pca": n_global_pca,
        "n_pca_per_class": n_pca_per_class,
        "lambda_reg": lambda_reg,
        "fc1_weight_mean": float(np.mean(W1)),
        "fc1_weight_std": float(np.std(W1)),
        "fc2_weight_mean": float(np.mean(W2)),
        "fc2_weight_std": float(np.std(W2)),
    }

    return {"weights": w, "diagnostics": diagnostics}
