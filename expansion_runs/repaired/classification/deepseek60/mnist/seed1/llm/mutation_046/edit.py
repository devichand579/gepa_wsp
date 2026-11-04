import numpy as np
import math

def edit(w, f):
    # Extract full training pool
    x = f["x"]                     # [2000, 1, 28, 28]
    y = f["y"]                     # [2000]
    N = x.shape[0]
    x_flat = x.reshape(N, -1)      # [2000, 784]

    # --- Build new fc1 filters ---
    # Class means
    classes = np.unique(y)
    n_classes = len(classes)
    means = np.zeros((n_classes, 784), dtype=np.float32)
    for i, c in enumerate(classes):
        means[i] = x_flat[y == c].mean(axis=0)
    global_mean = x_flat.mean(axis=0)

    filters = []
    # 1) Class means minus global mean
    for i in range(n_classes):
        filters.append(means[i] - global_mean)
    # 2) Pairwise class-mean differences
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            filters.append(means[i] - means[j])
    # 3) PCA components of centered data
    x_centered = x_flat - global_mean
    # Use SVD for PCA (economy size)
    U, S, Vt = np.linalg.svd(x_centered, full_matrices=False)
    # Vt rows are principal directions
    n_pca = 128 - len(filters)
    if n_pca > 0:
        for i in range(min(n_pca, Vt.shape[0])):
            filters.append(Vt[i])
    # If still short, pad with random orthogonal directions (should not happen)
    while len(filters) < 128:
        r = np.random.randn(784).astype(np.float32)
        r -= r.dot(global_mean) * global_mean / (global_mean.dot(global_mean) + 1e-8)
        filters.append(r)
    filters = np.array(filters[:128], dtype=np.float32)  # [128, 784]

    # Normalize each filter to unit L2 norm
    norms = np.linalg.norm(filters, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    filters = filters / norms

    # Set fc1 weights and bias (center the data)
    w["fc1.weight"] = filters
    w["fc1.bias"] = -filters.dot(global_mean).astype(np.float32)

    # --- Compute hidden features for the full pool ---
    H = x_flat.dot(filters.T) + w["fc1.bias"]  # [2000, 128]
    H = np.maximum(H, 0.0)                     # ReLU

    # --- Fit fc2 via ridge regression ---
    # Add bias column
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])  # [2000, 129]
    # One-hot targets
    Y = np.zeros((N, n_classes), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Ridge regression: W = (H_aug^T H_aug + lambda I)^{-1} H_aug^T Y
    lam = 1e-4
    A = H_aug.T.dot(H_aug) + lam * np.eye(H_aug.shape[1], dtype=np.float32)
    B = H_aug.T.dot(Y)
    W = np.linalg.solve(A, B)  # [129, 10]

    fc2_weight = W[:128].T.astype(np.float32)  # [10, 128]
    fc2_bias = W[128].astype(np.float32)       # [10]

    # --- Calibrate logits with one Newton step on scale alpha ---
    # Current logits Z = H_aug @ W
    Z = H_aug.dot(W)  # [2000, 10]
    # Compute CE and derivatives w.r.t. alpha (scaling Z)
    # We want to find alpha such that CE(alpha*Z) is minimized.
    # Use one Newton step from alpha=1.
    # p = softmax(Z)
    Z_max = Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z - Z_max)
    p = expZ / expZ.sum(axis=1, keepdims=True)  # [2000, 10]

    # Gradient of CE w.r.t. alpha at alpha=1:
    # dCE/dalpha = -1/N sum_i ( Z_i,y_i - sum_c p_i,c Z_i,c )
    # Let's compute per-sample: Z_y - sum_c p_c Z_c
    Z_y = Z[np.arange(N), y]
    Z_bar = (p * Z).sum(axis=1)
    grad = -(Z_y - Z_bar).mean()

    # Second derivative:
    # d2CE/dalpha2 = 1/N sum_i ( sum_c p_c Z_c^2 - (sum_c p_c Z_c)^2 )
    Z2_bar = (p * Z * Z).sum(axis=1)
    var = Z2_bar - Z_bar**2
    hess = var.mean()

    if hess > 1e-8:
        alpha = 1.0 - grad / hess
        # Clamp alpha to a reasonable positive range
        alpha = max(0.1, min(10.0, alpha))
    else:
        alpha = 1.0

    # Apply scaling
    fc2_weight *= alpha
    fc2_bias *= alpha

    w["fc2.weight"] = fc2_weight
    w["fc2.bias"] = fc2_bias

    # Diagnostics
    # Compute training accuracy and CE with new weights
    Z_new = H_aug.dot(np.vstack([fc2_weight.T, fc2_bias.reshape(1, -1)]))
    pred = Z_new.argmax(axis=1)
    acc = (pred == y).mean()
    # CE
    Z_new_max = Z_new.max(axis=1, keepdims=True)
    expZ_new = np.exp(Z_new - Z_new_max)
    p_new = expZ_new / expZ_new.sum(axis=1, keepdims=True)
    ce = -np.log(p_new[np.arange(N), y] + 1e-12).mean()

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "alpha": float(alpha),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
    }
    return {"weights": w, "diagnostics": diagnostics}
