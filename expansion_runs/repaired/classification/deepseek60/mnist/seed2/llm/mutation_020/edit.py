import numpy as np

def edit(w, f):
    # Read current weights
    W1 = w["fc1.weight"].astype(np.float64)   # [128, 784]
    b1 = w["fc1.bias"].astype(np.float64)     # [128]
    W2 = w["fc2.weight"].astype(np.float64)   # [10, 128]
    b2 = w["fc2.bias"].astype(np.float64)     # [10]

    # Feedback data
    X = f["x"].astype(np.float64)             # [2000, 1, 28, 28]
    y = f["y"].astype(np.int64)               # [2000]
    N = X.shape[0]
    X_flat = X.reshape(N, -1)                 # [2000, 784]

    # Compute current pre-activations and features
    Z = X_flat @ W1.T + b1                    # [2000, 128]
    H = np.maximum(Z, 0.0)                    # [2000, 128]

    # Re-center fc1 biases: set bias so that mean pre-activation is 0 per unit
    # This activates roughly half the units per example.
    mean_Z = Z.mean(axis=0)                   # [128]
    b1_new = b1 - mean_Z
    # Recompute features with new bias
    Z_new = X_flat @ W1.T + b1_new
    H_new = np.maximum(Z_new, 0.0)

    # Fit regularized LDA on H_new
    classes = np.unique(y)
    C = len(classes)
    d = H_new.shape[1]

    # Class means and overall mean
    means = np.zeros((C, d), dtype=np.float64)
    priors = np.zeros(C, dtype=np.float64)
    for i, c in enumerate(classes):
        idx = (y == c)
        means[i] = H_new[idx].mean(axis=0)
        priors[i] = idx.mean()

    mean_all = H_new.mean(axis=0)

    # Within-class scatter (pooled covariance)
    Sw = np.zeros((d, d), dtype=np.float64)
    for i, c in enumerate(classes):
        idx = (y == c)
        diff = H_new[idx] - means[i]
        Sw += diff.T @ diff
    Sw /= (N - C)

    # Shrinkage regularization
    alpha = 1e-2
    Sw_reg = Sw + alpha * np.eye(d)

    # LDA weights: W_lda = Sw_reg^{-1} means^T
    # Solve Sw_reg @ W_lda^T = means^T
    W_lda = np.linalg.solve(Sw_reg, means.T).T   # [C, d]

    # LDA biases
    b_lda = -0.5 * np.sum(means * W_lda, axis=1) + np.log(priors + 1e-12)

    # Write back
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W_lda.astype(np.float32)
    w["fc2.bias"] = b_lda.astype(np.float32)

    # Diagnostics
    train_logits = H_new @ W_lda.T + b_lda
    train_pred = np.argmax(train_logits, axis=1)
    train_acc = (train_pred == y).mean()

    diagnostics = {
        "train_accuracy": float(train_acc),
        "fc1_bias_mean": float(b1_new.mean()),
        "fc1_bias_std": float(b1_new.std()),
        "fc2_weight_norm": float(np.linalg.norm(W_lda)),
        "fc2_bias_norm": float(np.linalg.norm(b_lda)),
        "alpha": float(alpha),
    }

    return {"weights": w, "diagnostics": diagnostics}
