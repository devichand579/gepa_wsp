import numpy as np

def edit(w, f):
    # Extract arrays
    X = f["x"].reshape(len(f["x"]), -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    H = f["features"].astype(np.float64)  # (2000, 128) current hidden features (after ReLU)
    n, d = X.shape
    n_classes = 10
    n_hidden = H.shape[1]

    # Current fc1 and fc2
    W1 = w["fc1.weight"].astype(np.float64).copy()  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float64).copy()    # (128,)
    W2 = w["fc2.weight"].astype(np.float64).copy()  # (10, 128)
    b2 = w["fc2.bias"].astype(np.float64).copy()    # (10,)

    # Identify dead units: those with very negative bias or zero weights
    # We'll replace the 39 most negative-bias units (as in recent successful proposals)
    dead_indices = np.argsort(b1)[:39]  # most negative biases

    # Compute class means and global mean for raw images
    class_means = np.zeros((n_classes, d))
    for c in range(n_classes):
        class_means[c] = X[y == c].mean(axis=0)
    global_mean = X.mean(axis=0)

    # Between-class scatter
    Sb = np.zeros((d, d))
    for c in range(n_classes):
        diff = (class_means[c] - global_mean).reshape(-1, 1)
        Sb += (y == c).sum() * (diff @ diff.T)

    # Within-class scatter
    Sw = np.zeros((d, d))
    for c in range(n_classes):
        Xc = X[y == c]
        diff = Xc - class_means[c]
        Sw += diff.T @ diff

    # Regularized LDA: solve (Sw + alpha * I) W = Sb W
    alpha = 1e-2 * np.trace(Sw) / d
    Sw_reg = Sw + alpha * np.eye(d)
    # Compute top 9 LDA directions (since 10 classes -> 9 discriminative directions)
    # Use generalized eigenvalue problem: Sw_reg^{-1} Sb
    M = np.linalg.solve(Sw_reg, Sb)
    eigvals, eigvecs = np.linalg.eig(M)
    # Sort by real part of eigenvalues
    idx = np.argsort(np.real(eigvals))[::-1]
    lda_dirs = np.real(eigvecs[:, idx[:9]]).T  # (9, 784)

    # PCA directions from global covariance
    cov = np.cov(X, rowvar=False)
    eigvals_pca, eigvecs_pca = np.linalg.eigh(cov)
    idx_pca = np.argsort(eigvals_pca)[::-1]
    pca_dirs = eigvecs_pca[:, idx_pca[:30]].T  # (30, 784)

    # Combine directions: 9 LDA + 30 PCA = 39
    new_dirs = np.vstack([lda_dirs, pca_dirs])  # (39, 784)

    # Normalize directions to have unit norm (optional, but helps stability)
    norms = np.linalg.norm(new_dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    new_dirs = new_dirs / norms

    # Replace dead units
    W1[dead_indices] = new_dirs
    # Set biases so that activations are centered around zero for the training data
    # Compute pre-activations for new directions
    pre_act = X @ new_dirs.T  # (2000, 39)
    # We want ReLU output to have mean roughly equal to the mean of existing active units?
    # Simpler: set bias to -mean(pre_act, axis=0) so that half are positive?
    # Actually, we want to center at zero so that ReLU is active for positive part.
    # Let's set bias = -median(pre_act, axis=0) to have ~50% activation.
    new_b1 = -np.median(pre_act, axis=0)
    b1[dead_indices] = new_b1

    # Now recompute hidden features for all training data using updated fc1
    # H_new = relu(X @ W1.T + b1)
    H_new = np.maximum(0, X @ W1.T + b1)  # (2000, 128)

    # Refit fc2 using regularized LDA on the new hidden features
    # Compute class means and scatter for H_new
    H_means = np.zeros((n_classes, n_hidden))
    for c in range(n_classes):
        H_means[c] = H_new[y == c].mean(axis=0)
    H_global_mean = H_new.mean(axis=0)

    Sb_h = np.zeros((n_hidden, n_hidden))
    for c in range(n_classes):
        diff = (H_means[c] - H_global_mean).reshape(-1, 1)
        Sb_h += (y == c).sum() * (diff @ diff.T)

    Sw_h = np.zeros((n_hidden, n_hidden))
    for c in range(n_classes):
        Hc = H_new[y == c]
        diff = Hc - H_means[c]
        Sw_h += diff.T @ diff

    # Regularized LDA for fc2
    alpha_h = 1e-2 * np.trace(Sw_h) / n_hidden
    Sw_h_reg = Sw_h + alpha_h * np.eye(n_hidden)
    M_h = np.linalg.solve(Sw_h_reg, Sb_h)
    eigvals_h, eigvecs_h = np.linalg.eig(M_h)
    idx_h = np.argsort(np.real(eigvals_h))[::-1]
    # Take top 9 directions (since 10 classes)
    lda_dirs_h = np.real(eigvecs_h[:, idx_h[:9]]).T  # (9, 128)

    # We need 10 output classes. LDA gives 9 discriminative directions.
    # We can use the 9 LDA directions and add a constant direction (e.g., mean) or use the class means directly.
    # A common approach: use the class means as weights and a bias.
    # For LDA, the decision function is: argmax_c [ H_new @ W_c + b_c ]
    # We can set W2 = H_means (10, 128) and b2 = -0.5 * sum(H_means^2, axis=1) (for equal covariance).
    # This is the standard LDA classifier.
    # Let's use the class means as weights.
    W2_new = H_means.copy()  # (10, 128)
    # Bias: for LDA with shared covariance, bias_c = -0.5 * mu_c^T Sigma^{-1} mu_c + log(prior_c)
    # We can approximate Sigma^{-1} by using the regularized Sw_h.
    Sigma_inv = np.linalg.inv(Sw_h_reg)
    b2_new = np.zeros(n_classes)
    priors = np.array([(y == c).sum() for c in range(n_classes)]) / n
    for c in range(n_classes):
        mu = H_means[c]
        b2_new[c] = -0.5 * mu @ Sigma_inv @ mu + np.log(priors[c])

    # Alternatively, we could use the 9 LDA directions and a bias, but class means are more direct.
    # Let's use the class means approach.

    # Update weights
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2_new.astype(np.float32)
    w["fc2.bias"] = b2_new.astype(np.float32)

    # Diagnostics
    # Compute accuracy on full pool
    logits_new = H_new @ W2_new.T + b2_new
    preds = np.argmax(logits_new, axis=1)
    acc = np.mean(preds == y)
    # Cross-entropy
    # Softmax
    logits_max = logits_new - logits_new.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))

    diagnostics = {
        "full_pool_accuracy": float(acc),
        "full_pool_cross_entropy": float(ce),
        "replaced_units": dead_indices.tolist(),
        "num_lda_dirs": 9,
        "num_pca_dirs": 30,
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"]))
    }

    return {"weights": w, "diagnostics": diagnostics}
