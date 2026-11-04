import numpy as np

def edit(w, f):
    # Extract current weights
    W1 = w["fc1.weight"].copy()   # [128, 784]
    b1 = w["fc1.bias"].copy()     # [128]
    W2 = w["fc2.weight"].copy()   # [10, 128]
    b2 = w["fc2.bias"].copy()     # [10]

    X = f["x"]                    # [2000, 1, 28, 28]
    y = f["y"]                    # [2000]
    N = X.shape[0]
    X_flat = X.reshape(N, -1).astype(np.float32)  # [2000, 784]

    # Identify the 9 units with the most negative biases (likely dead or weak)
    dead_idx = np.argsort(b1)[:9]

    # Compute class means and overall mean for LDA directions
    classes = np.unique(y)
    mu_all = X_flat.mean(axis=0)
    mu_c = np.array([X_flat[y == c].mean(axis=0) for c in classes])  # [10, 784]

    # Between-class scatter
    Sb = np.zeros((784, 784), dtype=np.float64)
    for c in classes:
        diff = (mu_c[c] - mu_all).reshape(-1, 1)
        Sb += (y == c).sum() * (diff @ diff.T)

    # Within-class scatter (regularized)
    Sw = np.zeros((784, 784), dtype=np.float64)
    for c in classes:
        Xc = X_flat[y == c]
        Xc_centered = Xc - mu_c[c]
        Sw += Xc_centered.T @ Xc_centered
    Sw /= N
    Sw += 1e-2 * np.eye(784)  # regularization

    # Solve generalized eigenproblem for LDA directions
    # We want directions v that maximize v^T Sb v / v^T Sw v
    # Equivalent to solving Sw^{-1} Sb v = lambda v
    Sw_inv = np.linalg.inv(Sw)
    M = Sw_inv @ Sb
    eigvals, eigvecs = np.linalg.eig(M)
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals.real)[::-1]
    eigvecs = eigvecs[:, idx].real
    # Take top 9 directions
    lda_dirs = eigvecs[:, :9].T  # [9, 784]

    # Normalize directions to have unit norm
    lda_dirs = lda_dirs / (np.linalg.norm(lda_dirs, axis=1, keepdims=True) + 1e-12)

    # Replace the selected fc1 units with LDA directions
    for i, unit in enumerate(dead_idx):
        W1[unit] = lda_dirs[i].astype(np.float32)
        # Set bias so that the unit activates around the global mean
        b1[unit] = -np.dot(W1[unit], mu_all).astype(np.float32)

    # Compute new hidden features for the full pool
    H = np.maximum(0, X_flat @ W1.T + b1)  # [2000, 128]

    # Refit only the 9 new columns of fc2 using ridge regression
    # We keep the old columns fixed, so we solve for the new columns to
    # approximate the residual between one-hot targets and current logits.
    # Current logits from old columns (with new columns set to zero)
    W2_old = W2.copy()
    W2_old[:, dead_idx] = 0.0
    logits_old = H @ W2_old.T + b2  # [2000, 10]

    # One-hot targets
    T = np.zeros((N, 10), dtype=np.float32)
    T[np.arange(N), y] = 1.0

    # Residual to fit
    R = T - logits_old  # [2000, 10]

    # Features for the new columns
    H_new = H[:, dead_idx]  # [2000, 9]

    # Ridge regression: solve for W_new (9 x 10) such that H_new @ W_new ≈ R
    # We add a small ridge penalty to keep corrections small
    alpha = 1.0
    A = H_new.T @ H_new + alpha * np.eye(9)
    B = H_new.T @ R
    W_new = np.linalg.solve(A, B)  # [9, 10]

    # Update fc2 weights for the new columns
    W2[:, dead_idx] = W_new.T.astype(np.float32)

    # Optional: adjust fc2 bias slightly to account for mean of new features
    # (keep original bias, as the regression already includes intercept via residual)
    # We can also do a mild temperature scaling to reduce CE.
    # Compute new logits on full pool
    logits_new = H @ W2.T + b2

    # Temperature scaling: find a scalar a such that softmax(a * logits) has lower CE.
    # We use a simple heuristic: set a so that the average max probability matches accuracy.
    # Compute current accuracy and average max prob
    probs = np.exp(logits_new - logits_new.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    acc = (probs.argmax(axis=1) == y).mean()
    avg_max = probs.max(axis=1).mean()
    # If overconfident (avg_max > acc), scale down; if underconfident, scale up.
    # We use a gentle scaling factor.
    if avg_max > 0.01:
        a = np.clip(acc / avg_max, 0.5, 1.5)
    else:
        a = 1.0
    # Apply temperature to fc2 weights and bias
    W2 = (W2 * a).astype(np.float32)
    b2 = (b2 * a).astype(np.float32)

    # Ensure finite and within bounds
    W1 = np.clip(W1, -1000, 1000).astype(np.float32)
    b1 = np.clip(b1, -1000, 1000).astype(np.float32)
    W2 = np.clip(W2, -1000, 1000).astype(np.float32)
    b2 = np.clip(b2, -1000, 1000).astype(np.float32)

    w["fc1.weight"] = W1
    w["fc1.bias"] = b1
    w["fc2.weight"] = W2
    w["fc2.bias"] = b2

    diagnostics = {
        "replaced_units": dead_idx.tolist(),
        "lda_eigenvalues": eigvals[idx[:9]].real.tolist(),
        "ridge_alpha": alpha,
        "temperature": float(a),
        "full_pool_accuracy": float(acc),
        "full_pool_avg_max_prob": float(avg_max),
        "fc1_weight_norm": float(np.linalg.norm(W1)),
        "fc2_weight_norm": float(np.linalg.norm(W2)),
        "fc2_bias_norm": float(np.linalg.norm(b2)),
    }

    return {"weights": w, "diagnostics": diagnostics}
