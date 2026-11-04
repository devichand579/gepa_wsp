import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # Current fc1 and fc2
    W1 = w["fc1.weight"].copy()  # (128, 784)
    b1 = w["fc1.bias"].copy()    # (128,)
    W2 = w["fc2.weight"].copy()  # (10, 128)
    b2 = w["fc2.bias"].copy()    # (10,)

    # 1. Compute linear features on raw pixels
    # Augment X with bias column
    Xa = np.concatenate([X, np.ones((N, 1))], axis=1)  # (2000, 785)
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    # Ridge regression for 10 class scores
    lam_ridge = 1.0
    A = Xa.T @ Xa + lam_ridge * np.eye(D + 1)
    W_ridge = np.linalg.solve(A, Xa.T @ Y)  # (785, 10)

    # Class means
    mu = X.mean(axis=0)
    mu_c = np.zeros((C, D))
    for c in range(C):
        mu_c[c] = X[y == c].mean(axis=0)

    # Within-class scatter
    Sw = np.zeros((D, D))
    for c in range(C):
        Xc = X[y == c] - mu_c[c]
        Sw += Xc.T @ Xc
    Sw /= N
    # Regularize Sw
    Sw += 1e-3 * np.eye(D)

    # Between-class scatter
    Sb = np.zeros((D, D))
    for c in range(C):
        diff = (mu_c[c] - mu).reshape(-1, 1)
        Sb += (y == c).sum() * (diff @ diff.T)
    Sb /= N

    # LDA directions: eigenvectors of Sw^{-1} Sb
    Sw_inv = np.linalg.inv(Sw)
    M = Sw_inv @ Sb
    eigvals, eigvecs = np.linalg.eig(M)
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals.real)[::-1]
    eigvecs = eigvecs[:, idx].real
    # Take top 9 directions
    lda_dirs = eigvecs[:, :9].T  # (9, 784)

    # Combine linear features: 10 ridge + 9 LDA = 19 features
    # For ridge, the last row is bias
    ridge_w = W_ridge[:D, :].T  # (10, 784)
    ridge_b = W_ridge[D, :]     # (10,)
    # For LDA, bias = -v·mu
    lda_b = -lda_dirs @ mu      # (9,)

    lin_w = np.concatenate([ridge_w, lda_dirs], axis=0)  # (19, 784)
    lin_b = np.concatenate([ridge_b, lda_b], axis=0)     # (19,)

    # 2. Decide which existing units to replace
    # Importance = norm of fc2 weight column
    importance = np.linalg.norm(W2, axis=0)  # (128,)
    # We need 19 features * 2 units = 38 units. Replace 38 least important.
    n_replace = 38
    replace_idx = np.argsort(importance)[:n_replace]

    # 3. Build new fc1 weights and biases
    new_W1 = W1.copy()
    new_b1 = b1.copy()

    # For each linear feature, create two units: (v, b) and (-v, -b)
    # We'll assign them to the replace_idx positions.
    # We have 19 features, each uses 2 units -> 38 units.
    for i in range(19):
        v = lin_w[i]
        b = lin_b[i]
        idx1 = replace_idx[2 * i]
        idx2 = replace_idx[2 * i + 1]
        new_W1[idx1] = v
        new_b1[idx1] = b
        new_W1[idx2] = -v
        new_b1[idx2] = -b

    # 4. Compute new features on full training data
    H = np.maximum(0, X @ new_W1.T + new_b1)  # (2000, 128)

    # 5. Fit fc2 using regularized LDA on H
    # Compute class means of H
    H_mu = H.mean(axis=0)
    H_mu_c = np.zeros((C, H.shape[1]))
    for c in range(C):
        H_mu_c[c] = H[y == c].mean(axis=0)

    # Within-class scatter of H
    Sw_h = np.zeros((H.shape[1], H.shape[1]))
    for c in range(C):
        Hc = H[y == c] - H_mu_c[c]
        Sw_h += Hc.T @ Hc
    Sw_h /= N
    # Regularize
    alpha = 0.1
    Sw_h_reg = (1 - alpha) * Sw_h + alpha * np.eye(H.shape[1]) * np.trace(Sw_h) / H.shape[1]

    # Between-class scatter of H
    Sb_h = np.zeros((H.shape[1], H.shape[1]))
    for c in range(C):
        diff = (H_mu_c[c] - H_mu).reshape(-1, 1)
        Sb_h += (y == c).sum() * (diff @ diff.T)
    Sb_h /= N

    # LDA projection: solve Sw_h_reg @ W = Sb_h @ W? Actually we want directions that maximize between/within.
    # For multi-class LDA, the projection matrix W is eigenvectors of Sw^{-1} Sb.
    # We can compute the top C-1=9 directions, but we need 10 output classes.
    # Instead, we can use the class means as the classifier: logits = H @ W_lda + b_lda.
    # A common approach: W_lda = Sw_h_reg^{-1} @ H_mu_c.T, and bias = -0.5 * sum(mu_c^2) + log(prior).
    # This gives a linear discriminant classifier.
    Sw_inv_h = np.linalg.inv(Sw_h_reg)
    W_lda = Sw_inv_h @ H_mu_c.T  # (128, 10)
    # Bias: -0.5 * diag(H_mu_c @ Sw_inv_h @ H_mu_c.T) + log(prior)
    # Prior is uniform, so log(prior) = 0.
    b_lda = -0.5 * np.sum(H_mu_c @ Sw_inv_h * H_mu_c, axis=1)
    # Transpose to (10, 128)
    W2_new = W_lda.T
    b2_new = b_lda

    # 6. Update weights
    w["fc1.weight"] = new_W1.astype(np.float32)
    w["fc1.bias"] = new_b1.astype(np.float32)
    w["fc2.weight"] = W2_new.astype(np.float32)
    w["fc2.bias"] = b2_new.astype(np.float32)

    # Diagnostics
    train_logits = H @ W2_new.T + b2_new
    train_preds = np.argmax(train_logits, axis=1)
    train_acc = np.mean(train_preds == y)
    # Cross-entropy
    logits_shift = train_logits - train_logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shift) / np.sum(np.exp(logits_shift), axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "train_accuracy": float(train_acc),
        "train_cross_entropy": float(ce),
        "n_replaced": n_replace,
        "fc1_weight_mean": float(np.mean(new_W1)),
        "fc1_weight_std": float(np.std(new_W1)),
        "fc2_weight_mean": float(np.mean(W2_new)),
        "fc2_weight_std": float(np.std(W2_new)),
        "fc2_bias_mean": float(np.mean(b2_new)),
        "fc2_bias_std": float(np.std(b2_new)),
    }

    return {"weights": w, "diagnostics": diagnostics}
