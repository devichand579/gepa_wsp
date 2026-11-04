import numpy as np
import math

def edit(w, f):
    # Extract full feedback data
    X = f["x"].astype(np.float32).reshape(len(f["x"]), -1)  # [N, 784]
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    C = 10
    D = X.shape[1]
    H = f["features"].astype(np.float32)  # [N, 128] current hidden features
    H = np.maximum(H, 0.0)  # ensure non-negative (should already be ReLU)

    # Current fc2 weights for importance ranking
    fc2_w = w["fc2.weight"].astype(np.float32)  # [10, 128]
    fc2_b = w["fc2.bias"].astype(np.float32)

    # Compute importance of each hidden unit: norm of its weights across classes
    importance = np.linalg.norm(fc2_w, axis=0)  # [128]

    # We will replace the K least important units with supervised raw-pixel LDA directions.
    # Choose K conservatively (e.g., 10) to avoid destroying useful random features.
    K = 10
    replace_idx = np.argsort(importance)[:K]

    # Compute raw-pixel LDA directions (up to 9 directions)
    # Center X
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    # Class means
    class_means = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        class_means[c] = Xc[y == c].mean(axis=0)
    # Within-class scatter
    Sw = np.zeros((D, D), dtype=np.float32)
    for c in range(C):
        Xc_c = Xc[y == c]
        Sw += Xc_c.T @ Xc_c
    Sw /= (N - C)
    # Between-class scatter
    Sb = np.zeros((D, D), dtype=np.float32)
    for c in range(C):
        diff = class_means[c].reshape(-1, 1)
        Sb += (y == c).sum() * (diff @ diff.T)
    Sb /= (C - 1)
    # Add small ridge to Sw for stability
    Sw += np.eye(D, dtype=np.float32) * 1e-3 * np.trace(Sw) / D
    # Solve generalized eigenproblem Sw^{-1} Sb
    # Use Cholesky of Sw
    L = np.linalg.cholesky(Sw)
    L_inv = np.linalg.inv(L)
    M = L_inv @ Sb @ L_inv.T
    eigvals, eigvecs = np.linalg.eigh(M)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Transform back to original space
    lda_dirs = L_inv.T @ eigvecs  # [D, D]
    lda_dirs = lda_dirs[:, :C-1]  # take first 9 directions
    # Normalize directions
    lda_dirs = lda_dirs / (np.linalg.norm(lda_dirs, axis=0, keepdims=True) + 1e-8)

    # Build new fc1 weights and biases
    new_fc1_w = w["fc1.weight"].astype(np.float32).copy()  # [128, 784]
    new_fc1_b = w["fc1.bias"].astype(np.float32).copy()    # [128]

    # For the replaced units, set weights to LDA directions and bias to a positive shift
    # We need to map LDA directions to the replaced indices. We have K=10, but only 9 LDA dirs.
    # Use the 9 LDA dirs and one extra: the mean difference between class 0 and rest, or just repeat.
    # Actually we can use 9 LDA dirs and one random direction, or use the first 10 directions from LDA if we compute more.
    # Let's compute 10 directions by using the first 10 eigenvectors of Sw^{-1} Sb (there are 9 non-zero, but we can take 10).
    # We'll just take the first 9 and for the 10th use the first principal component of X.
    # Compute PCA direction for the 10th
    Xc_pca = Xc
    # Use SVD on a subset for speed? N=2000, D=784, SVD is fine.
    # But we can just use the first LDA direction again or a random direction.
    # Better: use the class mean difference for class 0 vs rest.
    mean0 = Xc[y == 0].mean(axis=0)
    mean_rest = Xc[y != 0].mean(axis=0)
    extra_dir = mean0 - mean_rest
    extra_dir = extra_dir / (np.linalg.norm(extra_dir) + 1e-8)

    dirs = np.zeros((K, D), dtype=np.float32)
    for i in range(K):
        if i < lda_dirs.shape[1]:
            dirs[i] = lda_dirs[:, i]
        else:
            dirs[i] = extra_dir

    # Set the replaced units
    for i, idx in enumerate(replace_idx):
        new_fc1_w[idx] = dirs[i]
        # Set bias so that ReLU preserves the linear signal. We want the pre-activation to be roughly centered.
        # Compute the mean pre-activation for this direction on the training data.
        pre_act = X @ dirs[i]  # [N]
        # Shift so that the median is 0, or use a positive shift.
        # We want the ReLU to be active for positive values. We can set bias = -np.percentile(pre_act, 50) + small.
        # But we need to keep the scale. Let's set bias to -np.mean(pre_act) + 1.0 to ensure some activation.
        new_fc1_b[idx] = -np.mean(pre_act) + 1.0

    # Now compute the new hidden features for all N
    # H_new = ReLU(X @ new_fc1_w.T + new_fc1_b)
    H_new = np.maximum(X @ new_fc1_w.T + new_fc1_b, 0.0)  # [N, 128]

    # Fit fc2 using LDA on H_new
    # Compute class means of H_new
    mu = np.zeros((C, 128), dtype=np.float32)
    for c in range(C):
        mu[c] = H_new[y == c].mean(axis=0)
    # Within-class scatter
    Sw_h = np.zeros((128, 128), dtype=np.float32)
    for c in range(C):
        Hc = H_new[y == c] - mu[c]
        Sw_h += Hc.T @ Hc
    Sw_h /= (N - C)
    # Add ridge
    Sw_h += np.eye(128, dtype=np.float32) * 1e-3 * np.trace(Sw_h) / 128
    # Solve for LDA weights: W = Sw_h^{-1} @ mu^T
    # Use Cholesky
    Lh = np.linalg.cholesky(Sw_h)
    # Solve Lh @ Lh.T @ W = mu.T
    # W = Sw_h^{-1} @ mu.T
    # We can use np.linalg.solve
    W_lda = np.linalg.solve(Sw_h, mu.T).T  # [10, 128]
    # Bias: -0.5 * mu_c^T Sw_h^{-1} mu_c + log(pi_c)
    # Compute Sw_h^{-1} mu_c for each c
    Sw_inv_mu = np.linalg.solve(Sw_h, mu.T)  # [128, 10]
    b_lda = -0.5 * np.sum(mu.T * Sw_inv_mu, axis=0) + np.log(np.array([np.sum(y == c) for c in range(C)]) / N)

    # Alternatively, use ridge regression on one-hot targets with intercept.
    # Let's compute both and choose the one with higher training accuracy.
    # Ridge regression:
    # Augment H_new with ones
    H_aug = np.concatenate([H_new, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, 129]
    Y_onehot = np.zeros((N, C), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0
    # Solve (H_aug^T H_aug + lambda I) W = H_aug^T Y
    # Use a small lambda
    lam = 1e-3
    A = H_aug.T @ H_aug + lam * np.eye(129, dtype=np.float32)
    B = H_aug.T @ Y_onehot
    W_ridge = np.linalg.solve(A, B)  # [129, 10]
    fc2_w_ridge = W_ridge[:128].T  # [10, 128]
    fc2_b_ridge = W_ridge[128]     # [10]

    # Evaluate training accuracy for both
    def accuracy(W, b):
        logits = H_new @ W.T + b
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    acc_lda = accuracy(W_lda, b_lda)
    acc_ridge = accuracy(fc2_w_ridge, fc2_b_ridge)

    if acc_ridge > acc_lda:
        new_fc2_w = fc2_w_ridge
        new_fc2_b = fc2_b_ridge
    else:
        new_fc2_w = W_lda
        new_fc2_b = b_lda

    # Update weights
    w["fc1.weight"] = new_fc1_w.astype(np.float32)
    w["fc1.bias"] = new_fc1_b.astype(np.float32)
    w["fc2.weight"] = new_fc2_w.astype(np.float32)
    w["fc2.bias"] = new_fc2_b.astype(np.float32)

    # Clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "train_accuracy_lda": float(acc_lda),
        "train_accuracy_ridge": float(acc_ridge),
        "chosen": "ridge" if acc_ridge > acc_lda else "lda",
        "replaced_units": replace_idx.tolist(),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
