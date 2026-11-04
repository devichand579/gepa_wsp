import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(len(f["x"]), -1).astype(np.float64)  # [2000, 784]
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # Center data
    mean_x = X.mean(axis=0, keepdims=True)
    Xc = X - mean_x

    # Compute class means and within-class scatter
    class_means = np.zeros((C, D), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if idx.sum() > 0:
            class_means[c] = Xc[idx].mean(axis=0)
    overall_mean = Xc.mean(axis=0, keepdims=True)
    S_B = np.zeros((D, D), dtype=np.float64)
    S_W = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        n_c = idx.sum()
        if n_c == 0:
            continue
        diff = class_means[c] - overall_mean
        S_B += n_c * np.outer(diff, diff)
        Xc_c = Xc[idx]
        S_W += Xc_c.T @ Xc_c

    # Regularize within-class scatter
    reg = 1e-3 * np.trace(S_W) / D
    S_W_reg = S_W + reg * np.eye(D)

    # Solve generalized eigenvalue problem for LDA: S_B v = lambda S_W v
    # Use Cholesky of S_W_reg
    L = np.linalg.cholesky(S_W_reg)
    L_inv = np.linalg.inv(L)
    M = L_inv @ S_B @ L_inv.T
    eigvals, eigvecs = np.linalg.eigh(M)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Take top 9 LDA directions (C-1)
    n_lda = min(C - 1, 9)
    V_lda = L_inv.T @ eigvecs[:, :n_lda]  # [D, n_lda]
    # Normalize LDA directions
    for i in range(n_lda):
        V_lda[:, i] /= np.linalg.norm(V_lda[:, i]) + 1e-12

    # PCA on the residual after removing LDA directions
    # Project Xc onto orthogonal complement of V_lda
    if n_lda > 0:
        Q, _ = np.linalg.qr(V_lda)
        X_res = Xc - Xc @ Q @ Q.T
    else:
        X_res = Xc
    # PCA on residual
    cov_res = X_res.T @ X_res / (N - 1)
    eigvals_pca, eigvecs_pca = np.linalg.eigh(cov_res)
    order_pca = np.argsort(eigvals_pca)[::-1]
    eigvecs_pca = eigvecs_pca[:, order_pca]
    # Number of PCA directions to reach 64 total directions
    n_pca = 64 - n_lda
    if n_pca > 0:
        V_pca = eigvecs_pca[:, :n_pca]
    else:
        V_pca = np.zeros((D, 0))
    # Combine directions
    V = np.hstack([V_lda, V_pca])  # [D, 64]
    # Normalize each direction
    for i in range(V.shape[1]):
        V[:, i] /= np.linalg.norm(V[:, i]) + 1e-12

    # Build fc1 weights: for each direction, create two units (+v, -v)
    # fc1.weight shape [128, 784]
    fc1_w = np.zeros((128, D), dtype=np.float32)
    fc1_b = np.zeros(128, dtype=np.float32)
    for i in range(64):
        v = V[:, i]
        fc1_w[2*i] = v.astype(np.float32)
        fc1_w[2*i+1] = -v.astype(np.float32)
        # Bias: use negative of mean projection to center around zero
        # Actually, we can set bias to 0 and let fc2 handle it, but centering helps.
        # Compute mean projection on training data
        proj_mean = (Xc @ v).mean()
        fc1_b[2*i] = -proj_mean
        fc1_b[2*i+1] = proj_mean

    # Compute new hidden features for the full pool
    H = np.maximum(0, Xc @ fc1_w.T + fc1_b)  # [N, 128]

    # Fit fc2 using regularized LDA on H
    # Compute class means of H
    H_means = np.zeros((C, 128), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if idx.sum() > 0:
            H_means[c] = H[idx].mean(axis=0)
    overall_H_mean = H.mean(axis=0, keepdims=True)
    S_B_H = np.zeros((128, 128), dtype=np.float64)
    S_W_H = np.zeros((128, 128), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        n_c = idx.sum()
        if n_c == 0:
            continue
        diff = H_means[c] - overall_H_mean
        S_B_H += n_c * np.outer(diff, diff)
        Hc = H[idx] - H_means[c]
        S_W_H += Hc.T @ Hc

    # Regularize S_W_H
    reg_H = 1e-2 * np.trace(S_W_H) / 128
    S_W_H_reg = S_W_H + reg_H * np.eye(128)

    # LDA for fc2: solve S_B_H w = lambda S_W_H w
    L_H = np.linalg.cholesky(S_W_H_reg)
    L_H_inv = np.linalg.inv(L_H)
    M_H = L_H_inv @ S_B_H @ L_H_inv.T
    eigvals_H, eigvecs_H = np.linalg.eigh(M_H)
    order_H = np.argsort(eigvals_H)[::-1]
    eigvecs_H = eigvecs_H[:, order_H]
    # Take top 9 directions for 10 classes
    n_fc2 = min(C - 1, 9)
    W_lda = L_H_inv.T @ eigvecs_H[:, :n_fc2]  # [128, 9]
    # We need 10 outputs. We can use the 9 LDA directions plus a bias.
    # Actually, for C classes, LDA gives C-1 directions. We can map to C logits by
    # using the class means in the LDA space. A common approach: compute the projection
    # of class means onto the LDA directions, then use nearest class mean.
    # Let's compute the class means in the LDA space.
    proj_means = H_means @ W_lda  # [C, 9]
    # For each class, the logit can be -||proj - proj_mean||^2, but we need linear.
    # Alternatively, we can use the standard LDA decision function:
    # logit_c = w_c^T x + b_c, where w_c = S_W^{-1} mu_c, b_c = -0.5 mu_c^T S_W^{-1} mu_c + log(pi_c)
    # This gives C logits directly. Let's do that.
    S_W_H_inv = np.linalg.inv(S_W_H_reg)
    fc2_w = np.zeros((C, 128), dtype=np.float64)
    fc2_b = np.zeros(C, dtype=np.float64)
    priors = np.array([np.sum(y == c) for c in range(C)], dtype=np.float64) / N
    for c in range(C):
        mu_c = H_means[c]
        w_c = S_W_H_inv @ mu_c
        b_c = -0.5 * mu_c @ S_W_H_inv @ mu_c + np.log(priors[c] + 1e-12)
        fc2_w[c] = w_c
        fc2_b[c] = b_c

    # Scale logits to reasonable range (optional). LDA logits can be small.
    # We can scale so that the average max logit is around 10-20.
    # Compute logits on full pool
    logits = H @ fc2_w.T + fc2_b
    # Scale factor to make standard deviation of logits around 10
    scale = 10.0 / (np.std(logits) + 1e-12)
    fc2_w *= scale
    fc2_b *= scale

    # Update weights
    w["fc1.weight"] = fc1_w.astype(np.float32)
    w["fc1.bias"] = fc1_b.astype(np.float32)
    w["fc2.weight"] = fc2_w.astype(np.float32)
    w["fc2.bias"] = fc2_b.astype(np.float32)

    # Diagnostics
    # Compute accuracy on full pool with new weights
    new_logits = H @ fc2_w.T + fc2_b
    preds = np.argmax(new_logits, axis=1)
    acc = np.mean(preds == y)
    # Cross-entropy
    probs = np.exp(new_logits - new_logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "full_acc": float(acc),
        "full_ce": float(ce),
        "n_lda": int(n_lda),
        "n_pca": int(n_pca),
        "fc1_weight_mean": float(np.mean(fc1_w)),
        "fc1_weight_std": float(np.std(fc1_w)),
        "fc2_weight_mean": float(np.mean(fc2_w)),
        "fc2_weight_std": float(np.std(fc2_w)),
        "fc2_bias_mean": float(np.mean(fc2_b)),
        "fc2_bias_std": float(np.std(fc2_b)),
    }
    return {"weights": w, "diagnostics": diagnostics}
