import numpy as np

def edit(w, f):
    # Read feedback data
    x = f["x"]  # (4000, 64) int64 token ids
    y = f["y"]  # (4000,) int64 labels
    N, T = x.shape
    V = 2000  # vocabulary size from embed.weight shape

    # Build binary bag-of-words presence matrix, normalized by sentence length
    mask = (x != 0)
    lengths = mask.sum(axis=1).astype(np.float32)
    lengths = np.maximum(lengths, 1.0)

    # BOW: (N, V)
    B = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        toks = x[i][mask[i]]
        if toks.size > 0:
            B[i, toks] = 1.0

    # Normalized features: average of one-hot token vectors
    F = B / lengths[:, None]  # (N, V)

    # Center for PCA
    F_mean = F.mean(axis=0, keepdims=True)
    Fc = F - F_mean

    # Covariance matrix (V, V)
    C = (Fc.T @ Fc) / float(N)

    # Top 32 eigenvectors
    eigvals, eigvecs = np.linalg.eigh(C)
    # eigh returns ascending eigenvalues; take largest 32
    idx = np.argsort(eigvals)[::-1][:32]
    V_pca = eigvecs[:, idx].astype(np.float32)  # (V, 32)

    # Normalize each token embedding row to zero mean and unit variance
    # so the final LayerNorm becomes approximately identity.
    row_mean = V_pca.mean(axis=1, keepdims=True)
    row_std = V_pca.std(axis=1, keepdims=True)
    row_std = np.maximum(row_std, 1e-6)
    V_norm = (V_pca - row_mean) / row_std  # (V, 32)

    # Compute actual pooled features using the normalized embeddings
    P = np.zeros((N, 32), dtype=np.float32)
    for i in range(N):
        toks = x[i][mask[i]]
        if toks.size > 0:
            P[i] = V_norm[toks].mean(axis=0)

    # Train a linear classifier on P using LDA (closed form)
    # Class means
    mu0 = P[y == 0].mean(axis=0)
    mu1 = P[y == 1].mean(axis=0)

    # Pooled covariance (regularized)
    P0 = P[y == 0] - mu0
    P1 = P[y == 1] - mu1
    Sigma = (P0.T @ P0 + P1.T @ P1) / float(N - 2)
    # Add small ridge for numerical stability
    Sigma += np.eye(32, dtype=np.float32) * 1e-4

    # LDA direction
    Sigma_inv = np.linalg.inv(Sigma)
    w_dir = Sigma_inv @ (mu1 - mu0)  # (32,)

    # Bias: decision boundary at w^T x + b = 0
    # Using priors equal (2000 vs 2000)
    b = -0.5 * (mu1 + mu0) @ w_dir

    # Set head weights: class 1 positive, class 0 negative
    head_w = np.zeros((2, 32), dtype=np.float32)
    head_w[1] = w_dir
    head_w[0] = -w_dir
    head_b = np.array([-b, b], dtype=np.float32)

    # Assign weights
    w["embed.weight"] = V_norm.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Zero out block residual projections and FF2 so blocks are identity
    for blk in ["block0", "block1"]:
        w[f"{blk}.proj.weight"] = np.zeros_like(w[f"{blk}.proj.weight"], dtype=np.float32)
        w[f"{blk}.proj.bias"] = np.zeros_like(w[f"{blk}.proj.bias"], dtype=np.float32)
        w[f"{blk}.ff2.weight"] = np.zeros_like(w[f"{blk}.ff2.weight"], dtype=np.float32)
        w[f"{blk}.ff2.bias"] = np.zeros_like(w[f"{blk}.ff2.bias"], dtype=np.float32)

    # Final norm: identity (since embeddings are already normalized)
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # Head
    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # Diagnostics
    train_pred = (P @ w_dir + b > 0).astype(np.int64)
    train_acc = float((train_pred == y).mean())
    diagnostics = {
        "method": "PCA32 + LDA on normalized BOW",
        "train_accuracy": train_acc,
        "pca_explained_variance_ratio_sum": float(eigvals[idx].sum() / eigvals.sum()),
        "head_weight_norm": float(np.linalg.norm(w_dir)),
        "head_bias": float(b),
    }

    return {"weights": w, "diagnostics": diagnostics}
