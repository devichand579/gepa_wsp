import numpy as np

def edit(w, f):
    # Read full feedback pool
    x = f["x"]          # [N,64] int64
    y = f["y"]          # [N] int64
    N = x.shape[0]
    V = w["embed.weight"].shape[0]   # 2000
    D = w["embed.weight"].shape[1]   # 32

    # Build token-example count matrix C [V, N]
    C = np.zeros((V, N), dtype=np.float32)
    # Flatten tokens and example indices, ignore padding (token 0)
    flat_x = x.reshape(-1)
    flat_i = np.repeat(np.arange(N), x.shape[1])
    mask = flat_x != 0
    np.add.at(C, (flat_x[mask], flat_i[mask]), 1.0)

    # Remove mean per token? Actually we want PCA on token vectors across examples.
    # Center the columns (examples) so that PCA captures co-occurrence structure.
    C_mean = C.mean(axis=1, keepdims=True)
    C_centered = C - C_mean

    # Compute top 32 principal components of tokens via SVD of C_centered.
    # C_centered is [2000, 4000]. We want left singular vectors (token embeddings).
    # To avoid full SVD, we can compute the Gram matrix of tokens: G = C_centered @ C_centered.T
    # G is [2000,2000]. Eigendecomposition of G gives eigenvectors = left singular vectors.
    G = C_centered @ C_centered.T
    # Add small ridge for numerical stability
    G += np.eye(V, dtype=np.float32) * 1e-3
    # Compute top D eigenvectors
    eigvals, eigvecs = np.linalg.eigh(G)
    # eigh returns ascending eigenvalues; take largest D
    idx = np.argsort(eigvals)[::-1][:D]
    E = eigvecs[:, idx]   # [V, D]
    # Scale by sqrt(eigenvalues) to get PCA scores? Actually for embedding we can just use eigenvectors.
    # But we want token embeddings that capture variance. Scaling by sqrt(eigval) gives standard PCA.
    E = E * np.sqrt(np.maximum(eigvals[idx], 1e-8))[None, :]

    # Normalise each token embedding to zero mean and unit variance across D dimensions
    # so that final LayerNorm (with weight=1, bias=0) is identity.
    E_mean = E.mean(axis=1, keepdims=True)
    E_centered = E - E_mean
    E_std = E_centered.std(axis=1, keepdims=True) + 1e-6
    E_norm = E_centered / E_std

    # Set embed.weight
    w["embed.weight"] = E_norm.astype(np.float32)

    # Compute pooled features for all training examples
    # For each example i, feature = mean over non-padding tokens of E_norm[token]
    # We can compute this efficiently using the count matrix C.
    # C is [V, N] counts. Sum over tokens of E_norm[token] * count = E_norm.T @ C
    # Then divide by length (number of non-padding tokens per example).
    lengths = (x != 0).sum(axis=1).astype(np.float32)  # [N]
    lengths[lengths == 0] = 1.0
    # pooled_features = (E_norm.T @ C) / lengths  -> [D, N]
    pooled = (E_norm.T @ C) / lengths[None, :]  # [D, N]
    pooled = pooled.T  # [N, D]

    # Fit a linear classifier on pooled features using LDA (closed form)
    # LDA: w = (Sigma_within)^-1 (mu1 - mu0), b = -w·(mu0+mu1)/2
    X = pooled.astype(np.float64)
    y_int = y.astype(np.int64)
    X0 = X[y_int == 0]
    X1 = X[y_int == 1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    # Within-class covariance
    C0 = X0 - mu0
    C1 = X1 - mu1
    Sigma = (C0.T @ C0 + C1.T @ C1) / (len(X0) + len(X1) - 2)
    # Regularise slightly
    Sigma += np.eye(D, dtype=np.float64) * 1e-3
    # Solve Sigma w = mu1 - mu0
    w_lda = np.linalg.solve(Sigma, mu1 - mu0)
    b_lda = -0.5 * w_lda @ (mu0 + mu1)

    # Set head weights and bias
    # head.weight is [2, D]. We want logit_1 - logit_0 = w_lda · pooled + b_lda
    # So set head.weight[0] = 0, head.weight[1] = w_lda, head.bias[0] = 0, head.bias[1] = b_lda
    w["head.weight"] = np.zeros_like(w["head.weight"], dtype=np.float32)
    w["head.weight"][1, :] = w_lda.astype(np.float32)
    w["head.bias"] = np.zeros_like(w["head.bias"], dtype=np.float32)
    w["head.bias"][1] = b_lda.astype(np.float32)

    # Ensure all other weights remain zero/identity as before (blocks identity)
    # pos.weight zero, norm weights 1, biases 0, block weights zero.
    # They are already in w, but we can explicitly set them to be safe.
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for name in w:
        if name.startswith("block"):
            w[name] = np.zeros_like(w[name], dtype=np.float32)
    for name in ["norm.weight", "norm.bias"]:
        if name in w:
            if name.endswith("weight"):
                w[name] = np.ones_like(w[name], dtype=np.float32)
            else:
                w[name] = np.zeros_like(w[name], dtype=np.float32)

    # Diagnostics
    train_pred = (X @ w_lda + b_lda) > 0
    train_acc = (train_pred == (y_int == 1)).mean()
    diagnostics = {
        "train_acc_lda": float(train_acc),
        "w_norm": float(np.linalg.norm(w_lda)),
        "b": float(b_lda),
        "embed_mean": float(E_norm.mean()),
        "embed_std": float(E_norm.std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
