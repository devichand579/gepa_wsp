import numpy as np
import math

def edit(w, f):
    # Read training data
    x = f["x"]          # [N, 64] int64
    y = f["y"].astype(np.int64)  # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # Build bag-of-words counts [N, V]
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        np.add.at(X[i], x[i], 1.0)

    # Remove padding token 0 from counts? Keep it, but it will be handled.
    # Compute class means
    mask0 = (y == 0)
    mask1 = (y == 1)
    n0 = mask0.sum()
    n1 = mask1.sum()
    mu0 = X[mask0].mean(axis=0)
    mu1 = X[mask1].mean(axis=0)

    # Compute within-class scatter
    # For efficiency, compute covariance per class
    X0 = X[mask0]
    X1 = X[mask1]
    # Center
    X0c = X0 - mu0
    X1c = X1 - mu1
    # Scatter matrices (V x V) - but V=2000, too large to form full matrix.
    # We can use the fact that we only need top 32 LDA directions.
    # Instead, we can compute the between-class scatter and within-class scatter in a reduced space.
    # Since we want a 32-dim projection, we can use PCA on the combined data first to reduce to e.g. 200 dims, then LDA.
    # Or we can use the fact that the LDA solution is given by solving Sw^{-1} Sb w = lambda w.
    # But Sw is V x V. We can use the dual formulation: Sw = X0c^T X0c + X1c^T X1c (V x V).
    # The LDA directions can be found by computing the eigenvectors of Sw^{-1} Sb.
    # Alternatively, we can use the fact that the data is sparse and we can use a linear classifier directly.
    # Simpler: Use PCA on the bag-of-words to get 32 dimensions, then LDA on those 32 dimensions.
    # Actually, we can just use the 32-dim LDA projection by first reducing with PCA to, say, 200 dims, then LDA to 32.
    # But we can also just use a linear classifier on the bag-of-words with 32 dimensions by solving a least squares problem.
    # Let's do PCA on the combined X to 200 dims, then LDA to 32 dims.
    # Center the data
    mu = X.mean(axis=0)
    Xc = X - mu
    # PCA via SVD on Xc (N x V). N=4000, V=2000. SVD is feasible.
    # We want top 200 components.
    # To save time, we can compute the covariance matrix in the sample space: Xc @ Xc.T is N x N (4000x4000), too big.
    # Better: compute SVD of Xc (4000 x 2000). U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # This might be slow but doable. 4000x2000 SVD takes some seconds. We can use a smaller number of components.
    # Actually, we can just use the 32-dim LDA directly by solving a generalized eigenvalue problem on the 2000-dim space using the dual.
    # The LDA solution can be found by solving: (Xc^T Xc + reg I) w = Xc^T Y ... 
    # Wait, LDA for binary classification: the direction is w = Sw^{-1} (mu1 - mu0).
    # Sw = X0c^T X0c + X1c^T X1c. This is V x V. We can compute w by solving Sw w = (mu1 - mu0).
    # Since V=2000, we can solve this linear system directly! Sw is 2000x2000. That's 4 million entries, which is fine.
    # Let's do that.
    # Compute Sw
    Sw = X0c.T @ X0c + X1c.T @ X1c
    # Add small ridge for stability
    Sw += np.eye(V, dtype=np.float32) * 1e-3
    # Compute difference of means
    diff = (mu1 - mu0).astype(np.float32)
    # Solve Sw w = diff
    w_lda = np.linalg.solve(Sw, diff)  # [V]
    # We need 32 dimensions. LDA gives one direction. We can get more by using the top eigenvectors of Sw^{-1} Sb.
    # Sb = (mu1 - mu0)(mu1 - mu0)^T * n0*n1/N. Rank 1. So LDA only gives 1 direction.
    # To get 32 dimensions, we can use PCA on the data, or we can use the top 32 eigenvectors of Sw^{-1} Sb? But Sb is rank 1, so only 1 non-zero eigenvalue.
    # So LDA only gives 1 discriminative direction. We need 32 dimensions. We can use PCA to get 32 dimensions that capture variance, and then maybe rotate to include the LDA direction.
    # Alternatively, we can just use the top 32 PCA components of the bag-of-words. That gives a 32-dim representation.
    # Let's do PCA to 32 dims. We can compute PCA via SVD on Xc.
    # To make it faster, we can compute the covariance matrix in the feature space? V=2000, so V x V is 4M, which is okay.
    # Cov = Xc.T @ Xc / N. This is 2000x2000. Then eigendecomposition of 2000x2000 is doable but maybe slow.
    # SVD on Xc (4000x2000) is probably faster.
    # Let's use SVD.
    # We only need top 32 components.
    # We can use np.linalg.svd(Xc, full_matrices=False) -> U (4000x2000), S (2000), Vt (2000x2000).
    # This might take a few seconds but is acceptable.
    # Actually, we can use the fact that we only need the projection matrix Vt[:32].T.
    # Let's do it.
    # To save memory, we can compute the covariance matrix: C = Xc.T @ Xc. Then use np.linalg.eigh(C) to get eigenvectors.
    # C is 2000x2000. eigh on 2000x2000 takes maybe 1-2 seconds. Let's do that.
    C = Xc.T @ Xc / N
    # Add small ridge
    C += np.eye(V, dtype=np.float32) * 1e-5
    # Get top 32 eigenvectors
    # eigh returns eigenvalues in ascending order
    evals, evecs = np.linalg.eigh(C)
    # Take top 32
    idx = np.argsort(evals)[::-1][:32]
    W_pca = evecs[:, idx]  # [V, 32]
    # Now we have a 32-dim projection. We can also incorporate the LDA direction by replacing the first component with w_lda normalized.
    # Let's just use PCA for now. Or we can use the LDA direction as one of the components.
    # Actually, we can do LDA on the 32-dim PCA features to get a better projection? But we only have 32 dims total.
    # Let's just use PCA. It captures the most variance.
    # Now we have W_pca: [V, 32]. We want embed.weight to be [V, 32] such that embed[v] = W_pca[v, :].
    # So embed.weight = W_pca.
    # Normalize each row of embed.weight to have zero mean and unit variance.
    embed = W_pca.copy()  # [V, 32]
    # For each word v, compute mean and std across the 32 dims
    mean = embed.mean(axis=1, keepdims=True)
    std = embed.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    embed = (embed - mean) / std
    # Set embed.weight
    w["embed.weight"] = embed.astype(np.float32)
    # Set pos.weight to zero
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    # Zero out all block parameters
    for block in ["block0", "block1"]:
        for name in ["norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                     "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{block}.{name}"
            if key in w:
                w[key] = np.zeros_like(w[key])
    # Set final norm to identity: weight=1, bias=0
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # Now compute the pooled features for the training set using the new weights.
    # Since blocks are zero, h = embed(ids) + pos(arange(64)). pos is zero.
    # So h[i, t] = embed[x[i, t]].
    # Then norm is applied per token. Since embed rows have zero mean and unit variance, LayerNorm with weight=1, bias=0 is identity.
    # So h_norm = h.
    # Then pooled = masked_mean(h, ids!=0).
    # So pooled[i] = mean over t where x[i,t] != 0 of embed[x[i,t]].
    # Let's compute this.
    pooled = np.zeros((N, 32), dtype=np.float32)
    for i in range(N):
        tokens = x[i]
        mask = tokens != 0
        if mask.sum() == 0:
            # all padding, use mean of all? just zero
            pooled[i] = 0.0
        else:
            pooled[i] = embed[tokens[mask]].mean(axis=0)
    # Now fit a linear classifier on pooled to predict y.
    # We can use least squares: solve for weights and bias.
    # Add bias term
    pooled_aug = np.concatenate([pooled, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, 33]
    # Target: +1 for class 1, -1 for class 0
    target = np.where(y == 1, 1.0, -1.0).astype(np.float32)
    # Solve least squares: pooled_aug @ theta = target
    # Use ridge for stability
    lam = 1e-3
    A = pooled_aug.T @ pooled_aug + lam * np.eye(33, dtype=np.float32)
    b = pooled_aug.T @ target
    theta = np.linalg.solve(A, b)  # [33]
    head_w = theta[:32]
    head_b = theta[32]
    # Set head weights: head.weight is [2, 32]. For class 0 (negative), we want score = - (head_w @ pooled + head_b).
    # For class 1 (positive), score = head_w @ pooled + head_b.
    # So head.weight[0] = -head_w, head.weight[1] = head_w.
    # head.bias[0] = -head_b, head.bias[1] = head_b.
    w["head.weight"] = np.stack([-head_w, head_w]).astype(np.float32)
    w["head.bias"] = np.array([-head_b, head_b], dtype=np.float32)

    # Diagnostics
    # Compute training accuracy
    logits = pooled @ head_w + head_b
    preds = (logits > 0).astype(np.int64)
    acc = (preds == y).mean()
    diagnostics = {
        "method": "PCA32 + row-normalized embedding + least-squares head",
        "train_accuracy": float(acc),
        "head_weight_norm": float(np.linalg.norm(head_w)),
        "head_bias": float(head_b),
        "embed_mean_abs": float(np.abs(w["embed.weight"]).mean()),
    }
    return {"weights": w, "diagnostics": diagnostics}
