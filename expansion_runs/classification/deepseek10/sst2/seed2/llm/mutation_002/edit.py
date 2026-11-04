import numpy as np

def edit(w, f):
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # 1. Build bag-of-words counts (ignore padding token 0)
    B = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        for t in x[i]:
            if t != 0:
                B[i, t] += 1.0

    # 2. Center and compute covariance
    B_mean = B.mean(axis=0, keepdims=True)
    Bc = B - B_mean
    C = (Bc.T @ Bc) / N

    # 3. Top-32 eigenvectors of covariance
    eigvals, eigvecs = np.linalg.eigh(C)   # ascending
    V_pca = eigvecs[:, -32:]               # [2000, 32]

    # 4. Normalize each token embedding (row) to zero mean and unit variance
    V_norm = V_pca.copy()
    for i in range(V):
        v = V_norm[i]
        v = v - v.mean()
        std = v.std()
        if std > 1e-8:
            v = v / std
        V_norm[i] = v

    w["embed.weight"] = V_norm.astype(np.float32)
    w["pos.weight"] = np.zeros((64, 32), dtype=np.float32)

    # 5. Make final LayerNorm identity: input already has zero mean/unit variance
    eps = 1e-5
    w["norm.weight"] = np.sqrt(1.0 + eps) * np.ones(32, dtype=np.float32)
    w["norm.bias"] = np.zeros(32, dtype=np.float32)

    # 6. Compute new pooled features (mean of token embeddings, ignoring padding)
    features = np.zeros((N, 32), dtype=np.float32)
    for i in range(N):
        tokens = x[i]
        mask = tokens != 0
        if mask.sum() > 0:
            emb = w["embed.weight"][tokens[mask]]
            features[i] = emb.mean(axis=0)

    # 7. Closed-form ridge regression for the head (with intercept)
    X = np.hstack([features, np.ones((N, 1), dtype=np.float32)])  # [N, 33]
    Y = np.eye(2, dtype=np.float32)[y]                            # [N, 2]
    lam = 1.0
    A = X.T @ X + lam * np.eye(33, dtype=np.float32)
    W_aug = np.linalg.solve(A, X.T @ Y)                           # [33, 2]

    w["head.weight"] = W_aug[:32, :].T.astype(np.float32)         # [2, 32]
    w["head.bias"] = W_aug[32, :].astype(np.float32)              # [2]

    # Diagnostics
    diag = {
        "pca_eigenvalues_top5": eigvals[-5:].tolist(),
        "embed_norm_mean": float(np.linalg.norm(w["embed.weight"], axis=1).mean()),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
        "train_accuracy": float((np.argmax(X @ W_aug, axis=1) == y).mean())
    }
    return {"weights": w, "diagnostics": diag}
