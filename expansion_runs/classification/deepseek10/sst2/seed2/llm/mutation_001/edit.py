import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]          # (4000, 64) int64
    y = f["y"]          # (4000,) int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # 1. Build bag-of-words matrix B (N, V)
    B = np.zeros((N, V), dtype=np.float32)
    np.add.at(B, (np.arange(N)[:, None], x), 1.0)

    # 2. Center and compute top-32 PCA directions
    B_mean = B.mean(axis=0, keepdims=True)
    B_centered = B - B_mean
    # Use SVD to get right singular vectors
    # B_centered is (4000, 2000); full_matrices=False gives Vt (2000, 2000)
    # We only need top 32 components.
    # To save time, we can compute the covariance matrix and use eigh, but SVD is fine.
    U, S, Vt = np.linalg.svd(B_centered, full_matrices=False)
    W_embed = Vt[:32].T.astype(np.float32)  # (2000, 32)

    # 3. Update weights: embed = PCA, pos = 0
    w["embed.weight"] = W_embed
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # 4. Compute new pooled features using the updated embed and zero pos
    # Simulate forward pass: h = embed[x] + pos, then LayerNorm, then masked mean
    h = w["embed.weight"][x]  # (N, L, 32)
    # pos is zero, so no addition needed
    mean = h.mean(axis=-1, keepdims=True)
    var = h.var(axis=-1, keepdims=True)
    h_norm = (h - mean) / np.sqrt(var + 1e-5)
    mask = (x != 0).astype(np.float32)  # (N, L)
    pooled = (h_norm * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    F = pooled.astype(np.float32)  # (N, 32)

    # 5. Fit ridge regression head with 5-fold CV to pick lambda
    Y = np.eye(2, dtype=np.float32)[y]  # (N, 2)
    F_mean = F.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    F_centered = F - F_mean
    Y_centered = Y - Y_mean

    # Candidate lambdas
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lambda = 1e-2
    best_acc = -1.0

    # 5-fold CV
    np.random.seed(42)
    indices = np.random.permutation(N)
    folds = np.array_split(indices, 5)
    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            F_tr = F_centered[train_idx]
            Y_tr = Y_centered[train_idx]
            F_val = F_centered[val_idx]
            Y_val = Y[val_idx]
            # Solve ridge
            A = F_tr.T @ F_tr + lam * np.eye(32, dtype=np.float32)
            W = np.linalg.solve(A, F_tr.T @ Y_tr)  # (32, 2)
            b = Y_mean - F_mean @ W
            pred = F_val @ W + b
            pred_class = np.argmax(pred, axis=1)
            true_class = np.argmax(Y_val, axis=1)
            acc = np.mean(pred_class == true_class)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # 6. Fit final head on all data with best lambda
    A = F_centered.T @ F_centered + best_lambda * np.eye(32, dtype=np.float32)
    W = np.linalg.solve(A, F_centered.T @ Y_centered)  # (32, 2)
    b = Y_mean - F_mean @ W

    w["head.weight"] = W.T.astype(np.float32)  # (2, 32)
    w["head.bias"] = b.astype(np.float32)      # (2,)

    # Optional: zero out block weights to keep them as identity (they already are zero)
    # But we can ensure they are zero to avoid any unexpected effects.
    for key in w:
        if "block0" in key or "block1" in key:
            if "weight" in key or "bias" in key:
                w[key] = np.zeros_like(w[key])

    # Diagnostics
    diag = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
        "embed_norm": float(np.linalg.norm(w["embed.weight"])),
    }

    return {"weights": w, "diagnostics": diag}
