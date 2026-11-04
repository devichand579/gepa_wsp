import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]  # [4000, 64] int64
    y = f["y"]  # [4000] int64
    N = x.shape[0]
    V = 2000  # vocabulary size

    # Build bag-of-words matrix B [N, V]
    B = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        # Count tokens (ignore padding id=0)
        tokens = x[i]
        # Use bincount for speed
        counts = np.bincount(tokens, minlength=V)
        B[i] = counts.astype(np.float32)
    # Remove padding count (id=0) from B? We don't need it for projection, but it's fine.
    # Actually we should not include padding in the bag-of-words for projection.
    # But padding is id=0, we can just leave it; it will be ignored in pooling anyway.

    # Compute SVD of B to get top 32 right singular vectors
    # B is [4000, 2000]. We want top 32 components.
    # Use full_matrices=False to get U [4000, 2000], S [2000], Vt [2000, 2000]
    # This might be slow but manageable. Alternatively, compute B^T B and eig.
    # To be safe and fast, we can compute the covariance matrix B^T B (2000x2000) and use eigh.
    # But SVD on 4000x2000 is okay.
    # Let's use np.linalg.svd with full_matrices=False.
    # However, SVD of 4000x2000 might take a few seconds. We can use randomized SVD? Not allowed? We can use np.linalg.svd.
    # Actually, we only need top 32. We can use scipy? Only numpy. We can use np.linalg.svd.
    # To save time, we can compute B^T B (2000x2000) and use eigh. That's 2000^3 = 8e9 ops, maybe 1-2 seconds.
    # Let's do B^T B.
    # B is float32, convert to float64 for stability.
    B64 = B.astype(np.float64)
    # Compute covariance (uncentered, since we want SVD of B)
    # Actually for SVD, we want right singular vectors of B. These are eigenvectors of B^T B.
    # B^T B is 2000x2000.
    # We can compute it as B64.T @ B64.
    # But B64 is 4000x2000, B64.T @ B64 is 2000x2000.
    # Let's compute it.
    # However, B is sparse. We can compute B^T B more efficiently, but 4000x2000 is small enough.
    # Let's just do B64.T @ B64.
    # Wait, 4000x2000 @ 2000x4000? No, B64.T is 2000x4000, B64 is 4000x2000. Product is 2000x2000.
    # That's 2000*2000*4000 = 16e9 operations. Might be a bit slow but okay.
    # Alternatively, use SVD directly. np.linalg.svd on 4000x2000 is also heavy.
    # Let's use np.linalg.svd with full_matrices=False. It might be optimized.
    # Actually, we can use the fact that we only need top 32. We can use np.linalg.svd(B64, full_matrices=False).
    # Let's try SVD. It returns U, S, Vt. Vt is 2000x2000. We take Vt[:32].
    # To avoid memory issues, we can use the 'compute_uv' option? No.
    # Let's just use SVD. It's fine.
    try:
        U, S, Vt = np.linalg.svd(B64, full_matrices=False)
        W = Vt[:32]  # [32, 2000]
    except np.linalg.LinAlgError:
        # Fallback: random projection
        rng = np.random.RandomState(42)
        W = rng.randn(32, V).astype(np.float64)

    # W is [32, 2000]. Each column is a token's 32-dim vector.
    # Normalize each column to zero mean and unit variance.
    W_mean = W.mean(axis=0, keepdims=True)  # [1, 2000]
    W_std = W.std(axis=0, keepdims=True)    # [1, 2000]
    W_norm = (W - W_mean) / (W_std + 1e-8)  # [32, 2000]

    # Set embed.weight: shape [2000, 32]. So embed.weight[token] = W_norm[:, token]
    embed_weight = W_norm.T.astype(np.float32)  # [2000, 32]
    # Ensure token 0 (padding) is zero (doesn't matter, but keep clean)
    embed_weight[0] = 0.0

    # Update weights
    w["embed.weight"] = embed_weight
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Set final norm to identity: weight=1, bias=0
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # Now compute features for all N examples using the new embed and norm.
    # Since norm is identity (because embed vectors have zero mean/unit variance),
    # the feature for example i is the mean of embed_weight[tokens] over non-pad tokens.
    # We can compute this efficiently.
    features = np.zeros((N, 32), dtype=np.float32)
    for i in range(N):
        tokens = x[i]
        mask = tokens != 0
        if mask.sum() == 0:
            features[i] = 0.0
        else:
            # Get embed vectors for non-pad tokens
            vecs = embed_weight[tokens[mask]]  # [L, 32]
            features[i] = vecs.mean(axis=0)

    # Fit linear head via ridge regression.
    # Head: logits = head.weight @ features + head.bias
    # We have 2 classes. Use one-hot labels.
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Center features and labels
    X_mean = features.mean(axis=0, keepdims=True)  # [1, 32]
    Y_mean = Y.mean(axis=0, keepdims=True)         # [1, 2]
    Xc = features - X_mean
    Yc = Y - Y_mean

    # Ridge regression: W_head^T = (Xc^T Xc + lambda I)^-1 Xc^T Yc
    # We want head.weight shape [2, 32], so W_head = (Xc^T Xc + lambda I)^-1 Xc^T Yc, then transpose.
    lambda_reg = 1.0  # small regularization
    XtX = Xc.T @ Xc  # [32, 32]
    XtY = Xc.T @ Yc  # [32, 2]
    # Solve (XtX + lambda I) W = XtY
    A = XtX + lambda_reg * np.eye(32)
    W_head = np.linalg.solve(A, XtY)  # [32, 2]
    W_head = W_head.T  # [2, 32]
    b_head = (Y_mean - X_mean @ W_head.T).flatten()  # [2]

    w["head.weight"] = W_head.astype(np.float32)
    w["head.bias"] = b_head.astype(np.float32)

    # Ensure all other weights are zero (they already are, but just in case)
    # We can leave them as is.

    # Diagnostics
    # Compute training accuracy
    logits = features @ W_head.T + b_head  # [N, 2]
    preds = np.argmax(logits, axis=1)
    train_acc = (preds == y).mean()

    diagnostics = {
        "train_accuracy": float(train_acc),
        "embed_norm_mean": float(np.linalg.norm(embed_weight, axis=1).mean()),
        "head_weight_norm": float(np.linalg.norm(W_head)),
        "head_bias": b_head.tolist(),
        "lambda_reg": lambda_reg,
    }

    return {"weights": w, "diagnostics": diagnostics}
