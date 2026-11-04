import numpy as np
import math

def edit(w, f):
    # Extract feedback data
    X = f["x"]                     # (4000, 64) int64 token ids
    y = f["y"]                     # (4000,) int64 labels
    N = X.shape[0]
    V = w["embed.weight"].shape[0] # 2000
    D = w["embed.weight"].shape[1] # 32

    # Build bag-of-words matrix (4000 x 2000)
    BOW = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        for t in X[i]:
            if t > 0:              # ignore padding (0)
                BOW[i, t] += 1.0

    # Optional: L2 normalize each document (helps linear models)
    # BOW = BOW / np.maximum(np.linalg.norm(BOW, axis=1, keepdims=True), 1e-8)

    # Compute a full linear classifier on BOW using ridge regression
    # We use one-vs-rest for the two classes, then combine.
    # Add a small bias term to the features.
    BOW_aug = np.concatenate([BOW, np.ones((N, 1))], axis=1)  # (N, V+1)
    # Ridge regression for each class
    lambda_reg = 1.0
    I = np.eye(V + 1)
    I[-1, -1] = 0.0  # don't regularize bias
    A = BOW_aug.T @ BOW_aug + lambda_reg * I
    # Targets: +1 for class 1, -1 for class 0
    Y = np.where(y == 1, 1.0, -1.0)
    W_full = np.linalg.solve(A, BOW_aug.T @ Y)  # (V+1,)
    # W_full[:V] are weights for tokens, W_full[V] is bias

    # We want to embed this linear classifier into the 32-dim bottleneck.
    # Create a 32 x V matrix P0 whose first row is W_full[:V] (the token weights).
    # The remaining rows can be random but we will orthogonalize them.
    P0 = np.zeros((D, V), dtype=np.float64)
    P0[0, :] = W_full[:V]
    # Fill remaining rows with random values
    rng = np.random.RandomState(42)
    for r in range(1, D):
        P0[r, :] = rng.randn(V)
    # Orthogonalize rows 1..D-1 against row 0 and each other
    # (Gram-Schmidt)
    for r in range(1, D):
        v = P0[r, :].copy()
        v -= np.dot(v, P0[0, :]) / np.dot(P0[0, :], P0[0, :]) * P0[0, :]
        for s in range(1, r):
            v -= np.dot(v, P0[s, :]) / np.dot(P0[s, :], P0[s, :]) * P0[s, :]
        P0[r, :] = v

    # Now normalize each column of P0 to have zero mean and unit variance.
    # This makes LayerNorm (with gamma=1, beta=0) an identity operation.
    P = P0.copy()
    col_mean = P.mean(axis=0, keepdims=True)   # (1, V)
    P = P - col_mean
    col_std = P.std(axis=0, keepdims=True)     # (1, V)
    col_std[col_std < 1e-8] = 1.0
    P = P / col_std

    # Set embedding weights: embed.weight is (V, D), so we need P.T
    w["embed.weight"] = P.T.astype(np.float32)

    # Zero position embeddings
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Zero all transformer block weights and biases
    for block in ["block0", "block1"]:
        for sub in ["norm1", "qkv", "proj", "norm2", "ff1", "ff2"]:
            for param in ["weight", "bias"]:
                key = f"{block}.{sub}.{param}"
                if key in w:
                    w[key] = np.zeros_like(w[key], dtype=np.float32)

    # Set final norm to identity: weight=1, bias=0
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # Now compute the pooled features for all training examples using the new weights.
    # Since blocks are zeroed and norm is identity, the forward pass simplifies to:
    # h = embed(ids) + pos(arange(64))  -> pos is zero, so h = embed(ids)
    # norm(h) = h (because each token embedding has zero mean and unit variance)
    # pooled = mean over non-padding tokens of h
    # logits = head @ pooled + head.bias
    # We can compute pooled features directly:
    # For each document, sum the embeddings of its tokens, then divide by number of tokens.
    # Embedding for token t is P[:, t] (since embed.weight[t] = P[:, t]).
    # So pooled = (1/L) * sum_{t in doc} P[:, t] = (1/L) * P @ BOW_doc.
    # We already have BOW (N x V). So pooled = (BOW @ P.T) / L, where L is number of tokens.
    # Compute L for each document (number of non-padding tokens)
    L = np.sum(X > 0, axis=1).astype(np.float64)  # (N,)
    L[L == 0] = 1.0  # avoid division by zero
    # pooled features: (N, D)
    pooled = (BOW @ P.T) / L[:, None]  # (N, D)

    # Fit the head (2 x D) and bias (2,) using least squares on these pooled features.
    # We want to predict logits for classes 0 and 1. We can use one-hot targets.
    # Use ridge regression with small regularization.
    pooled_aug = np.concatenate([pooled, np.ones((N, 1))], axis=1)  # (N, D+1)
    # Targets: one-hot for class 0 and class 1
    T = np.zeros((N, 2))
    T[np.arange(N), y] = 1.0
    # Ridge regression
    lambda_head = 1e-3
    I_head = np.eye(D + 1)
    I_head[-1, -1] = 0.0
    A_head = pooled_aug.T @ pooled_aug + lambda_head * I_head
    head_params = np.linalg.solve(A_head, pooled_aug.T @ T)  # (D+1, 2)
    # head_params[:D, :] are weights (D, 2), head_params[D, :] is bias (2,)
    w["head.weight"] = head_params[:D, :].T.astype(np.float32)  # (2, D)
    w["head.bias"] = head_params[D, :].astype(np.float32)       # (2,)

    # Diagnostics
    diagnostics = {
        "method": "full linear classifier on BOW embedded via column-normalized projection",
        "bow_shape": list(BOW.shape),
        "lambda_reg": lambda_reg,
        "lambda_head": lambda_head,
        "embed_norm_mean": float(np.mean(np.linalg.norm(w["embed.weight"], axis=1))),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
    }

    return {"weights": w, "diagnostics": diagnostics}
