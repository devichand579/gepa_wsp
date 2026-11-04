import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"]          # [4000, 64] int64 token IDs
    y = f["y"]          # [4000] int64 labels
    N, L = X.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # Build bag-of-words count matrix C [N, V]
    C = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        for t in range(L):
            tok = X[i, t]
            if tok != 0:  # ignore padding
                C[i, tok] += 1.0

    # Normalize counts by sentence length to match mean pooling
    lengths = C.sum(axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    C = C / lengths

    # Compute optimal linear classifier on bag-of-words via ridge regression
    # Add bias column
    Cb = np.concatenate([C, np.ones((N, 1), dtype=np.float32)], axis=1)
    # One-hot labels for two classes
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    # Ridge regression closed form: W = (Cb^T Cb + lambda I)^-1 Cb^T Y
    lam = 1.0
    A = Cb.T @ Cb + lam * np.eye(V + 1, dtype=np.float32)
    B = Cb.T @ Y
    W_cls = np.linalg.solve(A, B)  # [V+1, 2]
    # Separate weights and bias
    W_tok = W_cls[:V, :]   # [V, 2]
    b_cls = W_cls[V, :]    # [2]

    # We will set embed.weight so that pooled feature first two dims = C @ W_tok
    # and head reads those two dims. To keep final LayerNorm roughly linear,
    # we add constant-offset columns to make row means and stds constant.
    E = np.zeros((V, D), dtype=np.float32)
    E[:, 0] = W_tok[:, 0]
    E[:, 1] = W_tok[:, 1]

    # Compute row sums and sums of squares for first two columns
    s1 = E[:, 0] + E[:, 1]
    sq1 = E[:, 0]**2 + E[:, 1]**2

    # Target constant mean and std across features for each token.
    # We want sum_j E[v, j] = S_const, and sum_j E[v, j]^2 = Q_const.
    # Use column 2 to adjust sum, and column 3 to adjust sum of squares.
    S_const = 0.0
    Q_const = 1.0  # arbitrary, ensures std > 0

    E[:, 2] = S_const - s1
    # Now sum of squares so far: sq1 + E[:,2]^2
    # We need E[:,3]^2 = Q_const - (sq1 + E[:,2]^2)
    rem = Q_const - (sq1 + E[:, 2]**2)
    # If rem is negative, clamp to small positive to avoid sqrt of negative
    rem = np.maximum(rem, 1e-6)
    E[:, 3] = np.sqrt(rem)

    # Set head to read first two dims and ignore others
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[0, 0] = 1.0
    head_w[1, 1] = 1.0
    head_b = b_cls.astype(np.float32)

    # Update weights
    w["embed.weight"] = E.astype(np.float32)
    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # Zero out all block weights to keep them identity (they already are zero)
    # but ensure they remain zero to avoid unintended effects.
    for k in list(w.keys()):
        if k.startswith("block"):
            w[k] = np.zeros_like(w[k], dtype=np.float32)

    # Ensure pos is zero
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Set final norm to identity-like: weight=1, bias=0
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # Diagnostics
    # Compute training accuracy using the new linear model on C
    logits = Cb @ W_cls
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y)
    diagnostics = {
        "train_accuracy": float(acc),
        "W_tok_norm": float(np.linalg.norm(W_tok)),
        "b_cls": b_cls.tolist(),
        "n_samples": int(N),
        "n_features": int(D),
    }

    return {"weights": w, "diagnostics": diagnostics}
