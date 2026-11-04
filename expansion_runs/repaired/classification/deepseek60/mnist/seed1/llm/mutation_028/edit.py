def edit(w, f):
    import numpy as np

    # Load full feedback pool
    X = f["x"].reshape(len(f["x"]), -1).astype(np.float32)   # (2000, 784)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # Current weights
    W1 = w["fc1.weight"].copy()   # (128, 784)
    b1 = w["fc1.bias"].copy()     # (128,)
    W2 = w["fc2.weight"].copy()   # (10, 128)
    b2 = w["fc2.bias"].copy()     # (10,)

    # Center pixels
    X_mean = X.mean(axis=0)
    Xc = X - X_mean

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    Y_mean = Y.mean(axis=0)
    Yc = Y - Y_mean

    # 1) Ridge regression on centered pixels
    alpha_ridge = 1.0
    XtX = Xc.T @ Xc
    XtY = Xc.T @ Yc
    W_ridge = np.linalg.solve(
        XtX + alpha_ridge * np.eye(D, dtype=np.float32), XtY
    )  # (784, 10)

    # 2) LDA directions on centered pixels
    M = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        M[c] = Xc[y == c].mean(axis=0)

    Sw = np.zeros((D, D), dtype=np.float32)
    for c in range(C):
        Xc_c = Xc[y == c]
        diff = Xc_c - M[c]
        Sw += diff.T @ diff

    alpha_lda = 0.1
    Sw_reg = Sw + alpha_lda * np.trace(Sw) / D * np.eye(D, dtype=np.float32)
    W_lda = np.linalg.solve(Sw_reg, M.T)  # (784, 10)

    # 3) Replace 40 least important fc2 units
    importance = np.linalg.norm(W2, axis=0)
    replace_idx = np.argsort(importance)[:40]

    W1_new = W1.copy()
    b1_new = b1.copy()

    directions = []
    for c in range(C):
        directions.append(W_ridge[:, c])
    for c in range(C):
        directions.append(W_lda[:, c])

    idx_ptr = 0
    for w_dir in directions:
        if idx_ptr >= len(replace_idx):
            break
        i = replace_idx[idx_ptr]
        W1_new[i] = w_dir
        b1_new[i] = -np.dot(w_dir, X_mean)
        idx_ptr += 1

        if idx_ptr >= len(replace_idx):
            break
        i = replace_idx[idx_ptr]
        W1_new[i] = -w_dir
        b1_new[i] = np.dot(w_dir, X_mean)
        idx_ptr += 1

    # 4) Compute new hidden features
    H = X @ W1_new.T + b1_new
    H = np.maximum(H, 0.0)

    # 5) Refit fc2 with regularized LDA on H
    M_H = np.zeros((C, 128), dtype=np.float32)
    for c in range(C):
        M_H[c] = H[y == c].mean(axis=0)

    Sw_H = np.zeros((128, 128), dtype=np.float32)
    for c in range(C):
        H_c = H[y == c]
        diff = H_c - M_H[c]
        Sw_H += diff.T @ diff

    alpha_fc2 = 0.1
    Sw_H_reg = Sw_H + alpha_fc2 * np.trace(Sw_H) / 128 * np.eye(128, dtype=np.float32)
    W_fc2 = np.linalg.solve(Sw_H_reg, M_H.T)  # (128, 10)
    b_fc2 = -0.5 * np.sum(M_H * W_fc2.T, axis=1)  # (10,)

    # 6) Clip and cast
    W1_new = np.clip(W1_new, -1000, 1000).astype(np.float32)
    b1_new = np.clip(b1_new, -1000, 1000).astype(np.float32)
    W2_new = np.clip(W_fc2.T, -1000, 1000).astype(np.float32)
    b2_new = np.clip(b_fc2, -1000, 1000).astype(np.float32)

    w["fc1.weight"] = W1_new
    w["fc1.bias"] = b1_new
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new

    # Diagnostics
    logits = H @ W2_new.T + b2_new
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))

    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))

    diagnostics = {
        "train_accuracy": acc,
        "train_cross_entropy": ce,
        "n_replaced": int(len(replace_idx)),
        "fc1_weight_mean": float(W1_new.mean()),
        "fc1_weight_std": float(W1_new.std()),
        "fc2_weight_mean": float(W2_new.mean()),
        "fc2_weight_std": float(W2_new.std()),
        "fc2_bias_mean": float(b2_new.mean()),
        "fc2_bias_std": float(b2_new.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
