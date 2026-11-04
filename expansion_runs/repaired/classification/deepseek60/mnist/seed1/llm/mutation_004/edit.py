import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    num_classes = 10

    # Current hidden features (already computed for the full pool)
    H_cur = f["features"].astype(np.float64)  # (2000, 128)

    # PCA on raw pixels
    mean_X = X.mean(axis=0)
    Xc = X - mean_X
    # SVD to get principal components
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:128]  # (128, 784)

    # Helper: ridge regression for fc2 given features H and labels y
    def fit_fc2(H, y, lam):
        H_aug = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)  # (N, 129)
        Y = np.zeros((H.shape[0], num_classes))
        Y[np.arange(H.shape[0]), y] = 1.0
        A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1])
        B = H_aug.T @ Y
        W = np.linalg.solve(A, B)  # (129, 10)
        return W[:128].T, W[128]   # (10,128), (10,)

    def evaluate(H, y, fc2_w, fc2_b):
        logits = H @ fc2_w.T + fc2_b
        preds = np.argmax(logits, axis=1)
        return np.mean(preds == y)

    # Validation split
    np.random.seed(42)
    indices = np.random.permutation(N)
    val_size = 400
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    fc1_w_cur = w["fc1.weight"].astype(np.float64)
    fc1_b_cur = w["fc1.bias"].astype(np.float64)

    # Try different hybrid splits: s = number of current fc1 units kept
    splits = [0, 32, 64, 96, 128]
    best_acc = -1.0
    best_s = None
    best_lam = None
    best_fc2_w = None
    best_fc2_b = None
    best_W_hyb = None
    best_b_hyb = None

    for s in splits:
        # Build hybrid fc1
        if s == 0:
            W_hyb = V.copy()
            b_hyb = -mean_X @ V.T
        elif s == 128:
            W_hyb = fc1_w_cur.copy()
            b_hyb = fc1_b_cur.copy()
        else:
            W_cur = fc1_w_cur[:s]
            b_cur = fc1_b_cur[:s]
            V_pca = V[:128 - s]
            b_pca = -mean_X @ V_pca.T
            W_hyb = np.vstack([W_cur, V_pca])
            b_hyb = np.concatenate([b_cur, b_pca])

        # Compute features for all data
        H_hyb = np.maximum(0, X @ W_hyb.T + b_hyb)  # (2000, 128)

        for lam in lambdas:
            fc2_w, fc2_b = fit_fc2(H_hyb[train_idx], y[train_idx], lam)
            acc = evaluate(H_hyb[val_idx], y[val_idx], fc2_w, fc2_b)
            if acc > best_acc:
                best_acc = acc
                best_s = s
                best_lam = lam
                best_fc2_w = fc2_w
                best_fc2_b = fc2_b
                best_W_hyb = W_hyb
                best_b_hyb = b_hyb

    # Refit on all data using the best configuration
    if best_s == 0:
        H_final = np.maximum(0, X @ V.T - mean_X @ V.T)
    elif best_s == 128:
        H_final = H_cur
    else:
        H_final = np.maximum(0, X @ best_W_hyb.T + best_b_hyb)

    fc2_w_final, fc2_b_final = fit_fc2(H_final, y, best_lam)

    # Update weights
    if best_s != 128:
        w["fc1.weight"] = best_W_hyb.astype(np.float32)
        w["fc1.bias"] = best_b_hyb.astype(np.float32)
    w["fc2.weight"] = fc2_w_final.astype(np.float32)
    w["fc2.bias"] = fc2_b_final.astype(np.float32)

    # Clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "chosen_s": int(best_s),
        "chosen_lambda": float(best_lam),
        "val_acc": float(best_acc),
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
