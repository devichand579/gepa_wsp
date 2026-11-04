import numpy as np

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                  # (2000,)
    N = X.shape[0]
    num_classes = 10
    Y_onehot = np.eye(num_classes, dtype=np.float32)[y]

    # Current weights
    fc1_w_cur = w["fc1.weight"].copy()
    fc1_b_cur = w["fc1.bias"].copy()

    # Helper: compute hidden features
    def hidden(X, fc1_w, fc1_b):
        return np.maximum(0.0, X @ fc1_w.T + fc1_b)

    # Helper: fit fc2 with ridge regression (closed form)
    def fit_fc2(H, Y_onehot, lam):
        H_aug = np.concatenate([H, np.ones((H.shape[0], 1), dtype=np.float32)], axis=1)
        A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1], dtype=np.float32)
        B = H_aug.T @ Y_onehot
        W = np.linalg.solve(A, B)  # (129, 10)
        fc2_w = W[:-1, :].T        # (10, 128)
        fc2_b = W[-1, :]           # (10,)
        return fc2_w, fc2_b

    # Helper: accuracy
    def accuracy(H, y, fc2_w, fc2_b):
        logits = H @ fc2_w.T + fc2_b
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    # Candidate generators
    candidates = []

    # 1. Current fc1
    candidates.append(("current", fc1_w_cur, fc1_b_cur))

    # 2. PCA features
    mean_X = X.mean(axis=0)
    Xc = X - mean_X
    cov = (Xc.T @ Xc) / (N - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    topk = min(128, eigvecs.shape[1])
    pca_w = eigvecs[:, :topk].T.astype(np.float32)  # (topk, 784)
    pca_b = -pca_w @ mean_X.astype(np.float32)
    if topk < 128:
        pad_w = np.zeros((128 - topk, 784), dtype=np.float32)
        pad_b = np.zeros(128 - topk, dtype=np.float32)
        pca_w = np.vstack([pca_w, pad_w])
        pca_b = np.concatenate([pca_b, pad_b])
    candidates.append(("pca", pca_w, pca_b))

    # 3. Linear classifier on raw pixels (ReLU neutralized)
    # Try a few ridge lambdas for the linear classifier
    for lam_lin in [0.1, 1.0, 10.0, 100.0]:
        A = X.T @ X + lam_lin * np.eye(X.shape[1], dtype=np.float32)
        B = X.T @ Y_onehot
        W_lin = np.linalg.solve(A, B)  # (784, 10)
        # fc1: first 10 rows = W_lin.T, rest zeros
        lin_w = np.zeros((128, 784), dtype=np.float32)
        lin_w[:10, :] = W_lin.T
        lin_b = np.full(128, 1000.0, dtype=np.float32)
        candidates.append((f"linear_lam{lam_lin}", lin_w, lin_b))

    # Evaluate candidates via 5-fold CV
    best_candidate = None
    best_lam = None
    best_cv_acc = -1.0
    lambdas = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

    # 5-fold CV indices
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    for name, fc1_w, fc1_b in candidates:
        H = hidden(X, fc1_w, fc1_b)
        for lam in lambdas:
            accs = []
            for i in range(5):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                H_train = H[train_idx]
                Y_train = Y_onehot[train_idx]
                H_val = H[val_idx]
                y_val = y[val_idx]
                fc2_w, fc2_b = fit_fc2(H_train, Y_train, lam)
                acc = accuracy(H_val, y_val, fc2_w, fc2_b)
                accs.append(acc)
            mean_acc = np.mean(accs)
            if mean_acc > best_cv_acc:
                best_cv_acc = mean_acc
                best_candidate = (name, fc1_w, fc1_b)
                best_lam = lam

    # Train on full data with best candidate and lambda
    name, fc1_w_best, fc1_b_best = best_candidate
    H_full = hidden(X, fc1_w_best, fc1_b_best)
    fc2_w_best, fc2_b_best = fit_fc2(H_full, Y_onehot, best_lam)

    # Update weights
    w["fc1.weight"] = fc1_w_best.astype(np.float32)
    w["fc1.bias"] = fc1_b_best.astype(np.float32)
    w["fc2.weight"] = fc2_w_best.astype(np.float32)
    w["fc2.bias"] = fc2_b_best.astype(np.float32)

    # Clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    # Diagnostics
    train_logits = H_full @ fc2_w_best.T + fc2_b_best
    train_pred = np.argmax(train_logits, axis=1)
    train_acc = np.mean(train_pred == y)
    diagnostics = {
        "chosen_candidate": name,
        "chosen_lambda": float(best_lam),
        "cv_accuracy": float(best_cv_acc),
        "train_accuracy": float(train_acc),
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
