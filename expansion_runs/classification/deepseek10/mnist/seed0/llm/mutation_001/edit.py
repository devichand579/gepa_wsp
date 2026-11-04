import numpy as np

def edit(w, f):
    X = f["x"].reshape(len(f["x"]), -1).astype(np.float32)
    y = f["y"].astype(np.int64)
    H = f["features"].astype(np.float32)
    n = len(y)
    C = 10
    d = H.shape[1]

    Y = np.zeros((n, C), dtype=np.float32)
    Y[np.arange(n), y] = 1.0

    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    def ridge_fit(H_train, Y_train, lam=1e-3):
        H_aug = np.concatenate([H_train, np.ones((H_train.shape[0], 1), dtype=np.float32)], axis=1)
        A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1], dtype=np.float32)
        b = H_aug.T @ Y_train
        return np.linalg.solve(A, b)

    def eval_ridge(H_train, Y_train, H_val, y_val, lam=1e-3):
        W = ridge_fit(H_train, Y_train, lam)
        H_val_aug = np.concatenate([H_val, np.ones((H_val.shape[0], 1), dtype=np.float32)], axis=1)
        logits = H_val_aug @ W
        acc = (logits.argmax(1) == y_val).mean()
        return acc, W

    def lda_fit(H_train, y_train, lam=1e-2):
        M = np.zeros((C, H_train.shape[1]), dtype=np.float32)
        for c in range(C):
            M[c] = H_train[y_train == c].mean(0)
        overall = H_train.mean(0)
        S_w = np.zeros((H_train.shape[1], H_train.shape[1]), dtype=np.float32)
        for c in range(C):
            Hc = H_train[y_train == c] - M[c]
            S_w += Hc.T @ Hc
        S_w_reg = S_w + lam * np.eye(H_train.shape[1], dtype=np.float32)
        W = np.linalg.solve(S_w_reg, (M - overall).T)
        b = -W.T @ overall
        return W, b

    def eval_lda(H_train, y_train, H_val, y_val, lam=1e-2):
        W, b = lda_fit(H_train, y_train, lam)
        logits = H_val @ W + b
        acc = (logits.argmax(1) == y_val).mean()
        return acc, W, b

    acc_ridge, W_ridge = eval_ridge(H[train_idx], Y[train_idx], H[val_idx], y[val_idx], lam=1e-3)
    acc_lda, W_lda, b_lda = eval_lda(H[train_idx], y[train_idx], H[val_idx], y[val_idx], lam=1e-2)

    best_raw_acc = -1.0
    best_raw_params = None
    for alpha in [1e-2, 1e-1, 1.0, 10.0]:
        X_train = X[train_idx]
        y_train = y[train_idx]
        Mx = np.zeros((C, X_train.shape[1]), dtype=np.float32)
        for c in range(C):
            Mx[c] = X_train[y_train == c].mean(0)
        overall_x = X_train.mean(0)
        S_w_x = np.zeros((X_train.shape[1], X_train.shape[1]), dtype=np.float32)
        for c in range(C):
            Xc = X_train[y_train == c] - Mx[c]
            S_w_x += Xc.T @ Xc
        S_b_x = np.zeros((X_train.shape[1], X_train.shape[1]), dtype=np.float32)
        for c in range(C):
            diff = (Mx[c] - overall_x).reshape(-1, 1)
            S_b_x += (y_train == c).sum() * (diff @ diff.T)
        S_w_reg = S_w_x + alpha * np.eye(X_train.shape[1], dtype=np.float32)
        try:
            L = np.linalg.cholesky(S_w_reg)
            L_inv = np.linalg.inv(L)
            M_mat = L_inv @ S_b_x @ L_inv.T
            eigvals, eigvecs = np.linalg.eigh(M_mat)
            order = np.argsort(eigvals)[::-1][:d]
            V = eigvecs[:, order]
            U = L_inv.T @ V
            fc1_w = U.T.astype(np.float32)
            fc1_b = (-U.T @ overall_x).astype(np.float32)
            H_train_new = np.maximum(X_train @ fc1_w.T + fc1_b, 0).astype(np.float32)
            H_val_new = np.maximum(X[val_idx] @ fc1_w.T + fc1_b, 0).astype(np.float32)
            acc_raw, W_raw = eval_ridge(H_train_new, Y[train_idx], H_val_new, y[val_idx], lam=1e-3)
            if acc_raw > best_raw_acc:
                best_raw_acc = acc_raw
                best_raw_params = (fc1_w, fc1_b, W_raw)
        except np.linalg.LinAlgError:
            continue

    if best_raw_acc > max(acc_ridge, acc_lda):
        fc1_w, fc1_b, W_raw = best_raw_params
        w["fc1.weight"] = fc1_w
        w["fc1.bias"] = fc1_b
        w["fc2.weight"] = W_raw[:d].T.astype(np.float32)
        w["fc2.bias"] = W_raw[d].astype(np.float32)
        method = f"raw_lda_ridge_val_acc_{best_raw_acc:.4f}"
        best_val_acc = best_raw_acc
    elif acc_lda > acc_ridge:
        w["fc2.weight"] = W_lda.T.astype(np.float32)
        w["fc2.bias"] = b_lda.astype(np.float32)
        method = f"feature_lda_val_acc_{acc_lda:.4f}"
        best_val_acc = acc_lda
    else:
        w["fc2.weight"] = W_ridge[:d].T.astype(np.float32)
        w["fc2.bias"] = W_ridge[d].astype(np.float32)
        method = f"feature_ridge_val_acc_{acc_ridge:.4f}"
        best_val_acc = acc_ridge

    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
        w[k] = np.nan_to_num(w[k], nan=0.0, posinf=1000.0, neginf=-1000.0)

    diagnostics = {
        "method": method,
        "val_acc": float(best_val_acc),
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
