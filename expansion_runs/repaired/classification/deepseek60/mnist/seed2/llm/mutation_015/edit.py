import numpy as np

def edit(w, f):
    X = f["x"].reshape(2000, -1).astype(np.float32)
    y = f["y"].astype(np.int64)
    
    W1 = w["fc1.weight"].astype(np.float32).copy()
    b1 = w["fc1.bias"].astype(np.float32).copy()
    
    XW = X @ W1.T  # (2000, 128)
    
    def get_H(center_bias, normalize_rows, standardize_features):
        W1_use = W1.copy()
        b1_use = b1.copy()
        if normalize_rows:
            norms = np.linalg.norm(W1_use, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            W1_use = W1_use / norms
            XW_use = X @ W1_use.T
        else:
            XW_use = XW
        if center_bias:
            b1_use = -XW_use.mean(axis=0)
        Z = XW_use + b1_use
        H = np.maximum(Z, 0.0)
        if standardize_features:
            std = H.std(axis=0)
            std[std < 1e-6] = 1.0
            H = H / std
        return H, W1_use, b1_use
    
    def cv_score(H, y, alpha, method):
        n = H.shape[0]
        indices = np.arange(n)
        fold_size = n // 5
        accs = []
        for i in range(5):
            val_idx = indices[i*fold_size:(i+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            H_train = H[train_idx]
            y_train = y[train_idx]
            H_val = H[val_idx]
            y_val = y[val_idx]
            if method == 'lda':
                means = np.zeros((10, H.shape[1]), dtype=np.float32)
                for c in range(10):
                    mask = y_train == c
                    if mask.sum() > 0:
                        means[c] = H_train[mask].mean(axis=0)
                S = np.zeros((H.shape[1], H.shape[1]), dtype=np.float32)
                for c in range(10):
                    mask = y_train == c
                    if mask.sum() > 0:
                        diff = H_train[mask] - means[c]
                        S += diff.T @ diff
                S /= (len(train_idx) - 10)
                S += alpha * np.eye(H.shape[1], dtype=np.float32)
                W = np.linalg.solve(S, means.T)
                b = -0.5 * np.sum(means.T * W, axis=0)
                logits = H_val @ W + b
                pred = logits.argmax(axis=1)
            else:  # ridge
                H_aug = np.hstack([H_train, np.ones((H_train.shape[0], 1), dtype=np.float32)])
                Y = np.zeros((H_train.shape[0], 10), dtype=np.float32)
                Y[np.arange(H_train.shape[0]), y_train] = 1.0
                A = H_aug.T @ H_aug + alpha * np.eye(H_aug.shape[1], dtype=np.float32)
                B = H_aug.T @ Y
                W_aug = np.linalg.solve(A, B)
                H_val_aug = np.hstack([H_val, np.ones((H_val.shape[0], 1), dtype=np.float32)])
                logits = H_val_aug @ W_aug
                pred = logits.argmax(axis=1)
            accs.append((pred == y_val).mean())
        return np.mean(accs)
    
    best_score = -1.0
    best_params = None
    best_H = None
    best_W1 = None
    best_b1 = None
    
    options = []
    for center_bias in [True, False]:
        for normalize_rows in [False, True]:
            for standardize_features in [False, True]:
                options.append((center_bias, normalize_rows, standardize_features))
    
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    methods = ['lda', 'ridge']
    
    for opt in options:
        H, W1_use, b1_use = get_H(*opt)
        for method in methods:
            for alpha in alphas:
                score = cv_score(H, y, alpha, method)
                if score > best_score:
                    best_score = score
                    best_params = (opt, method, alpha)
                    best_H = H
                    best_W1 = W1_use
                    best_b1 = b1_use
    
    opt, method, alpha = best_params
    H, W1_use, b1_use = get_H(*opt)
    
    if method == 'lda':
        means = np.zeros((10, H.shape[1]), dtype=np.float32)
        for c in range(10):
            mask = y == c
            if mask.sum() > 0:
                means[c] = H[mask].mean(axis=0)
        S = np.zeros((H.shape[1], H.shape[1]), dtype=np.float32)
        for c in range(10):
            mask = y == c
            if mask.sum() > 0:
                diff = H[mask] - means[c]
                S += diff.T @ diff
        S /= (len(y) - 10)
        S += alpha * np.eye(H.shape[1], dtype=np.float32)
        W = np.linalg.solve(S, means.T)
        b = -0.5 * np.sum(means.T * W, axis=0)
        W2_new = W.T
        b2_new = b
    else:
        H_aug = np.hstack([H, np.ones((H.shape[0], 1), dtype=np.float32)])
        Y = np.zeros((H.shape[0], 10), dtype=np.float32)
        Y[np.arange(H.shape[0]), y] = 1.0
        A = H_aug.T @ H_aug + alpha * np.eye(H_aug.shape[1], dtype=np.float32)
        B = H_aug.T @ Y
        W_aug = np.linalg.solve(A, B)
        W2_new = W_aug[:128].T
        b2_new = W_aug[128]
    
    w["fc1.weight"] = np.clip(best_W1, -1000.0, 1000.0).astype(np.float32)
    w["fc1.bias"] = np.clip(best_b1, -1000.0, 1000.0).astype(np.float32)
    w["fc2.weight"] = np.clip(W2_new, -1000.0, 1000.0).astype(np.float32)
    w["fc2.bias"] = np.clip(b2_new, -1000.0, 1000.0).astype(np.float32)
    
    # Final training accuracy on the feedback pool
    H_final, _, _ = get_H(*opt)
    if method == 'lda':
        logits = H_final @ W2_new.T + b2_new
    else:
        logits = H_final @ W2_new.T + b2_new
    pred = logits.argmax(axis=1)
    train_acc = float((pred == y).mean())
    
    diagnostics = {
        "best_params": str(best_params),
        "best_cv_score": float(best_score),
        "train_accuracy": train_acc,
        "fc1_bias_mean": float(w["fc1.bias"].mean()),
        "fc1_bias_std": float(w["fc1.bias"].std()),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
