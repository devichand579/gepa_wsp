import numpy as np
import math

def edit(w, f):
    # Extract data
    H = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = H.shape
    C = 10
    
    # One-hot targets
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0
    
    # Current logits std for rescaling
    current_logits = f["logits"].astype(np.float64)
    current_std = np.std(current_logits)
    
    # Helper: ridge fit with bias
    def ridge_fit(H_train, Y_train, lam):
        N_tr = H_train.shape[0]
        X = np.hstack([H_train, np.ones((N_tr, 1))])
        D_aug = D + 1
        A = X.T @ X + lam * np.eye(D_aug)
        B = X.T @ Y_train
        W_aug = np.linalg.solve(A, B)  # (D_aug, C)
        W = W_aug[:D, :].T  # (C, D)
        b = W_aug[D, :]     # (C,)
        return W, b
    
    # Helper: predict
    def predict(H, W, b):
        logits = H @ W.T + b
        return np.argmax(logits, axis=1)
    
    # 5-fold CV indices
    np.random.seed(42)
    indices = np.arange(N)
    np.random.shuffle(indices)
    n_folds = 5
    folds = np.array_split(indices, n_folds)
    
    def cv_ridge(H, y, Y, lam):
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            H_train, Y_train = H[train_idx], Y[train_idx]
            H_val, y_val = H[val_idx], y[val_idx]
            W, b = ridge_fit(H_train, Y_train, lam)
            pred = predict(H_val, W, b)
            accs.append(np.mean(pred == y_val))
        return np.mean(accs)
    
    # LDA fit with shrinkage
    def lda_fit(H_train, y_train, lam):
        N_tr = H_train.shape[0]
        mu = np.zeros((C, D))
        counts = np.zeros(C)
        for c in range(C):
            idx = (y_train == c)
            counts[c] = np.sum(idx)
            if counts[c] > 0:
                mu[c] = np.mean(H_train[idx], axis=0)
        # Pooled covariance
        Sigma = np.zeros((D, D))
        for c in range(C):
            idx = (y_train == c)
            if counts[c] > 1:
                diff = H_train[idx] - mu[c]
                Sigma += diff.T @ diff
        Sigma /= (N_tr - C)
        Sigma += lam * np.eye(D)
        Sigma_inv = np.linalg.inv(Sigma)
        W = Sigma_inv @ mu.T  # (D, C)
        b = np.zeros(C)
        for c in range(C):
            b[c] = -0.5 * mu[c] @ Sigma_inv @ mu[c] + np.log(counts[c]/N_tr)
        return W.T, b  # (C, D), (C,)
    
    def cv_lda(H, y, lam):
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            H_train, y_train = H[train_idx], y[train_idx]
            H_val, y_val = H[val_idx], y[val_idx]
            W, b = lda_fit(H_train, y_train, lam)
            pred = predict(H_val, W, b)
            accs.append(np.mean(pred == y_val))
        return np.mean(accs)
    
    # Standardized features
    H_mean = np.mean(H, axis=0)
    H_std = np.std(H, axis=0)
    H_std[H_std < 1e-6] = 1.0
    H_std_feat = (H - H_mean) / H_std
    
    # Evaluate models
    best_acc = -1
    best_method = None
    best_params = None
    
    # Ridge on original
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for lam in lambdas:
        acc = cv_ridge(H, y, Y, lam)
        if acc > best_acc:
            best_acc = acc
            best_method = 'ridge_orig'
            best_params = lam
    
    # Ridge on standardized
    for lam in lambdas:
        acc = cv_ridge(H_std_feat, y, Y, lam)
        if acc > best_acc:
            best_acc = acc
            best_method = 'ridge_std'
            best_params = lam
    
    # LDA with shrinkage
    lda_lambdas = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    for lam in lda_lambdas:
        acc = cv_lda(H, y, lam)
        if acc > best_acc:
            best_acc = acc
            best_method = 'lda'
            best_params = lam
    
    # Fit best model on full data
    if best_method == 'ridge_orig':
        W, b = ridge_fit(H, Y, best_params)
    elif best_method == 'ridge_std':
        W_std, b_std = ridge_fit(H_std_feat, Y, best_params)
        # Convert back to original space
        W = W_std / H_std  # (C, D) / (D,) -> broadcasting
        b = b_std - np.sum(W_std * H_mean / H_std, axis=1)
    elif best_method == 'lda':
        W, b = lda_fit(H, y, best_params)
    else:
        # fallback to current fc2
        W = w["fc2.weight"].astype(np.float64)
        b = w["fc2.bias"].astype(np.float64)
    
    # Rescale logits to match current std
    new_logits = H @ W.T + b
    new_std = np.std(new_logits)
    scale = 1.0
    if new_std > 1e-6:
        scale = current_std / new_std
        # Check if scaling would exceed 1000
        if np.max(np.abs(W * scale)) <= 1000 and np.max(np.abs(b * scale)) <= 1000:
            W *= scale
            b *= scale
        else:
            scale = 1.0
    
    # Ensure float32 and finite
    W = np.asarray(W, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    W = np.clip(W, -1000, 1000)
    b = np.clip(b, -1000, 1000)
    
    w["fc2.weight"] = W
    w["fc2.bias"] = b
    
    # Diagnostics
    diag = {
        "best_method": best_method,
        "best_params": best_params,
        "cv_acc": float(best_acc),
        "current_std": float(current_std),
        "new_std_before_scale": float(new_std),
        "scale": float(scale),
        "fc2_weight_norm": float(np.linalg.norm(W)),
        "fc2_bias_norm": float(np.linalg.norm(b)),
        "fc2_weight_min": float(np.min(W)),
        "fc2_weight_max": float(np.max(W)),
        "fc2_bias_min": float(np.min(b)),
        "fc2_bias_max": float(np.max(b)),
    }
    
    return {"weights": w, "diagnostics": diag}
