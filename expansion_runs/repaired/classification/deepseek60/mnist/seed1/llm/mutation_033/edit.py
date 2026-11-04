import numpy as np
import math

def edit(w, f):
    # Load data
    X = f["x"].reshape(2000, 784).astype(np.float32)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    C = 10
    
    # Current fc1
    W1_curr = w["fc1.weight"].copy()
    b1_curr = w["fc1.bias"].copy()
    H_curr = np.maximum(0, X @ W1_curr.T + b1_curr)  # (2000,128)
    
    # Class means
    mu = X.mean(axis=0)  # (784,)
    mu_c = np.zeros((C, 784), dtype=np.float32)
    for c in range(C):
        mu_c[c] = X[y==c].mean(axis=0)
    
    # Within-class scatter
    S_w = np.zeros((784, 784), dtype=np.float32)
    for c in range(C):
        Xc = X[y==c] - mu_c[c]
        S_w += Xc.T @ Xc
    lambda_sw = 1e-2 * np.trace(S_w) / 784
    S_w_reg = S_w + lambda_sw * np.eye(784, dtype=np.float32)
    
    # LDA directions
    lda_dirs = []
    for c in range(C):
        w_dir = np.linalg.solve(S_w_reg, mu_c[c] - mu)
        lda_dirs.append(w_dir)
    lda_dirs = np.array(lda_dirs)
    
    # Pairwise LDA
    pair_dirs = []
    for i in range(C):
        for j in range(i+1, C):
            w_dir = np.linalg.solve(S_w_reg, mu_c[i] - mu_c[j])
            pair_dirs.append(w_dir)
    pair_dirs = np.array(pair_dirs)
    
    # Ridge directions
    XTX = X.T @ X
    lambda_ridge = 1e-2 * np.trace(XTX) / 784
    XTX_reg = XTX + lambda_ridge * np.eye(784, dtype=np.float32)
    ridge_dirs = []
    for c in range(C):
        y_c = (y == c).astype(np.float32)
        w_dir = np.linalg.solve(XTX_reg, X.T @ y_c)
        ridge_dirs.append(w_dir)
    ridge_dirs = np.array(ridge_dirs)
    
    # PCA directions
    X_centered = X - mu
    cov = X_centered.T @ X_centered / (N - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    pca_dirs = eigvecs[:, idx[:128]].T
    
    # Current fc1 weights
    curr_dirs = W1_curr
    
    # Random projections
    np.random.seed(42)
    rand_dirs = np.random.randn(500, 784).astype(np.float32)
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    
    # Class mean directions
    mean_dirs = mu_c - mu
    
    # Pairwise mean directions
    pair_mean_dirs = []
    for i in range(C):
        for j in range(i+1, C):
            pair_mean_dirs.append(mu_c[i] - mu_c[j])
    pair_mean_dirs = np.array(pair_mean_dirs)
    
    # Combine all candidates
    candidates = np.vstack([lda_dirs, pair_dirs, ridge_dirs, pca_dirs, curr_dirs, rand_dirs, mean_dirs, pair_mean_dirs])
    norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    candidates = candidates / norms
    
    # Compute F-statistic
    Z = X @ candidates.T
    Z_mean = Z.mean(axis=0)
    Z_c_mean = np.zeros((C, Z.shape[1]), dtype=np.float32)
    for c in range(C):
        Z_c_mean[c] = Z[y==c].mean(axis=0)
    SSB = np.zeros(Z.shape[1], dtype=np.float32)
    for c in range(C):
        n_c = np.sum(y==c)
        SSB += n_c * (Z_c_mean[c] - Z_mean)**2
    SSW = np.zeros(Z.shape[1], dtype=np.float32)
    for c in range(C):
        Zc = Z[y==c]
        SSW += np.sum((Zc - Z_c_mean[c])**2, axis=0)
    F = (SSB / (C - 1)) / (SSW / (N - C) + 1e-8)
    top_idx = np.argsort(F)[::-1][:128]
    W1_new = candidates[top_idx]
    
    # New hidden features
    Z_new = X @ W1_new.T
    b1_new = -Z_new.mean(axis=0)
    H_new = np.maximum(0, Z_new + b1_new)
    
    # One-hot labels
    Y_onehot = np.zeros((N, C), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0
    
    # CV evaluation
    def evaluate(H, method, lambda_val):
        indices = np.arange(N)
        np.random.seed(0)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_train = H[train_idx]
            H_val = H[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            Y_train = Y_onehot[train_idx]
            
            if method == 'ridge':
                H_train_aug = np.hstack([H_train, np.ones((H_train.shape[0], 1), dtype=np.float32)])
                H_val_aug = np.hstack([H_val, np.ones((H_val.shape[0], 1), dtype=np.float32)])
                HTH_aug = H_train_aug.T @ H_train_aug
                HTH_aug_reg = HTH_aug + lambda_val * np.eye(H_train_aug.shape[1], dtype=np.float32)
                HTY_aug = H_train_aug.T @ Y_train
                W_aug = np.linalg.solve(HTH_aug_reg, HTY_aug)
                logits_val = H_val_aug @ W_aug
            elif method == 'lda':
                mu_H = H_train.mean(axis=0)
                mu_H_c = np.zeros((C, H.shape[1]), dtype=np.float32)
                for c in range(C):
                    mu_H_c[c] = H_train[y_train==c].mean(axis=0)
                S_w_H = np.zeros((H.shape[1], H.shape[1]), dtype=np.float32)
                for c in range(C):
                    Hc = H_train[y_train==c] - mu_H_c[c]
                    S_w_H += Hc.T @ Hc
                S_w_H_reg = S_w_H + lambda_val * np.eye(H.shape[1], dtype=np.float32)
                W = np.zeros((H.shape[1], C), dtype=np.float32)
                for c in range(C):
                    W[:, c] = np.linalg.solve(S_w_H_reg, mu_H_c[c] - mu_H)
                b = (Y_train - H_train @ W).mean(axis=0)
                logits_val = H_val @ W + b
            else:
                raise ValueError
            
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y_val)
            accs.append(acc)
        return np.mean(accs)
    
    best_acc = -1
    best_config = None
    
    for method in ['ridge', 'lda']:
        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            acc = evaluate(H_curr, method, lam)
            if acc > best_acc:
                best_acc = acc
                best_config = ('curr', method, lam)
            acc = evaluate(H_new, method, lam)
            if acc > best_acc:
                best_acc = acc
                best_config = ('new', method, lam)
    
    # Final training
    if best_config[0] == 'curr':
        H_final = H_curr
        W1_final = W1_curr
        b1_final = b1_curr
    else:
        H_final = H_new
        W1_final = W1_new
        b1_final = b1_new
    
    method = best_config[1]
    lam = best_config[2]
    
    if method == 'ridge':
        H_aug = np.hstack([H_final, np.ones((N, 1), dtype=np.float32)])
        HTH = H_aug.T @ H_aug
        HTH_reg = HTH + lam * np.eye(H_aug.shape[1], dtype=np.float32)
        HTY = H_aug.T @ Y_onehot
        W_aug = np.linalg.solve(HTH_reg, HTY)
        W2 = W_aug[:128].T
        b2 = W_aug[128]
    else:
        mu_H = H_final.mean(axis=0)
        mu_H_c = np.zeros((C, H_final.shape[1]), dtype=np.float32)
        for c in range(C):
            mu_H_c[c] = H_final[y==c].mean(axis=0)
        S_w_H = np.zeros((H_final.shape[1], H_final.shape[1]), dtype=np.float32)
        for c in range(C):
            Hc = H_final[y==c] - mu_H_c[c]
            S_w_H += Hc.T @ Hc
        S_w_H_reg = S_w_H + lam * np.eye(H_final.shape[1], dtype=np.float32)
        W2 = np.zeros((C, H_final.shape[1]), dtype=np.float32)
        for c in range(C):
            W2[c] = np.linalg.solve(S_w_H_reg, mu_H_c[c] - mu_H)
        b2 = (Y_onehot - H_final @ W2.T).mean(axis=0)
    
    # Update weights
    w["fc1.weight"] = W1_final.astype(np.float32)
    w["fc1.bias"] = b1_final.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)
    
    diagnostics = {
        "best_config": best_config,
        "best_cv_acc": float(best_acc),
        "fc1_weight_mean": float(np.mean(w["fc1.weight"])),
        "fc1_weight_std": float(np.std(w["fc1.weight"])),
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
        "fc2_bias_mean": float(np.mean(w["fc2.bias"])),
        "fc2_bias_std": float(np.std(w["fc2.bias"])),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
