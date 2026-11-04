import numpy as np

def edit(w, f):
    X = f["x"].reshape(len(f["x"]), -1).astype(np.float32)
    y = f["y"].astype(np.int64)
    
    W1_orig = w["fc1.weight"].copy()
    b1_orig = w["fc1.bias"].copy()
    
    def compute_H(normalize):
        pre = X @ W1_orig.T + b1_orig
        if normalize:
            std = pre.std(axis=0) + 1e-6
            W1 = W1_orig / std[:, None]
            b1 = b1_orig / std
            pre = X @ W1.T + b1
        else:
            W1 = W1_orig
            b1 = b1_orig
        b1 = b1 - pre.mean(axis=0)
        H = np.maximum(0, X @ W1.T + b1)
        return H, W1, b1
    
    def lda_fit(H, y, alpha, standardize):
        if standardize:
            mu = H.mean(axis=0)
            sigma = H.std(axis=0) + 1e-6
            H_std = (H - mu) / sigma
        else:
            mu = H.mean(axis=0)
            sigma = None
            H_std = H
        
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = H.shape[1]
        
        mus = np.zeros((n_classes, n_features), dtype=np.float32)
        for i, c in enumerate(classes):
            mus[i] = H_std[y == c].mean(axis=0)
        
        Sw = np.zeros((n_features, n_features), dtype=np.float32)
        for i, c in enumerate(classes):
            Hc = H_std[y == c] - mus[i]
            Sw += Hc.T @ Hc
        
        Sw += alpha * np.eye(n_features, dtype=np.float32)
        
        W = np.linalg.solve(Sw, mus.T)
        priors = np.array([np.sum(y == c) for c in classes], dtype=np.float32) / len(y)
        b = -0.5 * np.sum(mus * W.T, axis=1) + np.log(priors)
        W = W.T
        
        if standardize:
            W = W / sigma
            b = b - W @ mu
        
        return W.astype(np.float32), b.astype(np.float32)
    
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    standardizes = [False, True]
    normalizes = [False, True]
    
    best_acc = -1.0
    best_params = (1e-3, False, False)
    
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    for normalize in normalizes:
        H, _, _ = compute_H(normalize)
        for alpha in alphas:
            for std in standardizes:
                accs = []
                for i in range(5):
                    val_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                    W2_cv, b2_cv = lda_fit(H[train_idx], y[train_idx], alpha, std)
                    logits_cv = H[val_idx] @ W2_cv.T + b2_cv
                    preds = np.argmax(logits_cv, axis=1)
                    accs.append(np.mean(preds == y[val_idx]))
                mean_acc = float(np.mean(accs))
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_params = (alpha, std, normalize)
    
    alpha, std, normalize = best_params
    H, W1_new, b1_new = compute_H(normalize)
    W2, b2 = lda_fit(H, y, alpha, std)
    
    W1_new = np.clip(W1_new, -1000, 1000).astype(np.float32)
    b1_new = np.clip(b1_new, -1000, 1000).astype(np.float32)
    W2 = np.clip(W2, -1000, 1000).astype(np.float32)
    b2 = np.clip(b2, -1000, 1000).astype(np.float32)
    
    w["fc1.weight"] = W1_new
    w["fc1.bias"] = b1_new
    w["fc2.weight"] = W2
    w["fc2.bias"] = b2
    
    diagnostics = {
        "best_alpha": float(alpha),
        "standardize": bool(std),
        "normalize": bool(normalize),
        "cv_accuracy": float(best_acc),
        "fc1_weight_norm": float(np.linalg.norm(W1_new)),
        "fc1_bias_norm": float(np.linalg.norm(b1_new)),
        "fc2_weight_norm": float(np.linalg.norm(W2)),
        "fc2_bias_norm": float(np.linalg.norm(b2)),
    }
    return {"weights": w, "diagnostics": diagnostics}
