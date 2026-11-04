import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["features"].astype(np.float64)  # [4000, 32]
    y = f["y"].astype(np.int64)           # [4000]
    
    # Standardize features
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Z = (X - mu) / sigma
    
    # Class means
    Z0 = Z[y == 0]
    Z1 = Z[y == 1]
    mu0 = Z0.mean(axis=0)
    mu1 = Z1.mean(axis=0)
    diff = mu1 - mu0
    
    # Within-class covariance
    d = Z.shape[1]
    S0 = np.cov(Z0, rowvar=False, bias=True) * (len(Z0) - 1)
    S1 = np.cov(Z1, rowvar=False, bias=True) * (len(Z1) - 1)
    Sigma = (S0 + S1) / (len(Z0) + len(Z1) - 2)
    
    # Cross-validate shrinkage alpha
    best_alpha = 0.15
    best_acc = -1.0
    alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    n_folds = 5
    indices = np.arange(len(y))
    np.random.seed(0)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    for alpha in alphas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            Z_tr, y_tr = Z[tr_idx], y[tr_idx]
            Z_val, y_val = Z[val_idx], y[val_idx]
            
            mu0_tr = Z_tr[y_tr == 0].mean(axis=0)
            mu1_tr = Z_tr[y_tr == 1].mean(axis=0)
            diff_tr = mu1_tr - mu0_tr
            
            S0_tr = np.cov(Z_tr[y_tr == 0], rowvar=False, bias=True) * (len(Z_tr[y_tr == 0]) - 1)
            S1_tr = np.cov(Z_tr[y_tr == 1], rowvar=False, bias=True) * (len(Z_tr[y_tr == 1]) - 1)
            Sigma_tr = (S0_tr + S1_tr) / (len(Z_tr) - 2)
            
            Sigma_shrunk = (1 - alpha) * Sigma_tr + alpha * np.trace(Sigma_tr) / d * np.eye(d)
            w_tr = np.linalg.solve(Sigma_shrunk, diff_tr)
            
            # Optimal bias on training part
            scores_tr = Z_tr @ w_tr
            # Find best threshold
            sorted_idx = np.argsort(scores_tr)
            sorted_scores = scores_tr[sorted_idx]
            sorted_y = y_tr[sorted_idx]
            # Cumulative counts
            n1 = np.cumsum(sorted_y == 1)
            n0 = np.cumsum(sorted_y == 0)
            total1 = (y_tr == 1).sum()
            total0 = (y_tr == 0).sum()
            # For threshold between i and i+1, predict 1 if score > threshold
            # Accuracy = (n1[i] + (total0 - n0[i])) / N
            accs_thr = (n1 + (total0 - n0)) / len(y_tr)
            best_i = np.argmax(accs_thr)
            if best_i < len(sorted_scores) - 1:
                b_tr = -(sorted_scores[best_i] + sorted_scores[best_i + 1]) / 2
            else:
                b_tr = -sorted_scores[best_i] - 1e-6
            # Evaluate on validation
            pred_val = (Z_val @ w_tr + b_tr) > 0
            accs.append((pred_val == y_val).mean())
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_alpha = alpha
    
    # Fit on full pool with best alpha
    Sigma_shrunk = (1 - best_alpha) * Sigma + best_alpha * np.trace(Sigma) / d * np.eye(d)
    w_z = np.linalg.solve(Sigma_shrunk, diff)
    
    # Optimal bias on full pool
    scores = Z @ w_z
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_y = y[sorted_idx]
    n1 = np.cumsum(sorted_y == 1)
    n0 = np.cumsum(sorted_y == 0)
    total1 = (y == 1).sum()
    total0 = (y == 0).sum()
    accs_thr = (n1 + (total0 - n0)) / len(y)
    best_i = np.argmax(accs_thr)
    if best_i < len(sorted_scores) - 1:
        b_z = -(sorted_scores[best_i] + sorted_scores[best_i + 1]) / 2
    else:
        b_z = -sorted_scores[best_i] - 1e-6
    
    # Transform back to original feature space
    w_orig = w_z / sigma
    b_orig = b_z - mu @ w_orig
    
    # Normalize w_orig to unit norm (optional, but keeps scale reasonable)
    norm = np.linalg.norm(w_orig)
    if norm > 1e-8:
        w_orig = w_orig / norm
        b_orig = b_orig / norm
    
    # Set head weights: class 1 uses w_orig, class 0 uses -w_orig
    w["head.weight"][1] = w_orig.astype(np.float32)
    w["head.weight"][0] = (-w_orig).astype(np.float32)
    w["head.bias"][1] = np.float32(b_orig)
    w["head.bias"][0] = np.float32(-b_orig)
    
    # Diagnostics
    train_pred = (X @ w_orig + b_orig) > 0
    train_acc = (train_pred == y).mean()
    diagnostics = {
        "best_alpha": float(best_alpha),
        "cv_accuracy": float(best_acc),
        "train_accuracy": float(train_acc),
        "w_norm": float(np.linalg.norm(w_orig)),
        "bias": float(b_orig),
        "n_samples": int(len(y)),
        "n_features": int(d)
    }
    
    return {"weights": w, "diagnostics": diagnostics}
