import numpy as np

def edit(w, f):
    F = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    N, D = F.shape
    C = 10
    Y = np.eye(C)[y]
    
    # Standardize features
    mu = F.mean(axis=0)
    sigma = F.std(axis=0) + 1e-8
    F_std = (F - mu) / sigma
    
    # Current fc2 in standardized space (for prior)
    W0 = w["fc2.weight"].astype(np.float64)
    b0 = w["fc2.bias"].astype(np.float64)
    W0_std = W0 * sigma[None, :]
    b0_std = b0 + mu @ W0.T
    
    # Cross-validation setup
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_acc = -1.0
    best_method = None
    best_params = None
    results = []
    
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    # 1. Ridge regression on standardized features
    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            F_train = F_std[train_idx]
            Y_train = Y[train_idx]
            F_val = F_std[val_idx]
            y_val = y[val_idx]
            
            Y_mean = Y_train.mean(axis=0)
            Y_centered = Y_train - Y_mean
            A = F_train.T @ F_train + lam * np.eye(D)
            B = F_train.T @ Y_centered
            W_std = np.linalg.solve(A, B)
            b_std = Y_mean
            logits = F_val @ W_std + b_std
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_val))
        mean_acc = np.mean(accs)
        results.append(("ridge", lam, float(mean_acc)))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_method = "ridge"
            best_params = lam
            
    # 2. LDA with ridge on pooled covariance
    for lam in lambdas:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            F_train = F_std[train_idx]
            y_train = y[train_idx]
            F_val = F_std[val_idx]
            y_val = y[val_idx]
            
            mu_c = np.zeros((C, D))
            priors = np.zeros(C)
            for c in range(C):
                mask = (y_train == c)
                if np.sum(mask) > 0:
                    mu_c[c] = F_train[mask].mean(axis=0)
                    priors[c] = np.sum(mask) / len(y_train)
                else:
                    mu_c[c] = 0
                    priors[c] = 1e-6
            Sigma = np.zeros((D, D))
            for c in range(C):
                mask = (y_train == c)
                if np.sum(mask) > 1:
                    diff = F_train[mask] - mu_c[c]
                    Sigma += diff.T @ diff
            Sigma /= (len(y_train) - C)
            Sigma += lam * np.eye(D)
            W = np.linalg.solve(Sigma, mu_c.T).T
            b = -0.5 * np.sum(mu_c * W, axis=1) + np.log(priors + 1e-12)
            logits = F_val @ W.T + b
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_val))
        mean_acc = np.mean(accs)
        results.append(("lda", lam, float(mean_acc)))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_method = "lda"
            best_params = lam
            
    # 3. Ridge with prior (shrinkage toward current fc2)
    for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0]:
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            F_train = F_std[train_idx]
            Y_train = Y[train_idx]
            F_val = F_std[val_idx]
            y_val = y[val_idx]
            
            Y_mean = Y_train.mean(axis=0)
            Y_centered = Y_train - Y_mean
            A = F_train.T @ F_train + lam * np.eye(D)
            B = F_train.T @ Y_centered + lam * W0_std
            W_std = np.linalg.solve(A, B)
            b_std = Y_mean
            logits = F_val @ W_std + b_std
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_val))
        mean_acc = np.mean(accs)
        results.append(("ridge_prior", lam, float(mean_acc)))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_method = "ridge_prior"
            best_params = lam
            
    # Fit on full data with best method
    if best_method == "ridge":
        lam = best_params
        Y_mean = Y.mean(axis=0)
        Y_centered = Y - Y_mean
        A = F_std.T @ F_std + lam * np.eye(D)
        B = F_std.T @ Y_centered
        W_std = np.linalg.solve(A, B)
        b_std = Y_mean
    elif best_method == "lda":
        lam = best_params
        mu_c = np.zeros((C, D))
        priors = np.zeros(C)
        for c in range(C):
            mask = (y == c)
            mu_c[c] = F_std[mask].mean(axis=0)
            priors[c] = np.sum(mask) / N
        Sigma = np.zeros((D, D))
        for c in range(C):
            mask = (y == c)
            diff = F_std[mask] - mu_c[c]
            Sigma += diff.T @ diff
        Sigma /= (N - C)
        Sigma += lam * np.eye(D)
        W_std = np.linalg.solve(Sigma, mu_c.T).T
        b_std = -0.5 * np.sum(mu_c * W_std, axis=1) + np.log(priors + 1e-12)
    elif best_method == "ridge_prior":
        lam = best_params
        Y_mean = Y.mean(axis=0)
        Y_centered = Y - Y_mean
        A = F_std.T @ F_std + lam * np.eye(D)
        B = F_std.T @ Y_centered + lam * W0_std
        W_std = np.linalg.solve(A, B)
        b_std = Y_mean
    else:
        W_std = W0_std
        b_std = b0_std
        
    # Convert back to original feature space
    W_orig = W_std / sigma[None, :]
    b_orig = b_std - mu @ W_orig.T
    
    # Clip to safe range and cast
    W_orig = np.clip(W_orig, -1000.0, 1000.0).astype(np.float32)
    b_orig = np.clip(b_orig, -1000.0, 1000.0).astype(np.float32)
    
    w["fc2.weight"] = W_orig
    w["fc2.bias"] = b_orig
    
    diagnostics = {
        "best_method": best_method,
        "best_params": best_params,
        "cv_accuracy": float(best_acc),
        "fc2_weight_norm": float(np.linalg.norm(W_orig)),
        "fc2_bias_norm": float(np.linalg.norm(b_orig)),
        "results": results
    }
    return {"weights": w, "diagnostics": diagnostics}
