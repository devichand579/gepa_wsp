import numpy as np

def edit(w, f):
    # Copy weights to avoid mutating input directly if needed
    w = {k: v.copy() for k, v in w.items()}
    
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    
    # Current fc1 weights and bias
    W1 = w["fc1.weight"].astype(np.float32)  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float32)    # (128,)
    
    # Compute current hidden features (should match f["features"] but recompute for safety)
    H = np.maximum(0, X @ W1.T + b1)  # (N, 128)
    
    # Identify globally dead units: columns with zero variance (all zeros)
    dead_mask = np.all(H == 0, axis=0)
    dead_idx = np.where(dead_mask)[0]
    D = len(dead_idx)
    
    # If no dead units, we can still try to replace the least useful units.
    # Measure usefulness by the absolute sum of fc2 weights for each unit.
    if D == 0:
        # Replace the 5 units with smallest L1 norm in fc2 weights
        W2 = w["fc2.weight"].astype(np.float32)  # (10, 128)
        unit_scores = np.sum(np.abs(W2), axis=0)
        dead_idx = np.argsort(unit_scores)[:5]
        D = len(dead_idx)
    
    # Generate candidate filters from the data
    # 1. Class mean differences
    global_mean = X.mean(axis=0)  # (784,)
    class_means = np.zeros((10, 784), dtype=np.float32)
    for c in range(10):
        class_means[c] = X[y == c].mean(axis=0)
    class_diff = class_means - global_mean  # (10, 784)
    
    # 2. PCA components of centered data
    X_centered = X - global_mean
    # Use SVD for PCA (N=2000, D=784)
    # Compute covariance matrix (784x784) is too big? 784x784 is fine.
    # But SVD on X_centered (2000x784) is also fine.
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Vt is (min(N,784), 784). We take top components.
    n_pca = min(20, Vt.shape[0])
    pca_filters = Vt[:n_pca]  # (n_pca, 784)
    
    # Combine candidates
    candidates = np.vstack([class_diff, pca_filters])  # (10+n_pca, 784)
    n_cand = candidates.shape[0]
    
    # Evaluate Fisher score for each candidate after ReLU with a chosen bias.
    # For each candidate, we will try both signs and a few bias values.
    # We want to select the best D filters.
    scores = []
    best_params = []  # (filter, bias, sign)
    
    for i in range(n_cand):
        filt = candidates[i]
        # Try both signs
        for sign in [1, -1]:
            f_signed = sign * filt
            proj = X @ f_signed  # (N,)
            # Try several bias values: percentiles of projection
            for pct in [10, 30, 50, 70, 90]:
                b_val = -np.percentile(proj, pct)
                act = np.maximum(0, proj + b_val)  # (N,)
                # Compute Fisher score: between-class variance / within-class variance
                # Class means of activation
                class_means_act = np.array([act[y == c].mean() for c in range(10)])
                overall_mean = act.mean()
                between_var = np.sum((class_means_act - overall_mean)**2) / 10.0
                within_var = 0.0
                for c in range(10):
                    within_var += np.var(act[y == c])
                within_var /= 10.0
                if within_var < 1e-8:
                    fisher = 0.0
                else:
                    fisher = between_var / within_var
                scores.append(fisher)
                best_params.append((f_signed, b_val))
    
    scores = np.array(scores)
    best_params = np.array(best_params, dtype=object)  # avoid dtype issues
    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1]
    
    # Select top D filters (avoid duplicates by taking unique filters? We can just take top D)
    selected = []
    seen = set()
    for idx in sorted_indices:
        filt, b_val = best_params[idx]
        # Use a hash of the filter to avoid exact duplicates
        key = (tuple(np.round(filt[:10], 4)), np.round(b_val, 4))
        if key not in seen:
            seen.add(key)
            selected.append((filt, b_val))
        if len(selected) >= D:
            break
    
    # If we couldn't find enough, fill with random or class means
    while len(selected) < D:
        c = len(selected) % 10
        selected.append((class_diff[c], 0.0))
    
    # Assign selected filters to dead units
    W1_new = W1.copy()
    b1_new = b1.copy()
    for j, (filt, b_val) in enumerate(selected):
        unit = dead_idx[j]
        W1_new[unit] = filt.astype(np.float32)
        b1_new[unit] = np.float32(b_val)
    
    # Compute new hidden features
    H_new = np.maximum(0, X @ W1_new.T + b1_new)  # (N, 128)
    
    # Prepare for Bayesian ridge update of fc2
    # Augment features with bias column
    H_aug = np.hstack([H_new, np.ones((N, 1), dtype=np.float32)])  # (N, 129)
    
    # Current fc2 weights and bias
    W2_old = w["fc2.weight"].astype(np.float32)  # (10, 128)
    b2_old = w["fc2.bias"].astype(np.float32)    # (10,)
    W2_old_aug = np.hstack([W2_old, b2_old[:, None]])  # (10, 129)
    
    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    
    # Bayesian ridge: posterior mean = (H^T H + lambda I)^{-1} (H^T Y + lambda W_prior^T)
    # We'll try a few lambda values and pick the best by cross-validation on the 2000.
    # To keep it fast, we can use a simple train/val split or 5-fold CV.
    # Since N=2000, we can do 5-fold CV for a few lambdas.
    lambdas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    best_lambda = 1.0
    best_cv_acc = -1.0
    
    # 5-fold CV
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(indices)
    
    for lam in lambdas:
        cv_acc = 0.0
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            
            H_train = H_aug[train_idx]
            Y_train = Y[train_idx]
            H_val = H_aug[val_idx]
            y_val = y[val_idx]
            
            # Solve for W (10, 129)
            # (H^T H + lam I) W^T = H^T Y + lam W_prior^T
            # Let A = H^T H + lam I (129x129)
            A = H_train.T @ H_train + lam * np.eye(H_aug.shape[1], dtype=np.float32)
            B = H_train.T @ Y_train + lam * W2_old_aug.T  # (129, 10)
            W_new_aug = np.linalg.solve(A, B).T  # (10, 129)
            
            # Predict on validation
            logits_val = H_val @ W_new_aug.T  # (val_size, 10)
            preds = np.argmax(logits_val, axis=1)
            acc = np.mean(preds == y_val)
            cv_acc += acc
        
        cv_acc /= n_folds
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam
    
    # Final fit on all data with best_lambda
    lam = best_lambda
    A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1], dtype=np.float32)
    B = H_aug.T @ Y + lam * W2_old_aug.T
    W2_new_aug = np.linalg.solve(A, B).T  # (10, 129)
    
    W2_new = W2_new_aug[:, :128].astype(np.float32)
    b2_new = W2_new_aug[:, 128].astype(np.float32)
    
    # Update weights
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new
    
    # Diagnostics
    diagnostics = {
        "dead_units_replaced": int(D),
        "best_lambda": float(best_lambda),
        "cv_acc": float(best_cv_acc),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
