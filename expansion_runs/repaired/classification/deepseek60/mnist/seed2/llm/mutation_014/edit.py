import numpy as np

def edit(w, f):
    # Convert to float64 for stability
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    C = w["fc2.weight"].shape[0]
    D = w["fc1.weight"].shape[0]
    
    W1 = w["fc1.weight"].astype(np.float64)
    b1 = w["fc1.bias"].astype(np.float64)
    
    # Compute pre-activations
    z = X @ W1.T + b1
    mean_z = z.mean(axis=0)
    std_z = z.std(axis=0) + 1e-6
    
    # Candidate transforms
    transforms = []
    # 1. Center only
    b1_center = b1 - mean_z
    transforms.append(("center", W1, b1_center))
    # 2. Center and standardize
    W1_std = W1 / std_z[:, None]
    b1_std = (b1 - mean_z) / std_z
    transforms.append(("center_std", W1_std, b1_std))
    
    best_cv_acc = -1.0
    best_transform_name = None
    best_alpha = None
    best_feat_std = None
    best_W1 = None
    best_b1 = None
    
    K = 5
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    feat_std_options = [False, True]
    
    # For reproducibility
    rng = np.random.RandomState(42)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = np.array_split(indices, K)
    
    for name, W1_t, b1_t in transforms:
        z_t = X @ W1_t.T + b1_t
        H = np.maximum(z_t, 0)
        
        for alpha in alphas:
            for feat_std in feat_std_options:
                accs = []
                for i in range(K):
                    val_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(K) if j != i])
                    H_train = H[train_idx]
                    y_train = y[train_idx]
                    H_val = H[val_idx]
                    y_val = y[val_idx]
                    
                    if feat_std:
                        mu = H_train.mean(axis=0)
                        sigma = H_train.std(axis=0) + 1e-6
                        H_train = (H_train - mu) / sigma
                        H_val = (H_val - mu) / sigma
                    
                    # LDA fit
                    means = np.zeros((C, H_train.shape[1]))
                    priors = np.zeros(C)
                    for c in range(C):
                        mask = (y_train == c)
                        if mask.sum() > 0:
                            means[c] = H_train[mask].mean(axis=0)
                            priors[c] = mask.sum() / len(y_train)
                        else:
                            means[c] = 0
                            priors[c] = 1.0 / C
                    
                    S = np.zeros((H_train.shape[1], H_train.shape[1]))
                    for c in range(C):
                        mask = (y_train == c)
                        if mask.sum() > 1:
                            diff = H_train[mask] - means[c]
                            S += diff.T @ diff
                    dof = len(y_train) - C
                    if dof > 0:
                        Sigma = S / dof
                    else:
                        Sigma = np.eye(H_train.shape[1])
                    trace = np.trace(Sigma)
                    Sigma_shrunk = (1 - alpha) * Sigma + alpha * (trace / H_train.shape[1]) * np.eye(H_train.shape[1])
                    try:
                        inv_Sigma = np.linalg.inv(Sigma_shrunk)
                    except np.linalg.LinAlgError:
                        inv_Sigma = np.linalg.pinv(Sigma_shrunk)
                    
                    W = inv_Sigma @ means.T  # (D, C)
                    b = -0.5 * np.sum(means * W.T, axis=1) + np.log(priors + 1e-12)
                    scores = H_val @ W + b
                    pred = np.argmax(scores, axis=1)
                    acc = np.mean(pred == y_val)
                    accs.append(acc)
                mean_acc = np.mean(accs)
                if mean_acc > best_cv_acc:
                    best_cv_acc = mean_acc
                    best_transform_name = name
                    best_alpha = alpha
                    best_feat_std = feat_std
                    best_W1 = W1_t
                    best_b1 = b1_t
    
    # Refit on full data with best params
    W1_final = best_W1
    b1_final = best_b1
    z_final = X @ W1_final.T + b1_final
    H_final = np.maximum(z_final, 0)
    
    if best_feat_std:
        mu = H_final.mean(axis=0)
        sigma = H_final.std(axis=0) + 1e-6
        H_final = (H_final - mu) / sigma
    else:
        mu = None
        sigma = None
    
    # LDA on full data
    means = np.zeros((C, H_final.shape[1]))
    priors = np.zeros(C)
    for c in range(C):
        mask = (y == c)
        means[c] = H_final[mask].mean(axis=0)
        priors[c] = mask.sum() / N
    S = np.zeros((H_final.shape[1], H_final.shape[1]))
    for c in range(C):
        mask = (y == c)
        diff = H_final[mask] - means[c]
        S += diff.T @ diff
    dof = N - C
    Sigma = S / dof
    trace = np.trace(Sigma)
    Sigma_shrunk = (1 - best_alpha) * Sigma + best_alpha * (trace / H_final.shape[1]) * np.eye(H_final.shape[1])
    inv_Sigma = np.linalg.inv(Sigma_shrunk)
    W = inv_Sigma @ means.T  # (D, C)
    b = -0.5 * np.sum(means * W.T, axis=1) + np.log(priors + 1e-12)
    
    W2 = W.T  # (C, D)
    b2 = b  # (C,)
    
    # If features were standardized for LDA, convert back to original feature space
    if best_feat_std:
        W_std = W2.copy()
        W2 = W_std / sigma[None, :]
        b2 = b2 - (mu / sigma) @ W_std.T
    
    # Assign to w
    w["fc1.weight"] = W1_final.astype(np.float32)
    w["fc1.bias"] = b1_final.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)
    
    # Clip to ensure constraints
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    # Diagnostics
    diag = {
        "best_transform": best_transform_name,
        "best_alpha": best_alpha,
        "best_feat_std": best_feat_std,
        "best_cv_acc": float(best_cv_acc),
        "fc1_bias_mean": float(w["fc1.bias"].mean()),
        "fc1_bias_std": float(w["fc1.bias"].std()),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diag}
