import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X_img = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)                                     # (2000,)
    N = X_img.shape[0]
    num_classes = 10

    # Current fc1 and fc2 parameters
    W1_old = w["fc1.weight"].astype(np.float64)  # (128, 784)
    b1_old = w["fc1.bias"].astype(np.float64)    # (128,)
    W2_old = w["fc2.weight"].astype(np.float64)  # (10, 128)
    b2_old = w["fc2.bias"].astype(np.float64)    # (10,)

    # Helper: compute ReLU features from images using given fc1
    def compute_features(W1, b1):
        H = X_img @ W1.T + b1
        return np.maximum(H, 0.0)

    # Helper: fit ridge on features and return W2, b2 in original feature space
    def fit_ridge(X_feat, y, alpha):
        # Standardize features
        mean = X_feat.mean(axis=0)
        std = X_feat.std(axis=0) + 1e-6
        Z = (X_feat - mean) / std
        Z_aug = np.hstack([Z, np.ones((Z.shape[0], 1))])
        Y_onehot = np.eye(num_classes)[y]
        # Ridge closed form
        A = Z_aug.T @ Z_aug + alpha * np.eye(Z_aug.shape[1])
        B = Z_aug.T @ Y_onehot
        Wz = np.linalg.solve(A, B)  # (129, 10)
        # Transform back to original feature space
        W_new = (Wz[:128, :].T) / std[None, :]          # (10, 128)
        b_new = Wz[128, :] - (mean / std) @ Wz[:128, :] # (10,)
        return W_new, b_new

    # Helper: evaluate CV accuracy for a given feature matrix and hyperparams
    def cv_score(X_feat, y, alphas, blends, n_folds=5):
        # Current logits on these features using old fc2
        logits_old = X_feat @ W2_old.T + b2_old  # (N, 10)
        # Shuffle indices for CV
        rng = np.random.RandomState(42)
        idx = rng.permutation(N)
        folds = np.array_split(idx, n_folds)
        best_acc = -1.0
        best_params = None
        for alpha in alphas:
            for blend in blends:
                accs = []
                for i in range(n_folds):
                    val_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                    X_tr = X_feat[train_idx]
                    y_tr = y[train_idx]
                    X_val = X_feat[val_idx]
                    y_val = y[val_idx]
                    W_new, b_new = fit_ridge(X_tr, y_tr, alpha)
                    logits_new = X_val @ W_new.T + b_new
                    logits_blend = blend * logits_new + (1.0 - blend) * logits_old[val_idx]
                    pred = np.argmax(logits_blend, axis=1)
                    accs.append(np.mean(pred == y_val))
                mean_acc = np.mean(accs)
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_params = (alpha, blend)
        return best_acc, best_params

    # Candidate 1: original fc1 features
    X_feat_orig = compute_features(W1_old, b1_old)

    # Candidate 2: repair dead units in fc1
    # Identify dead units: columns of X_feat_orig that are all zero
    dead_mask = np.all(X_feat_orig == 0.0, axis=0)
    num_dead = int(dead_mask.sum())
    X_feat_repaired = X_feat_orig.copy()
    W1_repaired = W1_old.copy()
    b1_repaired = b1_old.copy()
    if num_dead > 0:
        # Compute class means and global mean on images
        global_mean = X_img.mean(axis=0)  # (784,)
        class_means = np.zeros((num_classes, 784))
        for c in range(num_classes):
            class_means[c] = X_img[y == c].mean(axis=0)
        # For each dead unit, assign to a class in round-robin and set filter
        dead_indices = np.where(dead_mask)[0]
        for k, unit_idx in enumerate(dead_indices):
            c = k % num_classes
            w_vec = class_means[c] - global_mean
            # Normalize to have similar norm as average existing fc1 row
            avg_norm = np.mean(np.linalg.norm(W1_old, axis=1))
            w_norm = np.linalg.norm(w_vec)
            if w_norm > 1e-8:
                w_vec = w_vec * (avg_norm / w_norm)
            b_val = -np.dot(w_vec, global_mean)
            W1_repaired[unit_idx] = w_vec
            b1_repaired[unit_idx] = b_val
        X_feat_repaired = compute_features(W1_repaired, b1_repaired)

    # Hyperparameter grids
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    blends = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Evaluate both candidates
    acc_orig, params_orig = cv_score(X_feat_orig, y, alphas, blends)
    acc_rep, params_rep = cv_score(X_feat_repaired, y, alphas, blends)

    # Choose the better candidate
    if acc_rep > acc_orig:
        X_feat_final = X_feat_repaired
        W1_final = W1_repaired
        b1_final = b1_repaired
        best_acc = acc_rep
        best_params = params_rep
        chosen = "repaired"
    else:
        X_feat_final = X_feat_orig
        W1_final = W1_old
        b1_final = b1_old
        best_acc = acc_orig
        best_params = params_orig
        chosen = "original"

    alpha_best, blend_best = best_params

    # Fit final fc2 on the full pool using the best alpha
    W2_new, b2_new = fit_ridge(X_feat_final, y, alpha_best)

    # Blend with current fc2
    W2_final = blend_best * W2_new + (1.0 - blend_best) * W2_old
    b2_final = blend_best * b2_new + (1.0 - blend_best) * b2_old

    # Update weights in-place
    w["fc1.weight"] = W1_final.astype(np.float32)
    w["fc1.bias"] = b1_final.astype(np.float32)
    w["fc2.weight"] = W2_final.astype(np.float32)
    w["fc2.bias"] = b2_final.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "chosen": chosen,
        "num_dead_units": int(num_dead),
        "best_alpha": float(alpha_best),
        "best_blend": float(blend_best),
        "cv_acc_original": float(acc_orig),
        "cv_acc_repaired": float(acc_rep),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_mean": float(np.mean(w["fc2.weight"])),
        "fc2_weight_std": float(np.std(w["fc2.weight"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
