import numpy as np
import math

def edit(w, f):
    # Extract data
    X = f["x"].reshape(len(f["x"]), -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(int)
    H_orig = f["features"].astype(np.float64)               # (2000, 128)
    N, D = X.shape
    n_classes = 10

    # One-hot labels
    Y = np.zeros((N, n_classes))
    Y[np.arange(N), y] = 1.0

    # Center pixels
    X_mean = X.mean(axis=0)
    Xc = X - X_mean

    # ---------- Generate candidate linear features ----------
    candidates = []  # each: (values (N,), weight_vec (784,), bias (scalar))

    # Ridge regression with several alphas
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        # Augment with bias column
        X_aug = np.hstack([Xc, np.ones((N, 1))])
        # Solve (X_aug^T X_aug + alpha I) W = X_aug^T Y
        A = X_aug.T @ X_aug + alpha * np.eye(D + 1)
        B = X_aug.T @ Y
        W = np.linalg.solve(A, B)  # (785, 10)
        for c in range(n_classes):
            w_c = W[:D, c]
            b_c = W[D, c]
            s = Xc @ w_c + b_c
            # positive and negative parts
            candidates.append((np.maximum(s, 0.0), w_c, b_c))
            candidates.append((np.maximum(-s, 0.0), -w_c, -b_c))

    # LDA directions
    class_means = np.zeros((n_classes, D))
    for c in range(n_classes):
        class_means[c] = X[y == c].mean(axis=0)
    overall_mean = X.mean(axis=0)
    # Within-class scatter
    Sw = np.zeros((D, D))
    for c in range(n_classes):
        Xc_c = X[y == c] - class_means[c]
        Sw += Xc_c.T @ Xc_c
    # Between-class scatter
    Sb = np.zeros((D, D))
    for c in range(n_classes):
        diff = class_means[c] - overall_mean
        Sb += (y == c).sum() * np.outer(diff, diff)
    # Regularize Sw
    gamma = 1e-3 * np.trace(Sw) / D
    Sw_reg = Sw + gamma * np.eye(D)
    # Solve generalized eigenproblem: Sb v = lambda Sw_reg v
    # Compute eigenvectors of Sw_reg^{-1} Sb
    M = np.linalg.solve(Sw_reg, Sb)
    eigvals, eigvecs = np.linalg.eig(M)
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Take top 9 (real part)
    for i in range(min(9, eigvecs.shape[1])):
        v = np.real(eigvecs[:, i])
        v = v / (np.linalg.norm(v) + 1e-12)
        # Center the projection
        s = (X - overall_mean) @ v
        candidates.append((np.maximum(s, 0.0), v, -overall_mean @ v))
        candidates.append((np.maximum(-s, 0.0), -v, overall_mean @ v))

    # PCA directions (optional, add some diversity)
    # Compute PCA on Xc
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    for i in range(min(20, Vt.shape[0])):
        v = Vt[i]
        s = Xc @ v
        candidates.append((np.maximum(s, 0.0), v, 0.0))
        candidates.append((np.maximum(-s, 0.0), -v, 0.0))

    # Stack candidate values into a matrix (N, M)
    C = np.column_stack([c[0] for c in candidates])
    M = C.shape[1]

    # ---------- Rank candidate features using a linear model on [H_orig, C] ----------
    # Fit ridge on combined features to get importance
    H_comb = np.hstack([H_orig, C])
    # Standardize features? Not necessary for ridge, but we can just use ridge with small alpha
    alpha_sel = 1.0
    A_sel = H_comb.T @ H_comb + alpha_sel * np.eye(H_comb.shape[1])
    B_sel = H_comb.T @ Y
    W_sel = np.linalg.solve(A_sel, B_sel)  # (128+M, 10)
    # Importance of each candidate = norm of its weights across classes
    cand_weights = W_sel[128:, :]  # (M, 10)
    importance = np.linalg.norm(cand_weights, axis=1)
    # Rank candidates by importance descending
    ranked_idx = np.argsort(importance)[::-1]

    # ---------- Determine which original units to replace ----------
    # Use current fc2 weights to find least important original units
    fc2_w = w["fc2.weight"].astype(np.float64)  # (10, 128)
    orig_importance = np.linalg.norm(fc2_w, axis=0)
    orig_ranked = np.argsort(orig_importance)  # ascending: least important first

    # We will try several K values and pick the best via cross-validation
    # To keep runtime reasonable, we test K in [0, 20, 38, 60, 80, 100]
    K_candidates = [0, 20, 38, 60, 80, 100]
    best_acc = -1.0
    best_K = 0
    best_method = None
    best_params = None

    # Precompute LDA and ridge solvers for final classifier
    def fit_lda(H, y, shrinkage=0.1):
        # H: (n, d), y: (n,)
        n, d = H.shape
        classes = np.unique(y)
        n_classes = len(classes)
        means = np.zeros((n_classes, d))
        for i, c in enumerate(classes):
            means[i] = H[y == c].mean(axis=0)
        overall_mean = H.mean(axis=0)
        Sw = np.zeros((d, d))
        for i, c in enumerate(classes):
            Hc = H[y == c] - means[i]
            Sw += Hc.T @ Hc
        # Shrinkage
        Sw = (1 - shrinkage) * Sw + shrinkage * np.trace(Sw) / d * np.eye(d)
        # Solve for weights: for each class, w_c = Sw^{-1} mu_c
        # We can compute logits = H @ W^T + b
        W = np.linalg.solve(Sw, means.T).T  # (n_classes, d)
        b = -0.5 * np.sum(means * W, axis=1) + np.log(np.array([ (y==c).sum()/n for c in classes ]))
        return W, b

    def fit_ridge(H, y, alpha=1.0):
        n, d = H.shape
        Y = np.zeros((n, n_classes))
        Y[np.arange(n), y] = 1.0
        # Center H? We can just use H directly with bias
        H_aug = np.hstack([H, np.ones((n, 1))])
        A = H_aug.T @ H_aug + alpha * np.eye(d + 1)
        B = H_aug.T @ Y
        W = np.linalg.solve(A, B)  # (d+1, 10)
        W_final = W[:d, :].T  # (10, d)
        b_final = W[d, :]     # (10,)
        return W_final, b_final

    # Cross-validation setup
    n_folds = 5
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for K in K_candidates:
        if K > M:
            continue
        # Select top K candidate features
        sel_cand = ranked_idx[:K]
        # Select original units to keep: all except the K least important
        keep_orig = orig_ranked[K:]  # keep the more important ones
        # Build new feature matrix for all data
        # Order: first K new features, then kept original features
        H_new = np.zeros((N, 128))
        H_new[:, :K] = C[:, sel_cand]
        H_new[:, K:] = H_orig[:, keep_orig]
        # Now evaluate final classifier via CV
        for method in ['lda', 'ridge']:
            if method == 'lda':
                params_list = [0.0, 0.1, 0.3, 0.5]
            else:
                params_list = [0.1, 1.0, 10.0, 100.0]
            for param in params_list:
                accs = []
                for fold in folds:
                    train_idx = np.setdiff1d(indices, fold)
                    val_idx = fold
                    H_train = H_new[train_idx]
                    y_train = y[train_idx]
                    H_val = H_new[val_idx]
                    y_val = y[val_idx]
                    if method == 'lda':
                        W_cls, b_cls = fit_lda(H_train, y_train, shrinkage=param)
                    else:
                        W_cls, b_cls = fit_ridge(H_train, y_train, alpha=param)
                    logits = H_val @ W_cls.T + b_cls
                    pred = np.argmax(logits, axis=1)
                    acc = np.mean(pred == y_val)
                    accs.append(acc)
                mean_acc = np.mean(accs)
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_K = K
                    best_method = method
                    best_params = param

    # Now train final model on full data with best K, method, param
    K = best_K
    sel_cand = ranked_idx[:K]
    keep_orig = orig_ranked[K:]
    H_new = np.zeros((N, 128))
    H_new[:, :K] = C[:, sel_cand]
    H_new[:, K:] = H_orig[:, keep_orig]

    if best_method == 'lda':
        W_final, b_final = fit_lda(H_new, y, shrinkage=best_params)
    else:
        W_final, b_final = fit_ridge(H_new, y, alpha=best_params)

    # ---------- Update weights ----------
    # fc1: new units first, then kept original units
    fc1_w = w["fc1.weight"].astype(np.float64).copy()  # (128, 784)
    fc1_b = w["fc1.bias"].astype(np.float64).copy()    # (128,)
    # Create new fc1 weight and bias
    new_fc1_w = np.zeros((128, 784))
    new_fc1_b = np.zeros(128)
    # New units
    for i, cand_idx in enumerate(sel_cand):
        _, w_vec, b_val = candidates[cand_idx]
        new_fc1_w[i] = w_vec
        new_fc1_b[i] = b_val
    # Kept original units
    for i, orig_idx in enumerate(keep_orig):
        new_fc1_w[K + i] = fc1_w[orig_idx]
        new_fc1_b[K + i] = fc1_b[orig_idx]

    # fc2: fitted classifier
    new_fc2_w = W_final.astype(np.float64)  # (10, 128)
    new_fc2_b = b_final.astype(np.float64)  # (10,)

    # Clip to ensure finite and within bounds
    new_fc1_w = np.clip(new_fc1_w, -1000, 1000).astype(np.float32)
    new_fc1_b = np.clip(new_fc1_b, -1000, 1000).astype(np.float32)
    new_fc2_w = np.clip(new_fc2_w, -1000, 1000).astype(np.float32)
    new_fc2_b = np.clip(new_fc2_b, -1000, 1000).astype(np.float32)

    w["fc1.weight"] = new_fc1_w
    w["fc1.bias"] = new_fc1_b
    w["fc2.weight"] = new_fc2_w
    w["fc2.bias"] = new_fc2_b

    diagnostics = {
        "best_K": int(best_K),
        "best_method": best_method,
        "best_param": float(best_params),
        "cv_accuracy": float(best_acc),
        "fc1_weight_mean": float(new_fc1_w.mean()),
        "fc1_weight_std": float(new_fc1_w.std()),
        "fc2_weight_mean": float(new_fc2_w.mean()),
        "fc2_weight_std": float(new_fc2_w.std()),
        "fc2_bias_mean": float(new_fc2_b.mean()),
        "fc2_bias_std": float(new_fc2_b.std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
