import numpy as np
import math

def edit(w, f):
    # Extract current weights
    W1 = w["fc1.weight"].copy().astype(np.float32)  # (128, 784)
    b1 = w["fc1.bias"].copy().astype(np.float32)    # (128,)
    W2 = w["fc2.weight"].copy().astype(np.float32)  # (10, 128)
    b2 = w["fc2.bias"].copy().astype(np.float32)    # (10,)

    # Feedback data
    X = f["x"].reshape(2000, -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                       # (2000,)
    H_old = f["features"].astype(np.float32)          # (2000, 128)

    # Identify dead ReLU units (always zero)
    max_act = H_old.max(axis=0)
    dead_idx = np.where(max_act == 0)[0]

    # Current predictions and confusion matrix
    logits_old = H_old @ W2.T + b2
    pred_old = np.argmax(logits_old, axis=1)
    cm = np.zeros((10, 10), dtype=int)
    for i in range(len(y)):
        cm[y[i], pred_old[i]] += 1

    # Top confused pairs (true c1, predicted c2)
    pairs = []
    for c1 in range(10):
        for c2 in range(10):
            if c1 != c2 and cm[c1, c2] > 0:
                pairs.append((cm[c1, c2], c1, c2))
    pairs.sort(reverse=True)

    # Class means of raw inputs
    mu = np.zeros((10, X.shape[1]), dtype=np.float32)
    for c in range(10):
        mu[c] = X[y == c].mean(axis=0)

    # Build new fc1 by replacing dead units with discriminative directions
    W1_new = W1.copy()
    b1_new = b1.copy()
    for i, idx in enumerate(dead_idx):
        if i < len(pairs):
            _, c1, c2 = pairs[i]
        else:
            _, c1, c2 = pairs[0] if pairs else (0, 0, 1)
        w_dir = mu[c1] - mu[c2]
        b_dir = -0.5 * (mu[c1] + mu[c2]) @ w_dir
        W1_new[idx] = w_dir
        b1_new[idx] = b_dir

    # Compute new features
    H_new = np.maximum(0, X @ W1_new.T + b1_new)

    # Helper: ridge regression fit
    def ridge_fit(H, y, alpha):
        N, D = H.shape
        H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])
        Y_onehot = np.zeros((N, 10), dtype=np.float32)
        Y_onehot[np.arange(N), y] = 1.0
        A = H_aug.T @ H_aug + alpha * np.eye(D + 1, dtype=np.float32)
        W_aug = np.linalg.solve(A, H_aug.T @ Y_onehot)
        W2_new = W_aug[:D].T
        b2_new = W_aug[D]
        return W2_new, b2_new

    # Helper: LDA fit
    def lda_fit(H, y, lam):
        N, D = H.shape
        mu_h = np.zeros((10, D), dtype=np.float32)
        for c in range(10):
            mu_h[c] = H[y == c].mean(axis=0)
        # Pooled covariance
        S = np.zeros((D, D), dtype=np.float32)
        for c in range(10):
            Hc = H[y == c] - mu_h[c]
            S += Hc.T @ Hc
        S /= (N - 10)
        S += lam * np.eye(D, dtype=np.float32)
        W_lda = np.linalg.solve(S, mu_h.T)  # (D, 10)
        b_lda = -0.5 * np.sum(mu_h * W_lda.T, axis=1)
        return W_lda.T, b_lda

    # Cross-validation to select method and hyperparameters
    np.random.seed(42)
    n_folds = 5
    indices = np.random.permutation(2000)
    fold_size = 2000 // n_folds

    alphas_ridge = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    lambdas_lda = [0.001, 0.01, 0.1, 1.0, 10.0]

    best_acc = -1.0
    best_method = None
    best_param = None
    best_use_new = False

    # Evaluate original features with ridge and LDA
    for use_new in [False, True]:
        H_use = H_new if use_new else H_old
        # Ridge
        for alpha in alphas_ridge:
            accs = []
            for k in range(n_folds):
                val_idx = indices[k*fold_size:(k+1)*fold_size]
                train_idx = np.setdiff1d(indices, val_idx)
                W2_cv, b2_cv = ridge_fit(H_use[train_idx], y[train_idx], alpha)
                logits_cv = H_use[val_idx] @ W2_cv.T + b2_cv
                pred_cv = np.argmax(logits_cv, axis=1)
                accs.append(np.mean(pred_cv == y[val_idx]))
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_method = "ridge"
                best_param = alpha
                best_use_new = use_new
        # LDA
        for lam in lambdas_lda:
            accs = []
            for k in range(n_folds):
                val_idx = indices[k*fold_size:(k+1)*fold_size]
                train_idx = np.setdiff1d(indices, val_idx)
                W2_cv, b2_cv = lda_fit(H_use[train_idx], y[train_idx], lam)
                logits_cv = H_use[val_idx] @ W2_cv.T + b2_cv
                pred_cv = np.argmax(logits_cv, axis=1)
                accs.append(np.mean(pred_cv == y[val_idx]))
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_method = "lda"
                best_param = lam
                best_use_new = use_new

    # Final fit on full data using best method
    H_final = H_new if best_use_new else H_old
    if best_method == "ridge":
        W2_final, b2_final = ridge_fit(H_final, y, best_param)
    else:
        W2_final, b2_final = lda_fit(H_final, y, best_param)

    # Update weights
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2_final.astype(np.float32)
    w["fc2.bias"] = b2_final.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "dead_units": int(len(dead_idx)),
        "best_method": best_method,
        "best_param": float(best_param),
        "best_use_new": bool(best_use_new),
        "cv_accuracy": float(best_acc),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_max_abs": float(np.max(np.abs(w["fc2.weight"]))),
        "fc2_bias_max_abs": float(np.max(np.abs(w["fc2.bias"]))),
    }

    return {"weights": w, "diagnostics": diagnostics}
