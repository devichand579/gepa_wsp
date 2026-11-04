import numpy as np
import math

def edit(w, f):
    # ---------- extract data ----------
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)   # (2000, 784)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    num_classes = 10
    Y = np.zeros((N, num_classes), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # ---------- current fc1 ----------
    fc1_w = w["fc1.weight"].astype(np.float64).copy()   # (128, 784)
    fc1_b = w["fc1.bias"].astype(np.float64).copy()     # (128,)

    # ---------- build class-mean templates for the first 10 units ----------
    mean_all = X.mean(axis=0)                            # (784,)
    templates = np.zeros((num_classes, 784), dtype=np.float64)
    biases = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mean_c = X[y == c].mean(axis=0)
        w_c = mean_c - mean_all
        b_c = -np.dot(mean_all, w_c)
        templates[c] = w_c
        biases[c] = b_c

    fc1_w_rev = fc1_w.copy()
    fc1_b_rev = fc1_b.copy()
    fc1_w_rev[:num_classes] = templates
    fc1_b_rev[:num_classes] = biases

    # ---------- compute features for both scenarios ----------
    def compute_features(fc1_w, fc1_b):
        H = X @ fc1_w.T + fc1_b
        return np.maximum(0.0, H)

    H_orig = f["features"].astype(np.float64)            # (2000, 128)
    H_rev = compute_features(fc1_w_rev, fc1_b_rev)       # (2000, 128)

    # ---------- ridge fitting ----------
    def fit_ridge(H, Y, lam):
        N, D = H.shape
        H_aug = np.hstack([H, np.ones((N, 1))])
        reg = lam * np.eye(D + 1)
        reg[-1, -1] = 0.0                                # do not regularise bias
        A = H_aug.T @ H_aug + reg
        B = H_aug.T @ Y
        W = np.linalg.solve(A, B)
        return W[:-1].T, W[-1]                           # weight (10, D), bias (10,)

    # ---------- LDA fitting (optional, used for comparison) ----------
    def fit_lda(H, y, num_classes=10):
        N, D = H.shape
        means = np.zeros((num_classes, D))
        priors = np.zeros(num_classes)
        for c in range(num_classes):
            idx = (y == c)
            means[c] = H[idx].mean(axis=0)
            priors[c] = idx.mean()
        cov = np.zeros((D, D))
        for c in range(num_classes):
            idx = (y == c)
            diff = H[idx] - means[c]
            cov += diff.T @ diff
        cov /= (N - num_classes)
        cov += 1e-3 * np.eye(D)                          # regularise covariance
        inv_cov = np.linalg.inv(cov)
        W = np.zeros((num_classes, D))
        b = np.zeros(num_classes)
        for c in range(num_classes):
            W[c] = inv_cov @ means[c]
            b[c] = -0.5 * means[c] @ inv_cov @ means[c] + math.log(priors[c] + 1e-12)
        return W, b

    # ---------- cross-validation to choose feature set and model ----------
    def cv_score(H, y, model_type, lam=None):
        # 5-fold CV
        n_folds = 5
        indices = np.arange(N)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, n_folds)
        accs = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            H_train, y_train = H[train_idx], y[train_idx]
            H_val, y_val = H[val_idx], y[val_idx]
            if model_type == 'ridge':
                W, b = fit_ridge(H_train, Y[train_idx], lam)
            else:  # lda
                W, b = fit_lda(H_train, y_train)
            logits = H_val @ W.T + b
            preds = np.argmax(logits, axis=1)
            accs.append(np.mean(preds == y_val))
        return np.mean(accs)

    # Candidate models
    candidates = []
    # original features + ridge
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        acc = cv_score(H_orig, y, 'ridge', lam)
        candidates.append(('orig_ridge', H_orig, 'ridge', lam, acc))
    # original features + LDA
    acc = cv_score(H_orig, y, 'lda')
    candidates.append(('orig_lda', H_orig, 'lda', None, acc))
    # revived features + ridge
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        acc = cv_score(H_rev, y, 'ridge', lam)
        candidates.append(('rev_ridge', H_rev, 'ridge', lam, acc))
    # revived features + LDA
    acc = cv_score(H_rev, y, 'lda')
    candidates.append(('rev_lda', H_rev, 'lda', None, acc))

    # pick best by CV accuracy
    best = max(candidates, key=lambda x: x[4])
    name, H_best, model_type, lam, best_acc = best

    # ---------- fit best model on all data ----------
    if model_type == 'ridge':
        W, b = fit_ridge(H_best, Y, lam)
    else:
        W, b = fit_lda(H_best, y)

    # ---------- temperature scaling to minimise cross-entropy ----------
    logits_train = H_best @ W.T + b
    # grid search for temperature T in [0.1, 10]
    best_T = 1.0
    best_ce = float('inf')
    for T in np.linspace(0.1, 10.0, 100):
        scaled = logits_train / T
        # softmax
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        if ce < best_ce:
            best_ce = ce
            best_T = T
    # apply temperature
    W = W / best_T
    b = b / best_T

    # ---------- write back ----------
    if name.startswith('rev'):
        w["fc1.weight"][:num_classes] = fc1_w_rev[:num_classes].astype(np.float32)
        w["fc1.bias"][:num_classes] = fc1_b_rev[:num_classes].astype(np.float32)
    w["fc2.weight"] = W.astype(np.float32)
    w["fc2.bias"] = b.astype(np.float32)

    # ---------- diagnostics ----------
    diag = {
        "selected_model": name,
        "cv_accuracy": float(best_acc),
        "temperature": float(best_T),
        "train_cross_entropy": float(best_ce),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diag}
