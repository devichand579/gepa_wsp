import numpy as np

def edit(w, f):
    # Extract full training pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    N, D = X.shape
    C = 10
    H_dim = 128

    # One-hot labels
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Class means
    class_means = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        class_means[c] = X[y == c].mean(axis=0)

    mean_all = X.mean(axis=0)

    # Candidate pool
    candidates = []

    # 1. Current fc1 weights
    W_cur = w["fc1.weight"].astype(np.float32)  # (128, 784)
    candidates.append(W_cur)

    # 2. Class means
    candidates.append(class_means)

    # 3. Pairwise class mean differences
    pairwise = []
    for i in range(C):
        for j in range(i + 1, C):
            pairwise.append(class_means[i] - class_means[j])
    candidates.append(np.array(pairwise, dtype=np.float32))

    # 4. PCA directions (top 128)
    X_centered = X - mean_all
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    candidates.append(Vt[:128].astype(np.float32))

    # 5. Random projections
    rng = np.random.RandomState(42)
    rand = rng.randn(128, D).astype(np.float32)
    rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)
    candidates.append(rand)

    all_cands = np.vstack(candidates)  # (num_cands, 784)
    num_cands = all_cands.shape[0]

    # Project data onto all candidates
    proj = X @ all_cands.T  # (2000, num_cands)

    # Score each candidate by ANOVA F-value of its ReLU activation
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    best_F = np.zeros(num_cands, dtype=np.float32)
    best_bias = np.zeros(num_cands, dtype=np.float32)

    for i in range(num_cands):
        p = proj[:, i]
        biases = -np.quantile(p, quantiles)
        best_F_i = -1.0
        best_bias_i = 0.0
        for b in biases:
            a = np.maximum(0, p + b)
            mean_a = a.mean()
            sum_a = np.bincount(y, weights=a, minlength=C)
            count_a = np.bincount(y, minlength=C)
            mean_c = sum_a / count_a
            ss_between = np.sum(count_a * (mean_c - mean_a) ** 2)
            ss_total = np.sum((a - mean_a) ** 2)
            ss_within = ss_total - ss_between
            if ss_within > 1e-8:
                F = (ss_between / (C - 1)) / (ss_within / (N - C))
            else:
                F = 0.0
            if F > best_F_i:
                best_F_i = F
                best_bias_i = b
        best_F[i] = best_F_i
        best_bias[i] = best_bias_i

    # Select top 128 candidates
    top_indices = np.argsort(best_F)[::-1][:H_dim]
    selected_w = all_cands[top_indices]      # (128, 784)
    selected_bias = best_bias[top_indices]   # (128,)

    # Compute new hidden features
    H = np.maximum(0, X @ selected_w.T + selected_bias)  # (2000, 128)

    # Fit fc2 via ridge regression with 5-fold CV to choose lambda
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])  # (2000, 129)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_lambda = 1e-3
    best_cv_acc = 0.0

    rng = np.random.RandomState(123)
    indices = np.arange(N)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    for lam in lambdas:
        cv_acc = 0.0
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_train = H_aug[train_idx]
            Y_train = Y[train_idx]
            H_val = H_aug[val_idx]
            Y_val = Y[val_idx]

            HtH = H_train.T @ H_train
            reg = lam * np.eye(H_aug.shape[1], dtype=np.float32)
            reg[-1, -1] = 0.0  # don't regularize bias
            HtY = H_train.T @ Y_train
            W = np.linalg.solve(HtH + reg, HtY)

            logits_val = H_val @ W
            pred = np.argmax(logits_val, axis=1)
            cv_acc += np.mean(pred == y[val_idx])
        cv_acc /= 5.0
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Final fit on all data
    HtH = H_aug.T @ H_aug
    reg = best_lambda * np.eye(H_aug.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    HtY = H_aug.T @ Y
    W_out = np.linalg.solve(HtH + reg, HtY)  # (129, 10)

    fc2_weight = W_out[:-1].T.astype(np.float32)  # (10, 128)
    fc2_bias = W_out[-1].astype(np.float32)       # (10,)

    # Update weights
    w["fc1.weight"] = selected_w.astype(np.float32)
    w["fc1.bias"] = selected_bias.astype(np.float32)
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"] = fc2_bias

    # Clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    # Diagnostics
    train_logits = H @ fc2_weight.T + fc2_bias
    train_pred = np.argmax(train_logits, axis=1)
    train_acc = float(np.mean(train_pred == y))

    diagnostics = {
        "train_accuracy": train_acc,
        "cv_accuracy": float(best_cv_acc),
        "best_lambda": float(best_lambda),
        "num_candidates": int(num_cands),
        "selected_F_mean": float(np.mean(best_F[top_indices])),
        "fc1_weight_mean": float(np.mean(selected_w)),
        "fc1_weight_std": float(np.std(selected_w)),
        "fc2_weight_mean": float(np.mean(fc2_weight)),
        "fc2_weight_std": float(np.std(fc2_weight)),
        "fc2_bias_mean": float(np.mean(fc2_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
