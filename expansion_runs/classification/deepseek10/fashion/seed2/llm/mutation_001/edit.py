import numpy as np

def edit(w, f):
    # Extract features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # One-hot labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment with bias column
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N, D+1)
    D_aug = D + 1

    # 5-fold cross-validation to select ridge lambda
    lambdas = [0.01, 0.1, 1.0, 10.0]
    best_lambda = 1.0
    best_cv_acc = -1.0
    cv_results = {}

    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_train = X_aug[train_idx]
            Y_train = Y[train_idx]
            X_val = X_aug[val_idx]
            y_val = y[val_idx]

            reg = np.eye(D_aug, dtype=np.float64) * lam
            reg[-1, -1] = 0.0  # do not regularize bias

            A = X_train.T @ X_train + reg
            B = X_train.T @ Y_train
            W_cv = np.linalg.solve(A, B)

            logits_val = X_val @ W_cv
            preds_val = logits_val.argmax(axis=1)
            acc = (preds_val == y_val).mean()
            accs.append(acc)

        mean_acc = float(np.mean(accs))
        cv_results[lam] = mean_acc
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Final ridge fit on all data
    reg = np.eye(D_aug, dtype=np.float64) * best_lambda
    reg[-1, -1] = 0.0
    A = X_aug.T @ X_aug + reg
    B = X_aug.T @ Y
    W = np.linalg.solve(A, B)                     # (D+1, C)

    head_weight = W[:D, :].T                      # (C, D)
    head_bias = W[D, :]                           # (C,)

    # Update weights
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # Evaluate on full pool
    logits_new = X @ head_weight.T + head_bias
    logits_new -= logits_new.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_new)
    probs_new = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = probs_new.argmax(axis=1)
    acc = (preds == y).mean()
    ce = -np.log(probs_new[np.arange(N), y] + 1e-12).mean()

    # Evaluate on paired batch if available
    batch_ids = f.get("batch.ids")
    if batch_ids is not None:
        batch_ids = batch_ids.astype(np.int64)
        X_batch = X[batch_ids]
        y_batch = y[batch_ids]
        logits_batch = X_batch @ head_weight.T + head_bias
        logits_batch -= logits_batch.max(axis=1, keepdims=True)
        exp_batch = np.exp(logits_batch)
        probs_batch = exp_batch / exp_batch.sum(axis=1, keepdims=True)
        preds_batch = probs_batch.argmax(axis=1)
        acc_batch = (preds_batch == y_batch).mean()
        ce_batch = -np.log(probs_batch[np.arange(len(batch_ids)), y_batch] + 1e-12).mean()
    else:
        acc_batch = None
        ce_batch = None

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "cv_results": {str(k): v for k, v in cv_results.items()},
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "batch_accuracy": float(acc_batch) if acc_batch is not None else None,
        "batch_cross_entropy": float(ce_batch) if ce_batch is not None else None,
        "head_weight_norm": float(np.linalg.norm(head_weight)),
        "head_bias_norm": float(np.linalg.norm(head_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
