import numpy as np
import math

def edit(w, f):
    # Extract data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    N = X.shape[0]
    C = 10

    # Current fc1 parameters
    W1 = w["fc1.weight"].astype(np.float64)  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float64)    # (128,)

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Helper: compute hidden features for a given bias shift
    def hidden_features(delta):
        b = b1 + delta
        H = X @ W1.T + b  # (N, 128)
        return np.maximum(H, 0.0)

    # Helper: fit fc2 via ridge regression with bias term
    def fit_fc2(H, alpha):
        # Augment with bias column
        Ha = np.concatenate([H, np.ones((N, 1))], axis=1)  # (N, 129)
        # Solve (Ha^T Ha + alpha I) W = Ha^T Y
        A = Ha.T @ Ha + alpha * np.eye(Ha.shape[1])
        B = Ha.T @ Y
        W_aug = np.linalg.solve(A, B)  # (129, 10)
        W2 = W_aug[:128, :].T  # (10, 128)
        b2 = W_aug[128, :]     # (10,)
        return W2, b2

    # Helper: evaluate accuracy of a given (W2, b2) on given H
    def accuracy(H, W2, b2):
        logits = H @ W2.T + b2
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    # Cross-validation to select delta and alpha
    # We'll use a simple 5-fold CV on the 2000 examples.
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    # Deterministic shuffle for reproducibility
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
    if n_folds * fold_size < N:
        folds[-1] = np.concatenate([folds[-1], indices[n_folds*fold_size:]])

    # Candidate bias shifts (relative to current b1)
    # Current biases are very negative for many units, so positive shifts revive them.
    deltas = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 30.0]
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    best_cv_acc = -1.0
    best_delta = 0.0
    best_alpha = 1e-2
    best_W2 = None
    best_b2 = None

    for delta in deltas:
        H_all = hidden_features(delta)
        for alpha in alphas:
            acc_sum = 0.0
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                H_train = H_all[train_idx]
                Y_train = Y[train_idx]
                H_val = H_all[val_idx]
                # Fit on train
                Ha = np.concatenate([H_train, np.ones((len(train_idx), 1))], axis=1)
                A = Ha.T @ Ha + alpha * np.eye(Ha.shape[1])
                B = Ha.T @ Y_train
                W_aug = np.linalg.solve(A, B)
                W2 = W_aug[:128, :].T
                b2 = W_aug[128, :]
                # Evaluate on val
                logits = H_val @ W2.T + b2
                pred = np.argmax(logits, axis=1)
                acc_sum += np.mean(pred == y[val_idx])
            cv_acc = acc_sum / n_folds
            if cv_acc > best_cv_acc:
                best_cv_acc = cv_acc
                best_delta = delta
                best_alpha = alpha

    # Refit on full data with best hyperparameters
    H_best = hidden_features(best_delta)
    best_W2, best_b2 = fit_fc2(H_best, best_alpha)

    # Update weights
    w["fc1.bias"] = (b1 + best_delta).astype(np.float32)
    w["fc2.weight"] = best_W2.astype(np.float32)
    w["fc2.bias"] = best_b2.astype(np.float32)

    # Diagnostics
    train_acc = accuracy(H_best, best_W2, best_b2)
    logits_train = H_best @ best_W2.T + best_b2
    # Softmax cross-entropy
    logits_train -= np.max(logits_train, axis=1, keepdims=True)
    probs = np.exp(logits_train)
    probs /= np.sum(probs, axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "best_delta": float(best_delta),
        "best_alpha": float(best_alpha),
        "cv_accuracy": float(best_cv_acc),
        "train_accuracy": float(train_acc),
        "train_cross_entropy": float(ce),
        "fc1_bias_mean": float(np.mean(w["fc1.bias"])),
        "fc1_bias_std": float(np.std(w["fc1.bias"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"]))
    }

    return {"weights": w, "diagnostics": diagnostics}
