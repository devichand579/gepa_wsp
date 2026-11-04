import numpy as np
import math

def edit(w, f):
    # Extract feedback features and labels
    H = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current fc2 parameters as prior
    W2_cur = w["fc2.weight"].astype(np.float64)   # (10, 128)
    b2_cur = w["fc2.bias"].astype(np.float64)     # (10,)

    # Augment features with bias column
    H_aug = np.concatenate([H, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N, 129)
    W_aug_cur = np.concatenate([W2_cur, b2_cur[:, None]], axis=1)           # (10, 129)

    # 5-fold cross-validation to choose ridge strength (trust in prior)
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    folds = np.array_split(indices, 5)

    lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_lambda = 0.0
    best_cv_acc = -1.0

    for lam in lambdas:
        accs = []
        for k in range(5):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            H_tr = H_aug[train_idx]
            Y_tr = Y[train_idx]
            H_va = H_aug[val_idx]
            y_va = y[val_idx]

            # Ridge with prior towards current weights
            A = H_tr.T @ H_tr + lam * np.eye(D + 1)
            B = H_tr.T @ Y_tr + lam * W_aug_cur.T   # (129, 10)
            W_aug = np.linalg.solve(A, B).T         # (10, 129)

            logits = H_va @ W_aug.T
            pred = np.argmax(logits, axis=1)
            accs.append(np.mean(pred == y_va))
        cv_acc = np.mean(accs)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Final fit on all data using best lambda
    lam = best_lambda
    A = H_aug.T @ H_aug + lam * np.eye(D + 1)
    B = H_aug.T @ Y + lam * W_aug_cur.T
    W_aug = np.linalg.solve(A, B).T  # (10, 129)

    # Update weights
    w["fc2.weight"] = W_aug[:, :D].astype(np.float32)
    w["fc2.bias"] = W_aug[:, D].astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "best_cv_accuracy": float(best_cv_acc),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
