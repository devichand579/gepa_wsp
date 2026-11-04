import numpy as np

def edit(w, f):
    # Read current parameters
    W1 = w["fc1.weight"].astype(np.float64)  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float64)    # (128,)
    W2 = w["fc2.weight"].astype(np.float64)  # (10, 128)
    b2 = w["fc2.bias"].astype(np.float64)    # (10,)

    # Full feedback pool
    x = f["x"].astype(np.float64)            # (2000, 1, 28, 28)
    y = f["y"].astype(np.int64)              # (2000,)
    N = x.shape[0]
    x_flat = x.reshape(N, -1)                # (2000, 784)

    # Compute current pre-activations and ReLU features
    pre = x_flat @ W1.T + b1                 # (2000, 128)
    H = np.maximum(pre, 0.0)                 # (2000, 128)

    # Re-center each hidden unit: subtract its mean pre-activation from b1
    mean_pre = pre.mean(axis=0)              # (128,)
    b1_new = b1 - mean_pre
    pre_new = pre - mean_pre
    H_new = np.maximum(pre_new, 0.0)         # (2000, 128)

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Center features and targets for ridge with unregularized bias
    H_mean = H_new.mean(axis=0, keepdims=True)   # (1, 128)
    Y_mean = Y.mean(axis=0, keepdims=True)       # (1, 10)
    Hc = H_new - H_mean
    Yc = Y - Y_mean

    # Candidate ridge lambdas (closed-form fits only)
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lambda = None
    best_cv_acc = -1.0
    best_W2 = None
    best_b2 = None

    # 5-fold cross-validation on the full feedback pool
    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(0)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    for lam in lambdas:
        cv_acc = 0.0
        for k in range(n_folds):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])

            Hc_tr = Hc[tr_idx]
            Yc_tr = Yc[tr_idx]
            Hc_val = Hc[val_idx]
            Y_val = Y[val_idx]

            # Ridge solution: W2^T = (Hc^T Hc + lam I)^-1 Hc^T Yc
            A = Hc_tr.T @ Hc_tr + lam * np.eye(Hc_tr.shape[1])
            B = Hc_tr.T @ Yc_tr
            W2_tr = np.linalg.solve(A, B).T          # (10, 128)
            b2_tr = Y_mean[0] - W2_tr @ H_mean[0]    # (10,)

            logits_val = Hc_val @ W2_tr.T + b2_tr
            pred_val = np.argmax(logits_val, axis=1)
            cv_acc += np.mean(pred_val == y[val_idx])

        cv_acc /= n_folds
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Final fit on all data with best lambda
    A = Hc.T @ Hc + best_lambda * np.eye(Hc.shape[1])
    B = Hc.T @ Yc
    W2_new = np.linalg.solve(A, B).T                 # (10, 128)
    b2_new = Y_mean[0] - W2_new @ H_mean[0]          # (10,)

    # Write back as float32, preserving shapes
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2_new.astype(np.float32)
    w["fc2.bias"] = b2_new.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "best_cv_acc": float(best_cv_acc),
        "mean_pre_shift_norm": float(np.linalg.norm(mean_pre)),
        "fc2_weight_norm": float(np.linalg.norm(W2_new)),
        "fc2_bias_norm": float(np.linalg.norm(b2_new)),
        "hidden_zero_fraction_new": float(np.mean(H_new == 0.0)),
    }
    return {"weights": w, "diagnostics": diagnostics}
