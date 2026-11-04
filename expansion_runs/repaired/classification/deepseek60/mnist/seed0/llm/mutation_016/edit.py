import numpy as np

def edit(w, f):
    # Read current weights
    fc1_w = w["fc1.weight"]
    fc1_b = w["fc1.bias"]
    fc2_w = w["fc2.weight"].copy()
    fc2_b = w["fc2.bias"].copy()

    # Full feedback pool
    X = f["x"]          # [2000,1,28,28]
    y = f["y"]          # [2000]
    H = f["features"]   # [2000,128] current hidden features (after ReLU)

    N, D = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Add bias column to features
    H_aug = np.concatenate([H, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, D+1]

    # Ridge regression: solve (H_aug^T H_aug + lambda I) W = H_aug^T Y
    # We'll choose lambda by 5-fold cross-validation.
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)

    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lambda = None
    best_cv_acc = -1.0

    for lam in lambdas:
        cv_acc = 0.0
        for i in range(5):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_tr = H_aug[tr_idx]
            Y_tr = Y[tr_idx]
            H_val = H_aug[val_idx]
            y_val = y[val_idx]

            A = H_tr.T @ H_tr + lam * np.eye(D + 1, dtype=np.float32)
            B = H_tr.T @ Y_tr
            W = np.linalg.solve(A, B)  # [D+1, C]

            logits_val = H_val @ W
            pred_val = np.argmax(logits_val, axis=1)
            cv_acc += np.mean(pred_val == y_val)
        cv_acc /= 5.0
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Also evaluate current fc2 by the same CV scheme
    W_cur = np.concatenate([fc2_w.T, fc2_b.reshape(1, -1)], axis=0)  # [D+1, C]
    cv_acc_cur = 0.0
    for i in range(5):
        val_idx = folds[i]
        H_val = H_aug[val_idx]
        y_val = y[val_idx]
        logits_val = H_val @ W_cur
        pred_val = np.argmax(logits_val, axis=1)
        cv_acc_cur += np.mean(pred_val == y_val)
    cv_acc_cur /= 5.0

    # Fit final ridge on all data with best_lambda
    A = H_aug.T @ H_aug + best_lambda * np.eye(D + 1, dtype=np.float32)
    B = H_aug.T @ Y
    W_new = np.linalg.solve(A, B)  # [D+1, C]

    # Decide whether to replace fc2
    if best_cv_acc >= cv_acc_cur:
        fc2_w = W_new[:D, :].T.astype(np.float32)
        fc2_b = W_new[D, :].astype(np.float32)
        used_new = True
    else:
        used_new = False

    # Clip to safe range
    fc2_w = np.clip(fc2_w, -1000.0, 1000.0).astype(np.float32)
    fc2_b = np.clip(fc2_b, -1000.0, 1000.0).astype(np.float32)

    w["fc2.weight"] = fc2_w
    w["fc2.bias"] = fc2_b

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_acc_new": float(best_cv_acc),
        "cv_acc_cur": float(cv_acc_cur),
        "used_new_fc2": bool(used_new),
        "fc2_weight_norm": float(np.linalg.norm(fc2_w)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_b)),
    }
    return {"weights": w, "diagnostics": diagnostics}
