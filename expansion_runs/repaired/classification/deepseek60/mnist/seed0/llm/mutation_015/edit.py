import numpy as np

def edit(w, f):
    # Extract current weights
    fc1_w = w["fc1.weight"].copy()
    fc1_b = w["fc1.bias"].copy()
    fc2_w = w["fc2.weight"].copy()
    fc2_b = w["fc2.bias"].copy()

    # Full feedback pool
    X = f["x"]                     # [2000,1,28,28]
    y = f["y"]                     # [2000]
    H = f["features"]              # [2000,128] current hidden features
    N, D = H.shape

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Helper: ridge least-squares fit for fc2 given features H
    def fit_fc2(H, Y, lam):
        # Add bias column
        Hb = np.concatenate([H, np.ones((H.shape[0], 1), dtype=np.float32)], axis=1)
        # Solve (Hb^T Hb + lam I) W = Hb^T Y
        A = Hb.T @ Hb
        A += lam * np.eye(A.shape[0], dtype=np.float32)
        B = Hb.T @ Y
        W = np.linalg.solve(A, B)   # [129, 10]
        # Extract weight and bias (transpose to [10,128] and [10])
        W_out = W[:-1, :].T         # [10,128]
        b_out = W[-1, :]            # [10]
        return W_out.astype(np.float32), b_out.astype(np.float32)

    # Evaluate current fc2 on full pool
    logits_cur = H @ fc2_w.T + fc2_b
    pred_cur = np.argmax(logits_cur, axis=1)
    acc_cur = np.mean(pred_cur == y)

    # Try a few ridge penalties via 5-fold CV
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lam = None
    best_cv_acc = -1.0

    # 5-fold CV indices
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    for lam in lambdas:
        cv_acc = 0.0
        for i in range(5):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_tr, Y_tr = H[tr_idx], Y[tr_idx]
            H_val, y_val = H[val_idx], y[val_idx]
            W_new, b_new = fit_fc2(H_tr, Y_tr, lam)
            logits_val = H_val @ W_new.T + b_new
            pred_val = np.argmax(logits_val, axis=1)
            cv_acc += np.mean(pred_val == y_val)
        cv_acc /= 5.0
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lam = lam

    # Fit on full pool with best lambda
    W_new, b_new = fit_fc2(H, Y, best_lam)

    # Evaluate new fc2 on full pool
    logits_new = H @ W_new.T + b_new
    pred_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(pred_new == y)

    # Only replace if new is at least as accurate on the full pool
    if acc_new >= acc_cur:
        w["fc2.weight"] = W_new
        w["fc2.bias"] = b_new
        used_new = True
    else:
        used_new = False

    # Diagnostics
    diagnostics = {
        "acc_cur_full": float(acc_cur),
        "acc_new_full": float(acc_new),
        "best_lambda": float(best_lam),
        "best_cv_acc": float(best_cv_acc),
        "used_new_fc2": bool(used_new),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
