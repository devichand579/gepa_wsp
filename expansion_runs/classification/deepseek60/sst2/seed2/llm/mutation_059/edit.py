import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)   # [4000, 32]
    y = f["y"].astype(np.int64)            # [4000]
    N, D = X.shape

    # Build regression targets: +1 for positive, -1 for negative
    t = np.where(y == 1, 1.0, -1.0)

    # Add bias column
    Xb = np.concatenate([X, np.ones((N, 1))], axis=1)  # [4000, 33]

    # Candidate ridge lambdas (log-spaced)
    lambdas = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])

    # 5-fold cross-validation to select lambda
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)

    best_lambda = lambdas[0]
    best_cv_acc = -1.0

    for lam in lambdas:
        accs = []
        for k in range(5):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            Xtr, ttr = Xb[tr_idx], t[tr_idx]
            Xval, tval = Xb[val_idx], t[val_idx]

            # Ridge normal equations: (X^T X + lam I) beta = X^T t
            A = Xtr.T @ Xtr + lam * np.eye(D + 1)
            b = Xtr.T @ ttr
            beta = np.linalg.solve(A, b)

            pred = np.sign(Xval @ beta)
            acc = np.mean(pred == tval)
            accs.append(acc)
        cv_acc = np.mean(accs)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Final fit on all data with selected lambda
    A = Xb.T @ Xb + best_lambda * np.eye(D + 1)
    b = Xb.T @ t
    beta = np.linalg.solve(A, b)

    # beta has shape [33]: first 32 are weights, last is bias
    w_head = beta[:D]   # [32]
    b_head = beta[D]    # scalar

    # Convert to 2-class head: class 0 = negative, class 1 = positive
    # logits = [ - (w_head·x + b_head),  + (w_head·x + b_head) ]
    # This matches the existing head structure (row1 = -row0, bias1 = -bias0)
    w["head.weight"][0, :] = -w_head.astype(np.float32)
    w["head.weight"][1, :] =  w_head.astype(np.float32)
    w["head.bias"][0] = -b_head.astype(np.float32)
    w["head.bias"][1] =  b_head.astype(np.float32)

    # Ensure all other parameters remain finite and within bounds
    for k in w:
        w[k] = np.clip(w[k].astype(np.float32), -1000.0, 1000.0)

    diagnostics = {
        "selected_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "head_weight_norm": float(np.linalg.norm(w_head)),
        "head_bias": float(b_head),
        "in_sample_accuracy": float(np.mean(np.sign(Xb @ beta) == t)),
    }

    return {"weights": w, "diagnostics": diagnostics}
