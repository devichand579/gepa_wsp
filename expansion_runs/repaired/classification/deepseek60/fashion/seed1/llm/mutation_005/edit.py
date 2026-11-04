import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current head parameters (used as prior)
    W0 = w["head.weight"].astype(np.float64)      # [10, 1568]
    b0 = w["head.bias"].astype(np.float64)        # [10]

    # Center features and targets to handle bias separately
    X_mean = X.mean(axis=0, keepdims=True)        # [1, D]
    Y_mean = Y.mean(axis=0, keepdims=True)        # [1, C]
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Precompute Gram matrix and cross-covariance
    XtX = Xc.T @ Xc                               # [D, D]
    XtY = Xc.T @ Yc                               # [D, C]

    # Candidate regularization strengths
    lambdas = np.logspace(-4, 4, 17)

    # Simple hold-out split: use first 1600 for train, last 400 for validation
    n_train = 1600
    X_train = Xc[:n_train]
    Y_train = Yc[:n_train]
    X_val = Xc[n_train:]
    Y_val = Y[n_train:]                           # original one-hot labels for val

    # Precompute for training split
    XtX_tr = X_train.T @ X_train
    XtY_tr = X_train.T @ Y_train

    best_lambda = None
    best_acc = -1.0

    for lam in lambdas:
        # Solve ridge: W = (XtX + lam I)^-1 (XtY + lam W0^T)  (note W0 is [C, D], so W0^T is [D, C])
        # We also regularize bias towards b0, but since we centered, we can solve for W and then compute bias.
        # For simplicity, we solve for W (C x D) using the centered data.
        # The solution for W is: W = (XtY_tr + lam * W0.T) @ inv(XtX_tr + lam * I)  ... careful with dimensions.
        # Let's use the standard form: minimize ||X W^T - Y||^2 + lam ||W - W0||^2.
        # Derivative wrt W: 2 (X W^T - Y)^T X + 2 lam (W - W0) = 0
        # => W (X^T X + lam I) = Y^T X + lam W0
        # => W = (Y^T X + lam W0) @ inv(X^T X + lam I)
        # Here X is [n, D], Y is [n, C], W is [C, D].
        # So Y^T X is [C, D]. W0 is [C, D].
        # So W = (Y_train.T @ X_train + lam * W0) @ np.linalg.inv(XtX_tr + lam * np.eye(D))
        # But XtX_tr is [D, D], so we need to solve for W.
        # We can use np.linalg.solve for each row? Actually W is [C, D]. The equation is:
        # W @ (XtX_tr + lam I) = Y_train.T @ X_train + lam W0
        # So W = (Y_train.T @ X_train + lam W0) @ inv(XtX_tr + lam I)
        # Let's compute the right-hand side: RHS = Y_train.T @ X_train + lam * W0  # [C, D]
        # Then W = RHS @ inv(XtX_tr + lam I). We can use np.linalg.solve for each row of RHS? 
        # Since (XtX_tr + lam I) is [D, D], we can solve for each row of W: W[i] = solve(XtX_tr + lam I, RHS[i]).
        # But RHS[i] is [D], so W[i] is [D]. That works.
        A = XtX_tr + lam * np.eye(D)
        RHS = Y_train.T @ X_train + lam * W0   # [C, D]
        # Solve A @ W.T = RHS.T  => W.T = solve(A, RHS.T)
        W = np.linalg.solve(A, RHS.T).T        # [C, D]

        # Compute bias: b = Y_mean - W @ X_mean.T  (since Y = X W^T + b, and we centered X, Y)
        # Actually Y_mean = X_mean @ W.T + b => b = Y_mean - X_mean @ W.T
        b = Y_mean.flatten() - (X_mean @ W.T).flatten()   # [C]

        # Evaluate on validation set
        logits_val = X_val @ W.T + b
        pred_val = np.argmax(logits_val, axis=1)
        acc = np.mean(pred_val == y[n_train:])
        if acc > best_acc:
            best_acc = acc
            best_lambda = lam

    # Refit on all data with best_lambda
    lam = best_lambda
    A = XtX + lam * np.eye(D)
    RHS = Yc.T @ Xc + lam * W0   # [C, D]
    W = np.linalg.solve(A, RHS.T).T   # [C, D]
    b = Y_mean.flatten() - (X_mean @ W.T).flatten()

    # Clip to safe range (absolute value <= 1000)
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Update weights
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "val_acc": float(best_acc),
        "W_mean": float(np.mean(W)),
        "W_std": float(np.std(W)),
        "b_mean": float(np.mean(b)),
        "b_std": float(np.std(b)),
    }
    return {"weights": w, "diagnostics": diagnostics}
