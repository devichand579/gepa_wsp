def edit(w, f):
    import numpy as np

    # Extract final hidden features and labels from the permitted feedback pool.
    H = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    n, d = H.shape

    # One-hot targets for the 10 MNIST classes.
    Y = np.zeros((n, 10), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Standardize features so that ridge regularization acts uniformly.
    mean = H.mean(axis=0)
    std = H.std(axis=0)
    std[std < 1e-6] = 1.0
    H_std = (H - mean) / std

    # Augment with a bias column.
    X = np.hstack([H_std, np.ones((n, 1))])  # (2000, 129)
    XtX = X.T @ X
    XtY = X.T @ Y
    I = np.eye(d + 1)

    # Select ridge penalty by generalized cross-validation (closed-form).
    lambdas = np.logspace(-6, 2, 50)
    best_gcv = np.inf
    best_beta = None
    for lam in lambdas:
        A = XtX + lam * I
        beta = np.linalg.solve(A, XtY)
        pred = X @ beta
        resid = Y - pred
        rss = np.sum(resid ** 2)
        df = np.trace(np.linalg.solve(A, XtX))
        gcv = (rss / n) / (1.0 - df / n) ** 2
        if gcv < best_gcv:
            best_gcv = gcv
            best_beta = beta

    # Convert standardized-space coefficients back to original feature space.
    w_std = best_beta[:-1]          # (128, 10)
    b_std = best_beta[-1]           # (10,)
    w_orig = w_std / std[:, None]   # (128, 10)
    b_orig = b_std - np.sum(w_std * mean[:, None] / std[:, None], axis=0)

    # Write the new fc2 weights and biases.
    w["fc2.weight"] = w_orig.T.astype(np.float32)  # (10, 128)
    w["fc2.bias"] = b_orig.astype(np.float32)      # (10,)

    # Keep values finite and within the allowed bound.
    w["fc2.weight"] = np.clip(w["fc2.weight"], -1000.0, 1000.0)
    w["fc2.bias"] = np.clip(w["fc2.bias"], -1000.0, 1000.0)

    return w
