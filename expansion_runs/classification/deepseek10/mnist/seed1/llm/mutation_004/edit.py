import numpy as np

def edit(w, f):
    # Extract arrays
    X_feat = f["features"].astype(np.float64)          # [2000,128] current hidden features
    y = f["y"].astype(np.int64)                        # [2000]
    x = f["x"].astype(np.float64)                      # [2000,1,28,28]
    N = x.shape[0]
    x_flat = x.reshape(N, -1)                          # [2000,784]

    # Current fc2 weight to determine least important hidden units
    W2_old = w["fc2.weight"].astype(np.float64)        # [10,128]
    col_norms = np.linalg.norm(W2_old, axis=0)         # [128]
    # Choose 10 units with smallest column norms to replace
    replace_idx = np.argsort(col_norms)[:10]

    # Compute class mean images
    class_means = []
    for c in range(10):
        mask = (y == c)
        if np.sum(mask) == 0:
            mean_c = np.zeros(784, dtype=np.float64)
        else:
            mean_c = x_flat[mask].mean(axis=0)
        class_means.append(mean_c)
    class_means = np.stack(class_means, axis=0)        # [10,784]

    # Compute new features: ReLU(x_flat @ mean_c) for each class
    X_cm = np.zeros((N, 10), dtype=np.float64)
    for c in range(10):
        X_cm[:, c] = np.maximum(0.0, x_flat @ class_means[c])

    # Build new feature matrix: replace chosen columns with class-mean features
    X_new = X_feat.copy()
    for i, idx in enumerate(replace_idx):
        X_new[:, idx] = X_cm[:, i]

    # Standardize features for ridge fitting
    mu = X_new.mean(axis=0)
    sigma = X_new.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    X_std = (X_new - mu) / sigma

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Add bias column
    X_aug = np.concatenate([X_std, np.ones((N, 1))], axis=1)  # [N,129]

    # Choose ridge lambda via generalized cross-validation (GCV)
    # Precompute SVD of X_aug for efficiency
    U, s, Vt = np.linalg.svd(X_aug, full_matrices=False)
    s2 = s ** 2
    # GCV for ridge: minimize ||Y - X W||^2 / (1 - trace(H)/N)^2
    # trace(H) = sum(s2 / (s2 + lambda))
    # We can compute GCV for a range of lambdas
    lambdas = np.logspace(-4, 4, 50)
    best_lambda = 1.0
    best_gcv = np.inf
    for lam in lambdas:
        # Hat matrix trace
        trace_H = np.sum(s2 / (s2 + lam))
        # Residual sum of squares
        # W = V diag(s/(s2+lam)) U^T Y
        # Y_pred = U diag(s2/(s2+lam)) U^T Y
        # RSS = ||Y - Y_pred||^2
        # Compute efficiently using SVD
        UT_Y = U.T @ Y
        # Y_pred = U @ (s2/(s2+lam))[:,None] * UT_Y
        # RSS = ||Y||^2 - sum( (2*s2/(s2+lam) - (s2/(s2+lam))^2) * (UT_Y)^2 )
        # Simpler: compute Y_pred and RSS
        # Since N=2000, 10 targets, this is fast.
        coef = s2 / (s2 + lam)
        Y_pred = U @ (coef[:, None] * UT_Y)
        rss = np.sum((Y - Y_pred) ** 2)
        gcv = (rss / N) / ((1.0 - trace_H / N) ** 2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lambda = lam

    lam = best_lambda

    # Solve ridge normal equations: (X^T X + lam I) W = X^T Y
    # Using the augmented matrix, we want to fit W_aug [129,10]
    # X_aug^T X_aug + lam I (except bias maybe not regularized? We'll regularize all for simplicity)
    # Actually, we should not regularize the bias term. Let's separate bias.
    X_std_no_bias = X_std
    XtX = X_std_no_bias.T @ X_std_no_bias
    XtY = X_std_no_bias.T @ Y
    # Add small regularization to XtX
    XtX_reg = XtX + lam * np.eye(XtX.shape[0])
    # Solve for weights (without bias)
    W_no_bias = np.linalg.solve(XtX_reg, XtY)  # [128,10]
    # Compute bias to make mean prediction match mean target
    # We want X_std @ W_no_bias + b = Y. The mean of X_std is 0, so b = mean(Y) = 1/10.
    # But better: b = mean(Y - X_std @ W_no_bias, axis=0)
    b = (Y - X_std @ W_no_bias).mean(axis=0)  # [10]

    # W_no_bias is [128,10], we need [10,128]
    W2_new = W_no_bias.T
    b2_new = b

    # Update weights
    w["fc2.weight"] = W2_new.astype(np.float32)
    w["fc2.bias"] = b2_new.astype(np.float32)

    # Update fc1 weights for replaced units
    for i, idx in enumerate(replace_idx):
        w["fc1.weight"][idx] = class_means[i].astype(np.float32)
        w["fc1.bias"][idx] = 0.0

    # Diagnostics
    # Compute training accuracy with new weights on the pool
    logits_new = X_new @ W2_new.T + b2_new
    preds = np.argmax(logits_new, axis=1)
    acc = np.mean(preds == y)
    diagnostics = {
        "replaced_indices": replace_idx.tolist(),
        "best_lambda": float(best_lambda),
        "train_accuracy": float(acc),
        "fc2_weight_norm": float(np.linalg.norm(W2_new)),
        "fc2_bias_norm": float(np.linalg.norm(b2_new)),
    }

    return {"weights": w, "diagnostics": diagnostics}
