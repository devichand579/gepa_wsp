import numpy as np
import math

def edit(w, f):
    # Extract feedback arrays
    X = f["features"].astype(np.float64)   # (2000, 1568)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = X.shape
    C = 10

    # Compute class means
    mu = np.zeros((C, D), dtype=np.float64)
    priors = np.zeros(C, dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        mu[c] = X[idx].mean(axis=0)
        priors[c] = idx.sum() / N

    # Compute pooled within-class covariance
    Sigma = np.zeros((D, D), dtype=np.float64)
    for i in range(N):
        diff = X[i] - mu[y[i]]
        Sigma += np.outer(diff, diff)
    Sigma /= (N - C)

    # Add tiny ridge for numerical stability (Sigma is full rank here, but safe)
    Sigma += 1e-6 * np.eye(D, dtype=np.float64)

    # Solve Sigma * W^T = mu^T  =>  W = mu @ Sigma^{-1}
    # Use Cholesky or solve for stability
    try:
        L = np.linalg.cholesky(Sigma)
        # Solve Sigma * Z = mu.T
        Z = np.linalg.solve(L.T, np.linalg.solve(L, mu.T))  # (D, C)
        W = Z.T  # (C, D)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)
        W = mu @ Sigma_inv

    # Bias: -0.5 * mu_c^T Sigma^{-1} mu_c + log(prior_c)
    # mu_c^T Sigma^{-1} mu_c = mu_c^T W_c^T = sum(mu_c * W_c)
    bias = -0.5 * np.sum(mu * W, axis=1) + np.log(priors)

    # Compute logits on full feedback
    logits = X @ W.T + bias  # (N, C)

    # Temperature scaling: find T minimizing cross-entropy
    # Grid search over T
    best_T = 1.0
    best_ce = np.inf
    # Use a reasonable range; logits may be large
    logits_std = logits.std()
    T_grid = np.logspace(-2, 2, 200) * max(logits_std, 1e-6)
    for T in T_grid:
        scaled = logits / T
        # Stable softmax
        max_logits = scaled.max(axis=1, keepdims=True)
        exp_logits = np.exp(scaled - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Cross-entropy
        ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        if ce < best_ce:
            best_ce = ce
            best_T = T

    # Apply temperature scaling
    W_scaled = W / best_T
    bias_scaled = bias / best_T

    # Write back to w
    w["head.weight"] = W_scaled.astype(np.float32)
    w["head.bias"] = bias_scaled.astype(np.float32)

    # Diagnostics
    # Compute training accuracy and CE after scaling
    scaled_logits = X @ W_scaled.T + bias_scaled
    max_logits = scaled_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = np.argmax(scaled_logits, axis=1)
    train_acc = np.mean(preds == y)
    train_ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "method": "LDA + temperature scaling",
        "best_T": float(best_T),
        "train_accuracy": float(train_acc),
        "train_cross_entropy": float(train_ce),
        "head_weight_mean": float(w["head.weight"].mean()),
        "head_weight_std": float(w["head.weight"].std()),
        "head_bias": w["head.bias"].tolist(),
    }

    return {"weights": w, "diagnostics": diagnostics}
