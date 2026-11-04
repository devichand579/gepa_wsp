import numpy as np

def edit(w, f):
    # Extract full training pool features and labels
    X = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = X.shape
    C = 10

    # Standardize features (handle zero-variance columns)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma

    # Class means
    means = np.zeros((C, D))
    priors = np.zeros(C)
    for c in range(C):
        mask = (y == c)
        priors[c] = mask.mean()
        means[c] = Xs[mask].mean(axis=0)

    # Pooled within-class covariance
    S = np.zeros((D, D))
    for c in range(C):
        Xc = Xs[y == c]
        Xc_centered = Xc - means[c]
        S += Xc_centered.T @ Xc_centered
    S /= (N - C)

    # Regularize (small ridge for numerical stability)
    alpha = 1e-3
    S_reg = S + alpha * np.eye(D)

    # LDA weights and bias
    W = np.linalg.solve(S_reg, means.T)          # (D, C)
    bias = -0.5 * np.sum(means * W.T, axis=1) + np.log(priors)

    # Convert back to original feature space
    fc2_weight = (W / sigma[:, None]).T          # (C, D)
    fc2_bias = bias - (mu / sigma) @ W           # (C,)

    # Write into weights
    w["fc2.weight"] = fc2_weight.astype(np.float32)
    w["fc2.bias"] = fc2_bias.astype(np.float32)

    # Diagnostics
    diag = {
        "method": "lda_standardized",
        "alpha": alpha,
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diag}
