import numpy as np

def edit(w, f):
    # Extract full-pool features and labels
    X = f["features"].astype(np.float64)   # (4000, 32)
    y = f["y"].astype(np.int64)            # (4000,)

    # Split by class
    X0 = X[y == 0]
    X1 = X[y == 1]
    n0, n1 = len(X0), len(X1)

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    # Pooled within-class covariance
    X0c = X0 - mu0
    X1c = X1 - mu1
    Sigma = (X0c.T @ X0c + X1c.T @ X1c) / (n0 + n1 - 2)

    # Regularize to avoid overfitting on low-variance dims
    lambda_reg = 0.01
    Sigma += lambda_reg * np.eye(Sigma.shape[0])

    # LDA direction and intercept
    w_lda = np.linalg.solve(Sigma, mu1 - mu0)
    b_lda = -0.5 * (mu1 + mu0) @ w_lda

    # Scale so logit difference matches current logit scale
    scores = X @ w_lda + b_lda
    logits = f["logits"].astype(np.float64)
    logit_diff = logits[:, 1] - logits[:, 0]
    target_std = np.std(logit_diff)
    current_std = np.std(scores)
    if current_std > 1e-8:
        scale = target_std / (2.0 * current_std)
    else:
        scale = 1.0

    w_head = scale * w_lda
    b_head = scale * b_lda

    # Write new head weights (symmetric for binary classification)
    w["head.weight"] = np.zeros((2, 32), dtype=np.float32)
    w["head.weight"][1] = w_head.astype(np.float32)
    w["head.weight"][0] = (-w_head).astype(np.float32)

    w["head.bias"] = np.zeros(2, dtype=np.float32)
    w["head.bias"][1] = np.float32(b_head)
    w["head.bias"][0] = np.float32(-b_head)

    # Diagnostics
    diag = {
        "lda_scale": float(scale),
        "w_head_norm": float(np.linalg.norm(w_head)),
        "b_head": float(b_head),
        "train_accuracy": float(np.mean((scores > 0) == (y == 1))),
        "target_std": float(target_std),
        "current_std": float(current_std),
        "lambda_reg": float(lambda_reg)
    }

    return {"weights": w, "diagnostics": diag}
