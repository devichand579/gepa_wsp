import numpy as np

def edit(w, f):
    # Extract full-pool features and labels
    X = f["features"].astype(np.float64)   # [4000, 32]
    y = f["y"].astype(np.int64)            # [4000]
    n, d = X.shape

    # Convert labels to ±1
    t = np.where(y == 1, 1.0, -1.0)

    # Compute class means and pooled covariance
    X0 = X[y == 0]
    X1 = X[y == 1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    diff = mu1 - mu0

    # Pooled within-class covariance
    C0 = np.cov(X0, rowvar=False, bias=True)
    C1 = np.cov(X1, rowvar=False, bias=True)
    C = 0.5 * (C0 + C1)

    # Shrinkage for numerical stability
    shrinkage = 0.15
    C_shrunk = (1.0 - shrinkage) * C + shrinkage * np.eye(d) * np.trace(C) / d

    # LDA direction
    w_dir = np.linalg.solve(C_shrunk, diff)

    # Normalize direction to avoid extreme logits
    w_dir = w_dir / (np.linalg.norm(w_dir) + 1e-12)

    # Threshold: midpoint of projected class means
    proj0 = X0 @ w_dir
    proj1 = X1 @ w_dir
    thr = -0.5 * (proj0.mean() + proj1.mean())

    # Realize in head: logit1 - logit0 = 2 * (w_dir @ x) + 2*thr
    # Set head.weight[1] = w_dir, head.weight[0] = -w_dir
    # Set head.bias[1] = thr, head.bias[0] = -thr
    w["head.weight"][0] = -w_dir.astype(np.float32)
    w["head.weight"][1] =  w_dir.astype(np.float32)
    w["head.bias"][0] = -np.float32(thr)
    w["head.bias"][1] =  np.float32(thr)

    # Diagnostics
    logits = X @ w_dir + thr
    preds = (logits > 0).astype(np.int64)
    acc = (preds == y).mean()

    diagnostics = {
        "lda_shrinkage": shrinkage,
        "w_dir_norm": float(np.linalg.norm(w_dir)),
        "threshold": float(thr),
        "train_accuracy": float(acc),
        "feature_std_min": float(np.std(X, axis=0).min()),
        "feature_std_max": float(np.std(X, axis=0).max()),
    }

    return {"weights": w, "diagnostics": diagnostics}
