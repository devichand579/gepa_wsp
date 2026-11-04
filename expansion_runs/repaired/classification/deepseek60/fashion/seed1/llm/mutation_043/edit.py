import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)          # [2000,1568]
    y = f["y"].astype(np.int64)                   # [2000]
    ids = f["batch.ids"].astype(np.int64)         # [128]
    Xb = X[ids]                                   # paired batch features
    yb = y[ids]

    n, d = X.shape
    C = 10

    # class means
    mu = np.zeros((C, d), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            mu[c] = X[mask].mean(axis=0)

    mu_all = X.mean(axis=0)

    # pooled covariance with shrinkage
    # compute centered features
    Xc = X - mu[y]
    # covariance = Xc.T @ Xc / n
    # use shrinkage to identity for stability
    # we can compute Sigma = (1/n) Xc.T Xc + lambda * I
    # but d=1568, n=2000, so we can compute it.
    # To save time, use a moderate lambda.
    lam = 1e-2
    # Compute Sigma efficiently: Xc.T @ Xc is 1568x1568, doable.
    Sigma = (Xc.T @ Xc) / n
    # add shrinkage
    Sigma += lam * np.eye(d, dtype=np.float64)

    # LDA weights: W = Sigma^{-1} (mu_c - mu_all)
    # We can solve Sigma * W^T = (mu - mu_all).T
    # mu - mu_all: [C, d]
    diff = mu - mu_all[None, :]
    # solve Sigma * W^T = diff^T
    # W^T is [d, C]
    Wt = np.linalg.solve(Sigma, diff.T)   # [d, C]
    W = Wt.T                              # [C, d]

    # bias: b_c = -0.5 * (mu_c + mu_all) . W_c + log(count_c/n)
    # Actually for LDA: b_c = -0.5 * (mu_c + mu_all) @ W_c + log(prior_c)
    # Let's use standard LDA formula:
    # logit_c = x . W_c + b_c
    # W_c = Sigma^{-1} (mu_c - mu_all)
    # b_c = -0.5 * (mu_c + mu_all) . W_c + log(prior_c)
    b = np.zeros(C, dtype=np.float64)
    for c in range(C):
        b[c] = -0.5 * (mu[c] + mu_all) @ W[c] + np.log(counts[c] / n)

    # Now scale W and b so that max abs weight = 1000
    max_abs = np.max(np.abs(W))
    if max_abs > 0:
        scale = 1000.0 / max_abs
    else:
        scale = 1.0
    W = W * scale
    b = b * scale

    # Clip to be safe
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["head.weight"] = W
    w["head.bias"] = b

    # Diagnostics
    logits = Xb @ W.T + b
    # stable softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    acc = (pred == yb).mean()
    ce = -np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean()

    diag = {
        "paired_batch_accuracy": float(acc),
        "paired_batch_cross_entropy": float(ce),
        "max_abs_weight": float(np.max(np.abs(W))),
        "max_abs_bias": float(np.max(np.abs(b))),
    }
    return {"weights": w, "diagnostics": diag}
