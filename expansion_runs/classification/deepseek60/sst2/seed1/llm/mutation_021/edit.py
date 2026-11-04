import numpy as np

def edit(w, f):
    F = np.asarray(f["features"], dtype=np.float64)      # [4000,32] cached pooled features
    y = np.asarray(f["y"], dtype=np.int64)               # [4000]
    N, D = F.shape

    # ---- closed-form ridge-regularized least squares on +/-1 targets ----
    t = np.where(y == 1, 1.0, -1.0)
    # center features to decouple the bias from the direction
    mu = F.mean(axis=0, keepdims=True)
    Fc = F - mu
    # choose lambda from the data scale (no search loop): lambda ~ trace(Cov)/D
    cov = (Fc.T @ Fc) / N
    lam = float(np.trace(cov) / D) * 1e-2
    lam = max(lam, 1e-6)
    A = Fc.T @ Fc + lam * np.eye(D)
    w_dir = np.linalg.solve(A, Fc.T @ t)                 # [32]
    # bias so that the decision boundary is at the target mean
    b = -float(w_dir @ mu.ravel())

    # scale so logits are well conditioned (argmax unchanged)
    s = float(np.sqrt(np.mean((Fc @ w_dir) ** 2))) + 1e-9
    w_dir = w_dir / s
    b = b / s

    # realize in head: logit1 - logit0 = w_dir . f + b
    w["head.weight"] = np.vstack([-0.5 * w_dir, 0.5 * w_dir]).astype(np.float32)
    w["head.bias"] = np.array([-0.5 * b, 0.5 * b], dtype=np.float32)

    # ---- self-check with our own forward pass on the cached features ----
    logits = F @ w["head.weight"].T.astype(np.float64) + w["head.bias"].astype(np.float64)
    pred = np.argmax(logits, axis=1)
    acc = float((pred == y).mean())
    ce = float(-np.mean(np.log(np.clip(
        np.exp(logits - logits.max(axis=1, keepdims=True)) /
        np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True),
        1e-12, 1.0)[np.arange(N), y])))

    diag = {
        "train_accuracy": acc,
        "train_ce": ce,
        "lambda": lam,
        "w_norm": float(np.linalg.norm(w_dir)),
        "bias": float(b),
        "n_samples": int(N),
        "n_features": int(D),
    }
    return {"weights": w, "diagnostics": diag}
