import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)   # (N,32) pooled LN features
    y = np.asarray(f["y"], dtype=np.int64)            # (N,)
    N, D = X.shape

    # ---- build regression targets: +1 for class1, -1 for class0 ----
    t = np.where(y == 1, 1.0, -1.0)

    # ---- standardize features (helps conditioning of normal equations) ----
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xs = (X - mu) / sd

    # ---- closed-form ridge with a small CV sweep over lambda ----
    # We solve (Xs^T Xs + lam I) a = Xs^T t ; intercept handled by centering t.
    tc = t - t.mean()
    G = Xs.T @ Xs                       # (D,D)
    rhs = Xs.T @ tc                     # (D,)
    best = None
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0]:
        A = G + lam * np.eye(D)
        try:
            a = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        # training accuracy (closed-form, no iteration)
        scores = Xs @ a
        pred = (scores > 0).astype(np.int64)
        acc = (pred == y).mean()
        # margin-aware score: prefer larger acc, then larger margin
        key = (acc, np.linalg.norm(a))
        if best is None or key[0] > best[0]:
            best = (acc, lam, a)
    acc, lam, a = best

    # ---- map back to original feature space ----
    # score = ((X - mu)/sd) @ a = X @ (a/sd) - (mu/sd)@a
    w_dir = (a / sd.ravel()).astype(np.float64)          # (D,)
    b_val = float((mu.ravel() / sd.ravel()) @ a)         # threshold offset

    # normalize direction to keep magnitudes moderate
    nrm = np.linalg.norm(w_dir)
    if nrm > 1e-8:
        scale = 1.0 / nrm
        w_dir = w_dir * scale
        b_val = b_val * scale

    # ---- write antisymmetric head: logits = [w·f - b, -(w·f - b)] ----
    W = np.zeros((2, D), dtype=np.float32)
    W[0] = w_dir.astype(np.float32)
    W[1] = -w_dir.astype(np.float32)
    bvec = np.array([-b_val, b_val], dtype=np.float32)

    w["head.weight"] = W
    w["head.bias"] = bvec

    # ---- diagnostics ----
    scores = X @ w_dir - b_val
    pred = (scores > 0).astype(np.int64)
    diag = {
        "lambda": float(lam),
        "train_acc": float((pred == y).mean()),
        "class_acc": {
            "0": float((pred[y == 0] == 0).mean()),
            "1": float((pred[y == 1] == 1).mean()),
        },
        "w_norm": float(np.linalg.norm(w_dir)),
        "b": float(b_val),
    }
    return {"weights": w, "diagnostics": diag}
