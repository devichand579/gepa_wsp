import numpy as np

def edit(w, f):
    # ---- read feedback ----
    X = f["features"].astype(np.float64)   # (2000, 1568)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = X.shape
    C = 10

    # ---- one-hot targets ----
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # ---- center features and targets ----
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # ---- ridge regression (closed form) ----
    lam = 1.0
    XtX = Xc.T @ Xc
    XtX[np.diag_indices_from(XtX)] += lam
    W = np.linalg.solve(XtX, Xc.T @ Yc)    # (D, C)
    b = Y_mean - X_mean @ W                # (C,)

    # ---- write back to head ----
    w["head.weight"] = W.T.astype(np.float32)   # (10, 1568)
    w["head.bias"]   = b.astype(np.float32)     # (10,)

    # ---- diagnostics ----
    logits = X @ W + b
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    class_acc = {}
    for c in range(C):
        mask = (y == c)
        if mask.sum() > 0:
            class_acc[str(c)] = float((pred[mask] == c).mean())

    diag = {
        "train_acc": acc,
        "class_acc": class_acc,
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
    }
    return {"weights": w, "diagnostics": diag}
