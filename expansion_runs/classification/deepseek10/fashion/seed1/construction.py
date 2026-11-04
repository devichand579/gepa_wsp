import numpy as np

def edit(w, f):
    # Extract full training pool features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.eye(C)[y]  # (2000, 10)

    # Center features and targets for intercept handling
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Ridge regression closed-form solution
    lam = 1e-2
    XtX = Xc.T @ Xc
    XtY = Xc.T @ Yc
    W = np.linalg.solve(XtX + lam * np.eye(D), XtY)  # (D, 10)
    b = Y_mean - X_mean @ W                          # (10,)

    # Update head weights (shape [10, 1568])
    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Clip to safe range
    w["head.weight"] = np.clip(w["head.weight"], -1000, 1000)
    w["head.bias"] = np.clip(w["head.bias"], -1000, 1000)

    # Diagnostics
    logits = X @ W + b
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)

    orig_logits = f["logits"]
    orig_preds = np.argmax(orig_logits, axis=1)
    orig_acc = np.mean(orig_preds == y)

    diagnostics = {
        "method": "ridge",
        "lambda": lam,
        "train_accuracy": float(acc),
        "original_accuracy": float(orig_acc),
        "head_weight_mean": float(w["head.weight"].mean()),
        "head_weight_std": float(w["head.weight"].std()),
        "head_bias_mean": float(w["head.bias"].mean()),
        "head_bias_std": float(w["head.bias"].std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
