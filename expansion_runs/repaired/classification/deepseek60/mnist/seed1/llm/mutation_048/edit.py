import numpy as np

def edit(w, f):
    # Copy weights so we do not mutate the input dict
    w = {k: v.copy() for k, v in w.items()}

    # Extract full feedback pool
    X = f["features"].astype(np.float64)          # (2000, 128) current ReLU features
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Standardise features to improve conditioning
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xs = (X - mu) / sd

    # Add bias column
    Xb = np.concatenate([Xs, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N, D+1)

    # Ridge regression: solve (Xb^T Xb + lambda I) W = Xb^T Y
    lam = 1e-3
    A = Xb.T @ Xb + lam * np.eye(D + 1, dtype=np.float64)
    B = Xb.T @ Y
    W = np.linalg.solve(A, B)                     # (D+1, C)

    # Convert back to original feature scale
    # logits = Xs @ W[:D] + W[D]
    #        = (X - mu)/sd @ W[:D] + W[D]
    #        = X @ (W[:D]/sd) + (W[D] - mu @ (W[:D]/sd))
    W_feat = W[:D] / sd.T                         # (D, C)
    b_feat = W[D] - (mu @ W_feat).ravel()         # (C,)

    # fc2.weight is [out, in] = [C, D]
    w["fc2.weight"] = W_feat.T.astype(np.float32)
    w["fc2.bias"]   = b_feat.astype(np.float32)

    # Diagnostics
    logits_new = X @ w["fc2.weight"].T + w["fc2.bias"]
    pred = logits_new.argmax(axis=1)
    acc = float((pred == y).mean())
    ce = float(-np.mean(np.log(np.exp(logits_new - logits_new.max(axis=1, keepdims=True)) /
                               np.exp(logits_new - logits_new.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))[
                               np.arange(N), y] + 1e-12))

    diag = {
        "train_acc": acc,
        "train_ce": ce,
        "lambda": lam,
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diag}
