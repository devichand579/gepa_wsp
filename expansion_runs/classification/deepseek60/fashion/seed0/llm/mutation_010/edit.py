import numpy as np

def edit(w, f):
    # Work on copies so we can safely return a new dict.
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}

    X = np.asarray(f["features"], dtype=np.float32)      # [N,1568]
    y = np.asarray(f["y"], dtype=np.int64)               # [N]
    N, D = X.shape
    C = 10

    # One-hot targets.
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Standardize features for numerical stability of the ridge solve.
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    Xs = (X - mu) / sd

    # Add bias as an extra column.
    Xb = np.concatenate([Xs, np.ones((N, 1), dtype=np.float32)], axis=1)

    # Ridge closed form: W = (X^T X + lambda I)^-1 X^T Y.
    # Use a modest lambda; the feedback pool is small relative to D.
    lam = 1e-3
    A = Xb.T @ Xb
    A += lam * np.eye(D + 1, dtype=np.float32)
    B = Xb.T @ Y
    Wb = np.linalg.solve(A, B)  # [(D+1), C]

    # Convert back to original feature space:
    # logits = X @ W + b
    #        = ((X-mu)/sd) @ Ws + b
    #        = X @ (Ws/sd) + (b - mu @ (Ws/sd))
    Ws = Wb[:D, :].T.astype(np.float32)       # [C,D]
    bs = Wb[D, :].astype(np.float32)          # [C]
    W_orig = Ws / sd.reshape(1, D)
    b_orig = bs - (mu.reshape(1, D) @ W_orig.T).reshape(C)

    # Optional mild scaling to sharpen probabilities without changing argmax.
    # Choose scale by simple hold-out on the feedback pool.
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    n_val = max(1, N // 5)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    def fit_head(tr, lam_):
        Xt = Xb[tr]
        Yt = Y[tr]
        A_ = Xt.T @ Xt + lam_ * np.eye(D + 1, dtype=np.float32)
        B_ = Xt.T @ Yt
        Wb_ = np.linalg.solve(A_, B_)
        Ws_ = Wb_[:D, :].T.astype(np.float32)
        bs_ = Wb_[D, :].astype(np.float32)
        W_o = Ws_ / sd.reshape(1, D)
        b_o = bs_ - (mu.reshape(1, D) @ W_o.T).reshape(C)
        return W_o, b_o

    best = None
    for lam_ in [1e-4, 1e-3, 1e-2, 1e-1]:
        W_o, b_o = fit_head(tr_idx, lam_)
        logits_val = X[val_idx] @ W_o.T + b_o.reshape(1, C)
        for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            lv = logits_val * scale
            # Stable CE.
            lv = lv - lv.max(axis=1, keepdims=True)
            p = np.exp(lv)
            p /= p.sum(axis=1, keepdims=True)
            ce = -np.log(np.maximum(p[np.arange(len(val_idx)), y[val_idx]], 1e-12)).mean()
            acc = (lv.argmax(axis=1) == y[val_idx]).mean()
            # Prefer accuracy, then CE.
            key = (acc, -ce)
            if best is None or key > best[0]:
                best = (key, lam_, scale, W_o, b_o)

    _, best_lam, best_scale, W_o, b_o = best

    # Refit on all data with the selected lambda, then apply selected scale.
    W_o, b_o = fit_head(np.arange(N), best_lam)
    W_o = (W_o * best_scale).astype(np.float32)
    b_o = (b_o * best_scale).astype(np.float32)

    # Clip to the allowed range.
    W_o = np.clip(W_o, -1000.0, 1000.0).astype(np.float32)
    b_o = np.clip(b_o, -1000.0, 1000.0).astype(np.float32)

    w["head.weight"] = W_o
    w["head.bias"] = b_o

    # Keep conv layers unchanged.
    for k in ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"]:
        w[k] = np.asarray(w[k], dtype=np.float32)

    # Diagnostics.
    logits = X @ W_o.T + b_o.reshape(1, C)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean())
    ce = float(-np.log(np.maximum(probs[np.arange(N), y], 1e-12)).mean())

    diagnostics = {
        "head_ridge_lambda": float(best_lam),
        "head_scale": float(best_scale),
        "feedback_accuracy": acc,
        "feedback_cross_entropy": ce,
        "head_weight_norm": float(np.linalg.norm(W_o)),
        "head_bias_norm": float(np.linalg.norm(b_o)),
        "head_weight_max_abs": float(np.max(np.abs(W_o))),
        "head_bias_max_abs": float(np.max(np.abs(b_o))),
    }

    return {"weights": w, "diagnostics": diagnostics}
