import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [4000,64]
    Y = f["y"].astype(np.float32)        # [4000]
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- bag-of-words (excluding pad=0) ----
    mask = (X != 0).astype(np.float32)   # [N,64]
    BOW = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        m = mask[i]
        for t in range(L):
            if m[t] > 0:
                BOW[i, row[t]] += 1.0
    n = mask.sum(axis=1, keepdims=True)  # [N,1]
    n = np.maximum(n, 1.0)
    F = BOW / n                          # normalized BOW, [N,V]

    # ---- closed-form ridge linear classifier on F ----
    # target: +1 for positive (y=1), -1 for negative (y=0)
    t = 2.0 * Y - 1.0                    # [N]
    lam = 1.0
    # solve (F^T F + lam I) s = F^T t
    A = F.T @ F + lam * np.eye(V)
    b = F.T @ t
    s = np.linalg.solve(A, b)            # [V]
    # intercept
    c = float(np.mean(t - F @ s))

    # ---- build zero-mean / unit-variance embedding rows ----
    # column 0 = s (scaled so that LN identity holds), other columns are
    # deterministic orthogonal-ish fillers.
    rng = np.random.RandomState(12345)
    E = rng.randn(V, D).astype(np.float64)
    E[:, 0] = s
    # make each row zero-mean and unit-variance
    E = E - E.mean(axis=1, keepdims=True)
    std = E.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    E = E / std
    # re-center after scaling (std scaling preserves zero mean)
    E = E - E.mean(axis=1, keepdims=True)
    # ensure row 0 (pad) is exactly zero so padding contributes nothing
    E[0, :] = 0.0
    w["embed.weight"] = E.astype(np.float32)

    # ---- positional embedding: zero (keep it simple / stable) ----
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # ---- final LayerNorm: identity ----
    w["norm.weight"] = np.ones(D, dtype=np.float32)
    w["norm.bias"] = np.zeros(D, dtype=np.float32)

    # ---- head: read column 0, with fitted intercept ----
    hw = np.zeros((2, D), dtype=np.float32)
    hw[0, 0] = -1.0
    hw[1, 0] = 1.0
    w["head.weight"] = hw
    w["head.bias"] = np.array([-c, c], dtype=np.float32)

    # ---- zero out transformer blocks (they are already zero, but be safe) ----
    for name in list(w.keys()):
        if name.startswith("block0.") or name.startswith("block1."):
            w[name] = np.zeros_like(w[name], dtype=np.float32)
    # norm1/norm2 inside blocks: identity-ish (weights 1, bias 0) so that
    # if anything is ever turned on it behaves sanely.
    for b in ["block0", "block1"]:
        w[b + ".norm1.weight"] = np.ones(D, dtype=np.float32)
        w[b + ".norm1.bias"] = np.zeros(D, dtype=np.float32)
        w[b + ".norm2.weight"] = np.ones(D, dtype=np.float32)
        w[b + ".norm2.bias"] = np.zeros(D, dtype=np.float32)

    # ---- diagnostics ----
    # quick estimate of training accuracy on the fitted linear model
    logits = F @ s + c
    pred = (logits > 0).astype(np.float32)
    train_acc = float(np.mean(pred == Y))

    diag = {
        "method": "BOW ridge + LayerNorm-identity embedding",
        "lambda": lam,
        "train_accuracy_estimate": train_acc,
        "s_norm": float(np.linalg.norm(s)),
        "intercept": c,
        "embed_row_mean_maxabs": float(np.max(np.abs(w["embed.weight"].mean(axis=1)))),
        "embed_row_std_maxdev": float(np.max(np.abs(w["embed.weight"].std(axis=1) - 1.0))),
        "head_weight": w["head.weight"].tolist(),
        "head_bias": w["head.bias"].tolist(),
    }
    return {"weights": w, "diagnostics": diag}
