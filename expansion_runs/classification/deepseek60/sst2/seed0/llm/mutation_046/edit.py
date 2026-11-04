import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]       # 32

    # ---- length-normalised bag-of-words counts -------------------------------
    mask = (X != 0).astype(np.float32)               # [N,64]
    length = np.maximum(mask.sum(1, keepdims=True), 1.0)
    B = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    np.add.at(B, (rows, cols), 1.0)
    B /= length
    # drop PAD column (always zero) to keep things clean
    B = B[:, 1:]
    Vf = B.shape[1]

    t = np.where(y == 1, 1.0, -1.0)

    # ---- closed-form ridge with K-fold CV over lambda ------------------------
    def ridge_fit(Btr, ttr, lam):
        A = Btr.T @ Btr + lam * np.eye(Vf)
        b = Btr.T @ ttr
        return np.linalg.solve(A, b)

    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    K = 5
    folds = np.array_split(perm, K)
    lambdas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    best_lam, best_acc = lambdas[0], -1.0
    for lam in lambdas:
        accs = []
        for k in range(K):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(K) if j != k])
            wv = ridge_fit(B[tr], t[tr], lam)
            pred = (B[va] @ wv) > 0
            accs.append(np.mean(pred == (t[va] > 0)))
        a = float(np.mean(accs))
        if a > best_acc:
            best_acc, best_lam = a, lam

    wv = ridge_fit(B, t, best_lam)                    # [Vf]
    # intercept
    bias = float(np.mean(t - B @ wv))

    # ---- build a well-conditioned embedding ---------------------------------
    # direction d in R^D : use coord 0 for the signal, coords 1..D-1 tiny
    d = np.zeros(D, dtype=np.float64)
    d[0] = 1.0
    # small orthogonal ripple so LayerNorm has variance in other coords
    ripple = 0.02
    d[1:] = ripple / np.sqrt(D - 1)

    # token weights: 0 for PAD, wv for tokens 1..V-1
    tw = np.zeros(V, dtype=np.float64)
    tw[1:] = wv

    # scale so that embed values stay modest and LayerNorm is near-identity
    scale = 1.0
    E = np.outer(tw, d) * scale                        # [V, D]

    # centre embeddings so mean over tokens ~ 0 (helps LayerNorm)
    E = E - E.mean(axis=0, keepdims=True)

    # ---- head reads the same direction --------------------------------------
    # pooled = mean_tokens E[token] ; want logit1-logit0 = pooled . (d/scale)
    head = np.zeros((2, D), dtype=np.float64)
    head[1] = d / scale
    head[0] = 0.0
    hb = np.zeros(2, dtype=np.float64)
    hb[1] = bias

    # ---- write back (keep shapes, float32, finite, |.|<=1000) ---------------
    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for k in ["block0.qkv.weight","block0.qkv.bias","block0.proj.weight","block0.proj.bias",
              "block0.ff1.weight","block0.ff1.bias","block0.ff2.weight","block0.ff2.bias",
              "block1.qkv.weight","block1.qkv.bias","block1.proj.weight","block1.proj.bias",
              "block1.ff1.weight","block1.ff1.bias","block1.ff2.weight","block1.ff2.bias"]:
        if k in w:
            w[k] = np.zeros_like(w[k], dtype=np.float32)
    for k in ["block0.norm1.weight","block0.norm2.weight","block1.norm1.weight",
              "block1.norm2.weight","norm.weight"]:
        if k in w:
            w[k] = np.ones_like(w[k], dtype=np.float32)
    for k in ["block0.norm1.bias","block0.norm2.bias","block1.norm1.bias",
              "block1.norm2.bias","norm.bias"]:
        if k in w:
            w[k] = np.zeros_like(w[k], dtype=np.float32)

    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = hb.astype(np.float32)

    # ---- diagnostics ---------------------------------------------------------
    pred_tr = (B @ wv) + bias > 0
    train_acc = float(np.mean(pred_tr == (t > 0)))
    diag = {
        "best_lambda": float(best_lam),
        "cv_acc": float(best_acc),
        "train_acc": train_acc,
        "bias": float(bias),
        "w_norm": float(np.linalg.norm(wv)),
        "max_abs_w": float(np.max(np.abs(wv))),
        "embed_norm_mean": float(np.mean(np.linalg.norm(E, axis=1))),
        "embed_mean": float(np.mean(E)),
        "head_norm": float(np.linalg.norm(head)),
        "n_features": int(Vf),
    }
    return {"weights": w, "diagnostics": diag}
