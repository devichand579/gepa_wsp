import numpy as np

def edit(w, f):
    x = f["x"]; y = f["y"]
    N, L = x.shape
    V = w["embed.weight"].shape[0]

    # ---- 1. Build normalized bag-of-words design matrix (pad id 0 excluded) ----
    mask = (x != 0)                       # [N,64]
    T = mask.sum(1).astype(np.float64)    # non-pad length per example
    T[T == 0] = 1.0
    Phi = np.zeros((N, V), dtype=np.float64)
    for t in range(L):
        col = x[:, t]
        sel = mask[:, t]
        np.add.at(Phi, (np.arange(N)[sel], col[sel]), 1.0)
    Phi /= T[:, None]                     # matches the 1/T masked-mean pooling

    # ---- 2. Closed-form ridge fit of a linear BOW classifier (one-hot targets) ----
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    lam = 5.0
    A = Phi.T @ Phi                       # V x V
    A[np.diag_indices_from(A)] += lam
    Wt = np.linalg.solve(A, Phi.T @ Y)    # V x 2  (token weights)
    W = Wt.T.copy()                       # 2 x V
    b = W.mean(0) * 0.0
    b = (Y.mean(0) - Phi.mean(0) @ Wt)    # bias so class means match

    # ---- 3. Realize W as head @ u_i with u_i zero-mean unit-variance ----
    scale = np.sqrt(max(np.max(W[0] ** 2 + W[1] ** 2) / 0.5, 1e-12))
    a = 1.0 / scale                       # head = a * [[1,0..],[0,1,..]]
    u0 = W[0] * a
    u1 = W[1] * a

    U = np.zeros((V, 32), dtype=np.float64)
    U[:, 0] = u0
    U[:, 1] = u1
    s = -(u0 + u1)                        # remaining 30 coords must sum to s
    q = 32.0 - (u0 ** 2 + u1 ** 2)        # ... and sum of squares to q
    # split remaining 30 into 29 equal 'p' and one 'r': 29p+r=s, 29p^2+r^2=q
    Aq = 870.0
    Bq = -58.0 * s
    Cq = s ** 2 - q
    disc = np.maximum(Bq ** 2 - 4 * Aq * Cq, 0.0)
    p = (-Bq - np.sqrt(disc)) / (2 * Aq)
    r = s - 29.0 * p
    U[:, 2:31] = p[:, None]
    U[:, 31] = r

    # renormalize each row to exact zero-mean / unit-variance (guards eps)
    U -= U.mean(1, keepdims=True)
    U /= np.sqrt((U ** 2).mean(1, keepdims=True) + 1e-12)

    # ---- 4. Write weights (only embed + head matter; keep others as-is) ----
    w["embed.weight"] = U.astype(np.float32)
    head = np.zeros((2, 32), dtype=np.float64)
    head[0, 0] = a
    head[1, 1] = a
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # ---- 5. Diagnostics: recompute the forward pass to verify construction ----
    emb = w["embed.weight"].astype(np.float64)
    u = emb - emb.mean(1, keepdims=True)
    u = u / np.sqrt((u ** 2).mean(1, keepdims=True) + 1e-5)
    pooled = np.zeros((N, 32))
    for t in range(L):
        sel = mask[:, t]
        pooled[sel] += u[x[sel, t]]
    pooled /= T[:, None]
    logits = pooled @ w["head.weight"].astype(np.float64).T + w["head.bias"].astype(np.float64)
    pred = logits.argmax(1)
    acc = float((pred == y).mean())
    ce = float(-np.mean(np.log(np.clip(
        np.exp(logits - logits.max(1, keepdims=True)) /
        np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True),
        1e-12, 1.0)[np.arange(N), y])))

    diag = {
        "fit_train_acc": round(acc, 4),
        "fit_train_ce": round(ce, 4),
        "class_acc": {
            "0": round(float((pred[y == 0] == 0).mean()), 4),
            "1": round(float((pred[y == 1] == 1).mean()), 4),
        },
        "lambda": lam,
        "head_scale_a": float(a),
        "W_absmax": float(np.abs(W).max()),
        "n_samples": int(N),
    }
    return {"weights": w, "diagnostics": diag}
