def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = np.asarray(f["features"], dtype=np.float64)      # [N,1568] current final features
    y = np.asarray(f["y"]).astype(np.int64).ravel()
    N, D = X.shape
    C = 10

    # ---------------- current head (reference) ----------------
    W0 = np.asarray(w["head.weight"], dtype=np.float64)
    b0 = np.asarray(w["head.bias"], dtype=np.float64)
    L0 = X @ W0.T + b0
    cur_pred = L0.argmax(1)
    cur_acc = float((cur_pred == y).mean())

    # ---------------- class-balanced ridge (closed form, normal equations) ----
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    cnt = np.bincount(y, minlength=C).astype(np.float64)
    cnt[cnt == 0] = 1.0
    sw = 1.0 / cnt[y]                       # each class contributes total weight 1
    sw = sw / sw.mean()

    Xs = X * sw[:, None]
    Ys = Y * sw[:, None]

    G = Xs.T @ Xs                           # [D,D] weighted gram
    Bm = Xs.T @ Ys                          # [D,C] weighted cross-moment
    tr = float(np.trace(G)) / D
    lam = 0.10 * tr                         # analytic ridge level (scale from data)

    def solve(lamv, Gm=G, Bm=Bm):
        A = Gm + lamv * np.eye(D)
        W = np.linalg.solve(A, Bm)         # [D,C]
        b = (Ys.sum(0) - Xs @ W) / N
        return W, b

    def acc_of(W, b, Xd, yd):
        return float(((Xd @ W + b).argmax(1) == yd).mean())

    # ---------------- out-of-fold estimate of the balanced ridge ----------------
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    folds = np.array_split(perm, 5)
    oof_correct = 0
    oof_pred = np.zeros(N, dtype=np.int64)
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        Xtr, ytr = X[tr], y[tr]
        Ytr = np.zeros((tr.size, C)); Ytr[np.arange(tr.size), ytr] = 1.0
        ctr = np.bincount(ytr, minlength=C).astype(np.float64); ctr[ctr == 0] = 1.0
        swtr = (1.0 / ctr[ytr]); swtr = swtr / swtr.mean()
        Xs2 = Xtr * swtr[:, None]; Ys2 = Ytr * swtr[:, None]
        G2 = Xs2.T @ Xs2; B2 = Xs2.T @ Ys2
        tr2 = float(np.trace(G2)) / D
        A2 = G2 + lam * tr2 * np.eye(D)
        W2 = np.linalg.solve(A2, B2)
        b2 = (Ys2.sum(0) - Xs2 @ W2) / tr.size
        p = (X[te] @ W2 + b2).argmax(1)
        oof_pred[te] = p
        oof_correct += int((p == y[te]).sum())
    oof_acc = oof_correct / float(N)

    # ---------------- in-sample fit of the balanced ridge ----------------
    Wb, bb = solve(lam)
    bal_acc = acc_of(Wb, bb, X, y)

    # ---------------- decision: only replace if clearly better ----------------
    margin = 0.002
    use_new = (oof_acc > cur_acc + margin)

    diag = {
        "cur_acc_pool": round(cur_acc, 4),
        "ridge_bal_oof_acc": round(oof_acc, 4),
        "ridge_bal_insample_acc": round(bal_acc, 4),
        "lambda_over_trace": 0.10,
        "use_new_head": bool(use_new),
        "cur_pred_hist": np.bincount(cur_pred, minlength=C).tolist(),
        "ridge_pred_hist": np.bincount(oof_pred, minlength=C).tolist(),
        "true_hist": np.bincount(y, minlength=C).tolist(),
    }
    if use_new:
        cur_cm = np.zeros((C, C), dtype=np.int64)
        for t, p in zip(y, cur_pred):
            cur_cm[t, p] += 1
        new_cm = np.zeros((C, C), dtype=np.int64)
        for t, p in zip(y, oof_pred):
            new_cm[t, p] += 1
        diag["cur_per_class_acc"] = [round(float(cur_cm[i, i]) / max(1, cnt[i]), 4) for i in range(C)]
        diag["ridge_per_class_acc"] = [round(float(new_cm[i, i]) / max(1, cnt[i]), 4) for i in range(C)]
        diag["cur_confusion"] = cur_cm.tolist()
        diag["ridge_confusion_oof"] = new_cm.tolist()
        w["head.weight"] = Wb.T.astype(np.float32)
        w["head.bias"] = bb.astype(np.float32)
        diag["head_weight_absmax"] = round(float(np.abs(w["head.weight"]).max()), 4)
        diag["head_bias"] = [round(float(v), 4) for v in w["head.bias"]]

    # ---------------- safety: finite, float32, bounded ----------------
    for k in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "head.weight", "head.bias"):
        a = np.asarray(w[k], dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        w[k] = np.clip(a, -1000.0, 1000.0).astype(np.float32)

    return {"weights": w, "diagnostics": diag}
