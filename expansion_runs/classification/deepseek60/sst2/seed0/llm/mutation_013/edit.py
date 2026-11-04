import numpy as np
import math

def edit(w, f):
    X_ids = np.asarray(f["x"])          # [N,64] int64
    y = np.asarray(f["y"]).astype(np.float64)  # [N]
    N, L = X_ids.shape
    V = w["embed.weight"].shape[0]      # 2000

    # ---- build length-normalized unigram count design matrix ----
    mask = (X_ids != 0)
    lengths = np.maximum(mask.sum(axis=1), 1).astype(np.float64)
    X = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X_ids.reshape(-1)
    m = mask.reshape(-1)
    np.add.at(X, (rows[m], cols[m]), 1.0)
    X /= lengths[:, None]

    # ---- ridge fit with alpha chosen by k-fold CV (closed form) ----
    def ridge_fit(Xa, ya, alpha):
        A = Xa.T @ Xa + alpha * np.eye(V)
        b = Xa.T @ ya
        return np.linalg.solve(A, b)

    best_alpha, best_cv = 1.0, -1.0
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    K = 5
    folds = np.array_split(perm, K)
    for alpha in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        accs = []
        for k in range(K):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(K) if j != k])
            wv = ridge_fit(X[tr], y[tr], alpha)
            pred = (X[va] @ wv) > 0.5
            accs.append((pred == (y[va] > 0.5)).mean())
        cv = float(np.mean(accs))
        if cv > best_cv:
            best_cv, best_alpha = cv, alpha

    # final fit on all data
    wv = ridge_fit(X, y, best_alpha)

    # center scores (bias handled by head bias); keep magnitudes bounded
    b0 = 0.5 - 0.5 * float(np.mean(wv))  # intercept for 0/1 target
    # we predict sign(wv·x - 0.5 + 0.5) ... use decision value wv·x - 0.5
    # logit1-logit0 = wv·x + bias_diff ; want >0 <=> class1
    bias_diff = -0.5 + 0.5 * 0.0  # placeholder, recomputed below

    # ---- encode into embed + head ----
    D = 4.0
    a = wv / D
    a = np.clip(a, -3.99, 3.99)
    c = np.sqrt(np.maximum((16.0 - a * a) / 15.0, 1e-8))

    E = np.zeros((V, 32), dtype=np.float32)
    E[:, 0] = a
    E[:, 1] = -a
    E[:, 2:17] = c[:, None]
    E[:, 17:32] = -c[:, None]

    w["embed.weight"] = E

    # head: d = head1 - head0 = [D,0,...]; set head1 = d/2, head0 = -d/2
    head = np.zeros((2, 32), dtype=np.float32)
    head[1, 0] = D / 2.0
    head[0, 0] = -D / 2.0
    w["head.weight"] = head

    # bias: want logit1-logit0 = D*(mean e_v) + (bias1-bias0) = wv·x + (bias1-bias0)
    # decision boundary at wv·x = 0.5 -> bias1-bias0 = -0.5
    w["head.bias"] = np.array([0.25, -0.25], dtype=np.float32)  # bias1-bias0 = -0.5

    # keep pos zero (no positional signal needed)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    diag = {
        "best_alpha": float(best_alpha),
        "cv_acc": float(best_cv),
        "train_acc_linear": float(((X @ wv) > 0.5) == (y > 0.5)).mean(),
        "w_absmax": float(np.max(np.abs(wv))),
        "w_std": float(np.std(wv)),
        "embed_nonzero": int(np.count_nonzero(E)),
    }
    return {"weights": w, "diagnostics": diag}
