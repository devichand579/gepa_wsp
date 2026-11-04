import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [4000,64]
    y = f["y"].astype(np.int64)          # [4000]
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000

    # ---- 1. length-normalized token-count design matrix ----
    mask = (X != 0).astype(np.float32)   # ignore PAD (id 0)
    T = mask.sum(1)
    T[T == 0] = 1.0
    C = np.zeros((N, V), dtype=np.float32)
    for j in range(L):
        ids = X[:, j]
        m = mask[:, j]
        np.add.at(C, (np.arange(N), ids), m)
    C /= T[:, None]                      # [N,V]  mean token indicator

    # ---- 2. ridge fit of class score, alpha chosen by CV ----
    yt = y.astype(np.float32) * 2.0 - 1.0   # {-1,+1}
    best = None
    for alpha in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        # 5-fold CV
        accs = []
        idx = np.arange(N)
        rng = np.random.RandomState(0)
        rng.shuffle(idx)
        folds = np.array_split(idx, 5)
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            A = C[tr]
            G = A.T @ A + alpha * np.eye(V, dtype=np.float32)
            b = A.T @ yt[tr]
            s = np.linalg.solve(G, b)
            pred = np.sign(C[te] @ s)
            pred[pred == 0] = 1.0
            accs.append(np.mean(pred == yt[te]))
        acc = float(np.mean(accs))
        if best is None or acc > best[1]:
            best = (alpha, acc)
    alpha = best[0]

    # final fit on all data
    G = C.T @ C + alpha * np.eye(V, dtype=np.float32)
    b = C.T @ yt
    s = np.linalg.solve(G, b).astype(np.float32)   # per-token score s[v]
    s = s - s.mean()

    # ---- 3. encode s(v) into embed dim 0 with zero-mean / unit-norm rows ----
    # head diff = [k,0,...,0] -> logit diff = k * pooled_0 = k * mean_t t_{v,0}
    # choose k so scores are in range; t_{v,0} = s(v)/k must satisfy |t_0|<=sqrt(31)
    smax = float(np.max(np.abs(s))) + 1e-9
    k = smax / (np.sqrt(31.0) * 0.98)     # keep |t_0| < sqrt(31)
    a = (s / k).astype(np.float32)        # target dim-0 value, |a| <= 0.98*sqrt(31)

    # For each token build t with t_0 = a, sum_{i>=1} t_i = -a, sum_{i>=1} t_i^2 = 32 - a^2
    # Use 30 equal entries c and one entry d: 30c + d = -a ; 30c^2 + d^2 = 32 - a^2
    # => 465 c^2 + 30 a c + a^2 - 16 = 0
    disc = np.maximum(29760.0 - 960.0 * a * a, 0.0)
    c = (-30.0 * a + np.sqrt(disc)) / 930.0
    d = -a - 30.0 * c
    E = np.zeros((V, 32), dtype=np.float32)
    E[:, 0] = a
    E[:, 1:31] = c[:, None]
    E[:, 31] = d

    # safety: renormalize to exact zero-mean / unit-variance (removes any eps drift)
    E = E - E.mean(1, keepdims=True)
    nrm = np.sqrt((E * E).sum(1, keepdims=True))
    nrm[nrm < 1e-6] = 1.0
    E = E * (np.sqrt(32.0) / nrm)

    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # ---- 4. head: logit1 - logit0 = k * pooled_0 ----
    head = np.zeros_like(w["head.weight"], dtype=np.float32)
    head[0, 0] = -k * 0.5
    head[1, 0] = +k * 0.5
    w["head.weight"] = head
    w["head.bias"] = np.zeros_like(w["head.bias"], dtype=np.float32)

    diag = {
        "alpha": float(alpha),
        "cv_acc": float(best[1]),
        "k": float(k),
        "s_absmax": float(smax),
        "s_std": float(np.std(s)),
        "embed_rows_nonzero": int(np.count_nonzero(E)),
    }
    return {"weights": w, "diagnostics": diag}
