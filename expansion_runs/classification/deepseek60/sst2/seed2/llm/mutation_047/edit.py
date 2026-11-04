import numpy as np

def edit(w, f):
    X_ids = np.asarray(f["x"])          # [N,64] int64
    y     = np.asarray(f["y"]).astype(np.int64)   # [N]
    N, L  = X_ids.shape
    V     = w["embed.weight"].shape[0]  # 2000
    D     = w["embed.weight"].shape[1]  # 32

    # ---------- 1. length-normalised token-count matrix ----------
    mask = (X_ids != 0)
    lens = np.maximum(mask.sum(axis=1), 1).astype(np.float64)
    Xc = np.zeros((N, V), dtype=np.float64)
    for j in range(L):
        col = X_ids[:, j]
        valid = mask[:, j]
        np.add.at(Xc, (np.nonzero(valid)[0], col[valid]), 1.0)
    Xc /= lens[:, None]                 # (1/T) * counts

    # ---------- 2. ridge fit of per-token scores S (V x 2) ----------
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    Y = 2.0 * Y - 1.0                   # +/-1 targets
    G = Xc.T @ Xc                       # V x V
    B = Xc.T @ Y                        # V x 2

    def ridge_S(lam):
        A = G + lam * np.eye(V)
        return np.linalg.solve(A, B)

    # k-fold CV to pick lambda
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    K = 5
    folds = np.array_split(perm, K)
    lam_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_lam, best_cv = lam_grid[0], -1.0
    for lam in lam_grid:
        accs = []
        for k in range(K):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(K) if j != k])
            Gtr = Xc[tr].T @ Xc[tr]
            Btr = Xc[tr].T @ Y[tr]
            Str = np.linalg.solve(Gtr + lam * np.eye(V), Btr)
            pred = np.argmax(Xc[te] @ Str, axis=1)
            accs.append((pred == y[te]).mean())
        cv = float(np.mean(accs))
        if cv > best_cv:
            best_cv, best_lam = cv, lam
    S = ridge_S(best_lam)               # V x 2

    # ---------- 3. realise S = embed @ head.T with zero-mean unit-norm rows ----------
    # head = [[beta,0,...],[0,beta,0,...]]  =>  head @ embed[t] = beta * embed[t][:2]
    # want beta * embed[t][:2] = S[t]  =>  embed[t][:2] = S[t]/beta
    # rows must be zero-mean & unit-norm -> pick global scale alpha, then fill 30 dims.
    # feasibility: alpha^2*(p^2+q^2+(p+q)^2/30) <= 1 for every token
    p = S[:, 0]; q = S[:, 1]
    need = p * p + q * q + (p + q) ** 2 / 30.0
    mx = float(need.max())
    if mx <= 0:
        mx = 1.0
    alpha = 0.98 / np.sqrt(mx)           # safe global scale
    a = alpha * p                        # embed[t,0]
    b = alpha * q                        # embed[t,1]

    # fill dims 2..31 so that sum = -(a+b) and sum of squares = 1 - a^2 - b^2
    n = D - 2                            # 30
    s_target = -(a + b)                  # required sum of fill
    q_target = 1.0 - a * a - b * b       # required sum of squares of fill
    q_target = np.maximum(q_target, 0.0)
    base = s_target / n                  # equal part
    # residual sum of squares to distribute
    resid = np.maximum(q_target - n * base * base, 0.0)
    fill = np.full((V, n), base[:, None], dtype=np.float64)
    # put residual on first two fill dims as +/- d
    d = np.sqrt(resid / 2.0)
    fill[:, 0] += d
    fill[:, 1] -= d

    embed_new = np.zeros((V, D), dtype=np.float64)
    embed_new[:, 0] = a
    embed_new[:, 1] = b
    embed_new[:, 2:] = fill

    # head: beta chosen so that beta * (a,b) = S  => beta = 1/alpha
    beta = 1.0 / alpha
    head_new = np.zeros((2, D), dtype=np.float64)
    head_new[0, 0] = beta
    head_new[1, 1] = beta

    # ---------- 4. intercept: fit on the exact realised logits ----------
    # realised logits (pre-bias) = Xc @ S  (since head@features = (1/T)sum S)
    logits_pre = Xc @ S                  # N x 2
    # choose bias to maximise accuracy (closed form: median of decision margins)
    marg = logits_pre[:, 1] - logits_pre[:, 0]   # >0 predicts class1
    # best threshold on marg
    order = np.argsort(marg)
    ys = y[order]
    # cumulative accuracy for threshold between sorted values
    c1 = np.cumsum(ys)                   # predicted 0 for first k, 1 for rest
    c0 = np.cumsum(1 - ys)
    total1 = ys.sum()
    total0 = N - total1
    # if threshold after position k (k items predicted 0): correct = c0[k] + (total1 - c1[k])
    k = np.arange(N + 1)
    c0k = np.concatenate([[0], c0])
    c1k = np.concatenate([[0], c1])
    correct = c0k + (total1 - c1k)
    kbest = int(np.argmax(correct))
    if kbest == 0:
        thr = marg[order[0]] - 1e-6
    elif kbest == N:
        thr = marg[order[-1]] + 1e-6
    else:
        thr = 0.5 * (marg[order[kbest - 1]] + marg[order[kbest]])
    # logits = logits_pre + bias ; want logits[:,1]-logits[:,0] > 0 <=> marg > thr
    # so bias[0]-bias[1] = thr  -> set bias = [thr/2, -thr/2]
    bias_new = np.array([thr / 2.0, -thr / 2.0], dtype=np.float64)

    # ---------- 5. write back ----------
    w["embed.weight"] = embed_new.astype(np.float32)
    w["pos.weight"]   = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for name in ["block0.norm1.weight", "block0.norm1.bias",
                 "block0.qkv.weight", "block0.qkv.bias",
                 "block0.proj.weight", "block0.proj.bias",
                 "block0.norm2.weight", "block0.norm2.bias",
                 "block0.ff1.weight", "block0.ff1.bias",
                 "block0.ff2.weight", "block0.ff2.bias",
                 "block1.norm1.weight", "block1.norm1.bias",
                 "block1.qkv.weight", "block1.qkv.bias",
                 "block1.proj.weight", "block1.proj.bias",
                 "block1.norm2.weight", "block1.norm2.bias",
                 "block1.ff1.weight", "block1.ff1.bias",
                 "block1.ff2.weight", "block1.ff2.bias"]:
        w[name] = np.zeros_like(w[name], dtype=np.float32)
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"]   = np.zeros_like(w["norm.bias"], dtype=np.float32)
    w["head.weight"] = head_new.astype(np.float32)
    w["head.bias"]   = bias_new.astype(np.float32)

    # ---------- diagnostics ----------
    pred = (logits_pre[:, 1] - logits_pre[:, 0] > thr).astype(np.int64)
    train_acc = float((pred == y).mean())
    diag = {
        "lambda": float(best_lam),
        "cv_acc": float(best_cv),
        "train_acc": train_acc,
        "alpha": float(alpha),
        "beta": float(beta),
        "thr": float(thr),
        "class_acc": {
            "0": float((pred[y == 0] == 0).mean()),
            "1": float((pred[y == 1] == 1).mean()),
        },
        "embed_norm_mean": float(np.linalg.norm(embed_new, axis=1).mean()),
        "embed_norm_std": float(np.linalg.norm(embed_new, axis=1).std()),
        "embed_mean_abs": float(np.abs(embed_new.mean(axis=1)).max()),
        "S_absmax": float(np.abs(S).max()),
    }
    return {"weights": w, "diagnostics": diag}
