def edit(w, f):
    import numpy as np

    X = f["x"].astype(np.int64)          # (N,64) token ids
    y = f["y"].astype(np.int64)          # (N,)
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32
    mask = (X != 0)
    lens = mask.sum(1).astype(np.float64)
    lens[lens == 0] = 1.0

    # ---- bag-of-words mean design matrix Xb[i,t] = count(t in i)/len(i) ----
    Xb = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    np.add.at(Xb, (rows, cols), 1.0)
    Xb /= lens[:, None]

    yv = (2.0 * y - 1.0).astype(np.float64)   # ±1

    # ---- (a) ridge regression on ±1 targets (normal equations) ----
    G = Xb.T @ Xb                              # (V,V)
    b = Xb.T @ yv                              # (V,)
    lam = 1.0
    A = G + lam * np.eye(V)
    try:
        w_ridge = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w_ridge = np.linalg.lstsq(A, b, rcond=None)[0]

    # ---- (b) Naive-Bayes log-odds from class moments ----
    pos = (y == 1)
    neg = (y == 0)
    cp = Xb[pos].sum(0)                        # token mass in class 1
    cn = Xb[neg].sum(0)
    alpha = 1.0
    pp = (cp + alpha) / (cp.sum() + alpha * V)
    pn = (cn + alpha) / (cn.sum() + alpha * V)
    w_nb = np.log(pp / pn)

    # ---- explicit reduced forward pass: LN(embed[ids]) -> masked mean ----
    def pooled_from_v(v):
        E = w["embed.weight"].astype(np.float64).copy()
        E[:, 1] = v
        H = E[X]                               # (N,64,32)
        mu = H.mean(2, keepdims=True)
        var = H.var(2, keepdims=True)
        Hn = (H - mu) / np.sqrt(var + 1e-5)    # gamma=1,beta=0
        Hn = Hn * mask[:, :, None]
        return Hn.sum(1) / lens[:, None]       # (N,32)

    def train_acc(v):
        P = pooled_from_v(v)
        # closed-form 1-D threshold on the discriminative direction
        # use ridge head fit on P for a fair score
        Ph = np.concatenate([P, np.ones((N, 1))], 1)
        W = np.linalg.solve(Ph.T @ Ph + 1e-3 * np.eye(D + 1), Ph.T @ yv)
        s = Ph @ W
        return float(((s > 0).astype(np.int64) == y).mean())

    # scale candidates so LN realization stays in a well-conditioned range
    def scale_to(v, m=1.5):
        a = np.abs(v).max()
        return v * (m / a) if a > 0 else v

    cands = {
        "ridge": scale_to(w_ridge),
        "nb": scale_to(w_nb),
        "ridge+nb": scale_to(w_ridge / (np.abs(w_ridge).max() + 1e-9)
                            + 0.5 * w_nb / (np.abs(w_nb).max() + 1e-9)),
    }
    best_name, best_v, best_acc = None, None, -1.0
    for name, v in cands.items():
        acc = train_acc(v)
        if acc > best_acc:
            best_name, best_v, best_acc = name, v, acc

    # ---- write per-token weights into embed[:,1]; keep dim-31 structure ----
    E = w["embed.weight"].astype(np.float32).copy()
    E[:, 1] = best_v.astype(np.float32)
    w["embed.weight"] = E

    # ---- refit head (2x32) in closed form on the ACTUAL new features ----
    P = pooled_from_v(best_v)                  # (N,32)
    Ph = np.concatenate([P, np.ones((N, 1))], 1)
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    W = np.linalg.solve(Ph.T @ Ph + 1e-2 * np.eye(D + 1), Ph.T @ Y)  # (33,2)
    w["head.weight"] = W[:D].T.astype(np.float32)   # (2,32)
    w["head.bias"] = W[D].astype(np.float32)        # (2,)

    # ---- diagnostics ----
    logits = Ph @ W
    pred = logits.argmax(1)
    acc = float((pred == y).mean())
    diag = {
        "method": best_name,
        "train_acc": acc,
        "cand_accs": {k: train_acc(v) for k, v in cands.items()},
        "v_absmax": float(np.abs(best_v).max()),
        "v_norm": float(np.linalg.norm(best_v)),
        "head_w_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
    }
    return {"weights": w, "diagnostics": diag}
