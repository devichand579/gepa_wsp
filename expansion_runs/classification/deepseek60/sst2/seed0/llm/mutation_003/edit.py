import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # (N,64) token ids
    y = f["y"].astype(np.int64)          # (N,) labels
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    mask = (X != 0)                      # non-pad
    T = mask.sum(axis=1).astype(np.float64)
    T[T == 0] = 1.0

    # length-normalized bag-of-words design matrix M[i,v] = count(i,v)/T_i
    M = np.zeros((N, V), dtype=np.float64)
    for j in range(L):
        col = X[:, j]
        rows = np.nonzero(mask[:, j])[0]
        if rows.size:
            np.add.at(M, (rows, col[rows]), 1.0)
    M /= T[:, None]

    yc = (y.astype(np.float64) - 0.5) * 2.0   # {-1,+1}

    # ---- ridge fit with CV over alpha ----
    # precompute Gram matrix (V x V) once
    G = M.T @ M
    b = M.T @ yc
    diag = np.diag(G).copy()

    # deterministic split for CV
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    folds = np.array_split(perm, 5)

    alphas = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_alpha, best_acc = alphas[0], -1.0
    for alpha in alphas:
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Gtr = M[tr].T @ M[tr]
            btr = M[tr].T @ yc[tr]
            s = np.linalg.solve(Gtr + alpha * np.eye(V), btr)
            pred = np.sign(M[te] @ s)
            accs.append(np.mean(pred == yc[te]))
        acc = float(np.mean(accs))
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha

    # final fit on all data
    s = np.linalg.solve(G + best_alpha * np.eye(V), b)

    # ---- encode s(v) into embed + head ----
    # target: logit_1 - logit_0 = (1/T) sum_t s(v_t) + b0
    # choose head_1-head_0 = c*e_0 ; g_v[0] = s(v)/c ; rest fills zero-mean/unit-var
    c = 100.0
    a = s / c                                   # dim-0 component, tiny
    # filler direction d in R^31: zero-sum, unit-norm
    d = np.zeros(D - 1, dtype=np.float64)
    d[0] = 1.0 / np.sqrt(2.0)
    d[1] = -1.0 / np.sqrt(2.0)
    # remaining coords: b_v = alpha_v * 1 + beta_v * d
    alpha_v = -a / (D - 1)
    beta_v = np.sqrt(np.maximum(32.0 - a * a * (D / (D - 1.0)), 1e-8))
    g = np.zeros((V, D), dtype=np.float64)
    g[:, 0] = a
    g[:, 1:] = alpha_v[:, None] + beta_v[:, None] * d[None, :]

    # enforce exact zero-mean / unit-variance numerically (safety)
    g -= g.mean(axis=1, keepdims=True)
    g /= np.sqrt((g * g).mean(axis=1, keepdims=True) + 1e-12)

    # head: symmetric split so both logits are meaningful
    head = np.zeros((2, D), dtype=np.float64)
    head[1, 0] = 0.5 * c
    head[0, 0] = -0.5 * c
    # bias: intercept b0 placed symmetrically (b1-b0 = b0)
    # choose b0 so that sign matches ridge intercept (ridge had no explicit bias -> 0)
    head_bias = np.array([0.0, 0.0], dtype=np.float64)

    # ---- write weights ----
    w["embed.weight"] = g.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for name in ["block0", "block1"]:
        for sub in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                    "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            w[f"{name}.{sub}"] = np.zeros_like(w[f"{name}.{sub}"], dtype=np.float32)
        w[f"{name}.norm1.weight"] = np.ones_like(w[f"{name}.norm1.weight"], dtype=np.float32)
        w[f"{name}.norm1.bias"] = np.zeros_like(w[f"{name}.norm1.bias"], dtype=np.float32)
        w[f"{name}.norm2.weight"] = np.ones_like(w[f"{name}.norm2.weight"], dtype=np.float32)
        w[f"{name}.norm2.bias"] = np.zeros_like(w[f"{name}.norm2.bias"], dtype=np.float32)
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    # clamp for safety
    for k in w:
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(arr, -1000.0, 1000.0, out=arr)
        w[k] = arr

    diag = {
        "best_alpha": float(best_alpha),
        "cv_acc": float(best_acc),
        "s_absmax": float(np.abs(s).max()),
        "s_std": float(s.std()),
        "c": float(c),
        "embed_rows_nonzero": int(np.count_nonzero(np.abs(g).sum(axis=1) > 1e-9)),
    }
    return {"weights": w, "diagnostics": diag}
