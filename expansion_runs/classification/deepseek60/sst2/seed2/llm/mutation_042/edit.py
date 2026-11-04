import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]                  # 2000

    # ---- normalized bag-of-words counts C[n,i] = count_i(n)/T_n ----
    mask = (X != 0)
    T = mask.sum(axis=1).astype(np.float64)
    T[T == 0] = 1.0
    C = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    np.add.at(C, (rows, cols), 1.0)
    C /= T[:, None]

    # ---- ridge fit of 2-output linear model:  C @ S + b ~ one-hot(y) ----
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    lam = 1.0
    # normal equations on [C | 1]
    A = np.concatenate([C, np.ones((N, 1))], axis=1)          # [N, V+1]
    G = A.T @ A                                                # [V+1, V+1]
    G[np.arange(V + 1), np.arange(V + 1)] += lam
    G[V, V] -= lam                                             # don't penalize bias
    R = A.T @ Y
    W = np.linalg.solve(G, R)                                  # [V+1, 2]
    S = W[:V, :]                                               # [V,2]
    b = W[V, :]                                                # [2]

    # ---- choose head scale a so that (S0^2+S1^2)/a^2 stays well below 32 ----
    mag = np.sqrt((S ** 2).sum(axis=1))
    a = max(mag.max() / 5.0, 1e-6)

    # ---- build zero-mean, unit-variance v_i with head @ v_i = S_i ----
    t0 = S[:, 0] / a
    t1 = S[:, 1] / a
    m = -(t0 + t1)                       # required sum of the 30 filler coords
    q = 32.0 - t0 ** 2 - t1 ** 2         # required sum of squares of filler
    q = np.maximum(q, m ** 2 / 30.0 + 1e-6)

    # 29 coords = c, 1 coord = m - 29c ; solve 870 c^2 - 58 m c + (m^2 - q) = 0
    disc = 3480.0 * q - 116.0 * m ** 2
    disc = np.maximum(disc, 0.0)
    c = (58.0 * m - np.sqrt(disc)) / 1740.0
    last = m - 29.0 * c

    E = np.zeros((V, 32), dtype=np.float64)
    E[:, 0] = t0
    E[:, 1] = t1
    E[:, 2:31] = c[:, None]
    E[:, 31] = last

    # renormalize to be safe against numerical drift (LN is scale invariant)
    mu = E.mean(axis=1, keepdims=True)
    sd = E.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    E = (E - mu) / sd

    w["embed.weight"] = E.astype(np.float32)

    H = np.zeros((2, 32), dtype=np.float64)
    H[0, 0] = a
    H[1, 1] = a
    w["head.weight"] = H.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # ---- diagnostics: check realized linear model on feedback pool ----
    realized = C @ S + b
    pred = realized.argmax(axis=1)
    acc = float((pred == y).mean())
    ce = float(-np.mean(np.log(np.clip(
        np.exp(realized - realized.max(axis=1, keepdims=True)) /
        np.exp(realized - realized.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True),
        1e-12, 1.0))[np.arange(N), y]))
    diag = {
        "fit_acc": acc,
        "fit_ce": ce,
        "a": float(a),
        "S_norm_mean": float(np.linalg.norm(S, axis=1).mean()),
        "S_norm_max": float(np.linalg.norm(S, axis=1).max()),
        "b": [float(b[0]), float(b[1])],
        "embed_std": float(E.std()),
        "head_norm": float(np.linalg.norm(H)),
    }
    return {"weights": w, "diagnostics": diag}
