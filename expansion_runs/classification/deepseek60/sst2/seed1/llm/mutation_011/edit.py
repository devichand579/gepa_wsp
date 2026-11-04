import numpy as np

def edit(w, f):
    X_ids = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    y     = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, T  = X_ids.shape
    V     = w["embed.weight"].shape[0]                  # 2000

    # ---- length-normalized bag-of-words features (pad id 0 excluded) ----
    mask   = (X_ids != 0)
    L      = np.maximum(mask.sum(axis=1), 1).astype(np.float64)   # [N]
    X      = np.zeros((N, V), dtype=np.float64)
    rows   = np.repeat(np.arange(N), T)
    cols   = X_ids.reshape(-1)
    valid  = mask.reshape(-1)
    np.add.at(X, (rows[valid], cols[valid]), 1.0)
    X     /= L[:, None]                                  # [N,V]

    # ---- closed-form ridge on signed targets (no iterative optimization) ----
    t = np.where(y == 1, 1.0, -1.0).astype(np.float64)   # [N]
    lam = 1.0
    Xa  = np.concatenate([X, np.ones((N, 1))], axis=1)   # [N, V+1]
    G   = Xa.T @ Xa                                      # [V+1, V+1]
    reg = np.full(V + 1, lam, dtype=np.float64)
    reg[-1] = 0.0                                        # don't regularize bias
    G[np.diag_indices(V + 1)] += reg
    rhs = Xa.T @ t
    sol = np.linalg.solve(G, rhs)
    wv  = sol[:V].astype(np.float64)                     # per-token scalar weight
    b   = float(sol[V])

    # ---- realize s[v] = (0, wv[v]) via head = c*[e0;e1] ----
    # d[v][0] = 0, d[v][1] = wv[v]/c ; remaining 30 coords solved for
    # zero-mean & unit-variance.  Choose c so the construction is feasible.
    maxw = float(np.max(np.abs(wv))) + 1e-12
    c    = maxw / np.sqrt(30.0) * 1.5 + 1e-6             # keeps d0^2+d1^2 << 32
    d0   = np.zeros(V, dtype=np.float64)
    d1   = wv / c
    S    = -(d0 + d1)                                    # required sum of coords 2..31
    Q    = 32.0 - d0**2 - d1**2                          # required sum of squares
    # 29 coords = t, 1 coord = u ;  29t+u=S, 29t^2+u^2=Q
    disc = -116.0 * S**2 + 3480.0 * Q
    disc = np.maximum(disc, 0.0)
    tt   = (58.0 * S + np.sqrt(disc)) / 1740.0
    uu   = S - 29.0 * tt

    D = np.zeros((V, 32), dtype=np.float64)
    D[:, 0] = d0
    D[:, 1] = d1
    D[:, 2:31] = tt[:, None]
    D[:, 31]   = uu
    # exact renormalization to zero-mean / unit-variance (guards float error)
    D -= D.mean(axis=1, keepdims=True)
    D /= np.sqrt((D**2).mean(axis=1, keepdims=True) + 1e-12)

    # ---- write embedding (pos stays zero, blocks stay zero) ----
    emb = np.zeros_like(w["embed.weight"], dtype=np.float32)
    emb[:] = D.astype(np.float32)
    w["embed.weight"] = emb

    # ---- head: rows are c*e0 and c*e1 ; bias = (0, b) ----
    head = np.zeros_like(w["head.weight"], dtype=np.float32)
    head[0, 0] = np.float32(c)
    head[1, 1] = np.float32(c)
    w["head.weight"] = head
    w["head.bias"]   = np.array([0.0, b], dtype=np.float32)

    # ---- diagnostics ----
    diag = {
        "n_tokens": int(V),
        "n_samples": int(N),
        "mean_len": float(L.mean()),
        "ridge_lambda": lam,
        "c": float(c),
        "wv_absmax": float(np.max(np.abs(wv))),
        "wv_absmean": float(np.mean(np.abs(wv))),
        "bias": float(b),
        "d_var_mean": float(np.mean((D**2).mean(axis=1))),
        "d_mean_absmax": float(np.max(np.abs(D.mean(axis=1)))),
        "head_norm": float(np.linalg.norm(head)),
    }
    return {"weights": w, "diagnostics": diag}
