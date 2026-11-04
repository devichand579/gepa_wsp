import numpy as np

def edit(w, f):
    X = np.asarray(f["x"])            # [N,64] int64
    y = np.asarray(f["y"]).astype(np.float64)  # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]    # 2000
    D = w["embed.weight"].shape[1]    # 32

    # ---- 1. length-normalized unigram count matrix P = C / n ----
    mask = (X != 0)
    n = mask.sum(axis=1).astype(np.float64)
    n[n == 0] = 1.0
    C = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    valid = (cols != 0)
    np.add.at(C, (rows[valid], cols[valid]), 1.0)
    P = C / n[:, None]

    # ---- 2. ridge fit of score = P w + b  (targets in {-1,+1}) ----
    t = 2.0 * y - 1.0
    lam = 1.0
    A = np.concatenate([P, np.ones((N, 1))], axis=1)   # [N, V+1]
    G = A.T @ A
    G[np.arange(V + 1), np.arange(V + 1)] += lam
    rhs = A.T @ t
    sol = np.linalg.solve(G, rhs)
    w_tok = sol[:V]          # per-token score  alpha_t
    b_diff = float(sol[V])

    # ---- 3. realize alpha_t = d0 . LN(e_t) ----
    # d0 = k * d_unit, d_unit mean-0 unit vector;  need |alpha_t| <= k*sqrt(32)
    d_unit = np.zeros(D, dtype=np.float64)
    d_unit[0] = 1.0 / np.sqrt(2.0)
    d_unit[1] = -1.0 / np.sqrt(2.0)          # mean 0, unit norm
    v_perp = np.zeros(D, dtype=np.float64)
    v_perp[0] = 1.0 / np.sqrt(6.0)
    v_perp[1] = 1.0 / np.sqrt(6.0)
    v_perp[2] = -2.0 / np.sqrt(6.0)          # mean 0, unit, orthogonal to d_unit

    amax = float(np.max(np.abs(w_tok))) if V else 0.0
    k = max(amax / np.sqrt(32.0) * 1.05, 1e-6)

    a = w_tok / k
    a = np.clip(a, -np.sqrt(32.0) * 0.999, np.sqrt(32.0) * 0.999)
    c = np.sqrt(np.maximum(32.0 - a * a, 0.0))

    # LN(e_t) = a_t * d_unit + c_t * v_perp  (mean 0, norm sqrt(32))
    U = a[:, None] * d_unit[None, :] + c[:, None] * v_perp[None, :]
    # e_t = U  (already mean 0, std 1  -> LN(e_t) ~= e_t)
    emb = np.zeros((V, D), dtype=np.float64)
    emb[:] = U

    # ---- 4. head: difference d = k * d_unit, symmetric split ----
    d = k * d_unit
    head = np.zeros((2, D), dtype=np.float64)
    head[0] = -d / 2.0
    head[1] =  d / 2.0
    bias = np.array([-b_diff / 2.0, b_diff / 2.0], dtype=np.float64)

    # ---- 5. write back (float32, finite, |.|<=1000) ----
    w["embed.weight"] = np.clip(emb, -1000.0, 1000.0).astype(np.float32)
    w["head.weight"]  = np.clip(head, -1000.0, 1000.0).astype(np.float32)
    w["head.bias"]    = np.clip(bias, -1000.0, 1000.0).astype(np.float32)

    # ---- diagnostics: simulate forward on the pool ----
    Uf = np.asarray(w["embed.weight"], dtype=np.float64)
    # LN over last axis
    mu = Uf.mean(axis=1, keepdims=True)
    var = Uf.var(axis=1, keepdims=True)
    Un = (Uf - mu) / np.sqrt(var + 1e-5)
    pooled = np.zeros((N, D), dtype=np.float64)
    np.add.at(pooled, rows[valid], Un[cols[valid]])
    pooled /= n[:, None]
    logits = pooled @ np.asarray(w["head.weight"], dtype=np.float64).T + np.asarray(w["head.bias"], dtype=np.float64)
    pred = logits.argmax(axis=1)
    acc = float((pred == y.astype(np.int64)).mean())
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.clip(p[np.arange(N), y.astype(np.int64)], 1e-12, 1.0)).mean())

    diag = {
        "train_acc": acc,
        "train_ce": ce,
        "k": float(k),
        "alpha_absmax": amax,
        "b_diff": b_diff,
        "n_nonzero_tok": int((np.abs(w_tok) > 1e-9).sum()),
        "emb_norm_mean": float(np.linalg.norm(np.asarray(w["embed.weight"]), axis=1).mean()),
    }
    return {"weights": w, "diagnostics": diag}
