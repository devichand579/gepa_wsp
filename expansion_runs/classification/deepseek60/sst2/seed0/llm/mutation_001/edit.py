import numpy as np

def edit(w, f):
    X_ids = np.asarray(f["x"])            # (4000,64) int64
    y     = np.asarray(f["y"]).astype(np.int64)
    N, L  = X_ids.shape
    V     = w["embed.weight"].shape[0]    # 2000
    D     = w["embed.weight"].shape[1]    # 32

    # ---------- 1. bag-of-words (normalized by non-pad length) ----------
    mask = (X_ids != 0)                    # pad id = 0
    T = mask.sum(axis=1).astype(np.float64)
    T[T == 0] = 1.0
    C = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X_ids.reshape(-1)
    np.add.at(C, (rows, cols), 1.0)
    C[:, 0] = 0.0                          # pad contributes nothing
    Xb = C / T[:, None]                    # (N,V)

    # ---------- 2. closed-form ridge regression on +-1 targets ----------
    t = np.where(y == 1, 1.0, -1.0)        # (N,)
    lam = 1.0                              # regularization
    # solve (Xb^T Xb + lam I) w = Xb^T t  (bias handled separately, unregularized)
    G = Xb.T @ Xb + lam * np.eye(V)
    rhs = Xb.T @ t
    wv = np.linalg.solve(G, rhs)           # (V,)
    b0 = float(t.mean() - Xb.mean(axis=0) @ wv)

    # ---------- 3. encode s(v) = [0, wv] into zero-mean unit directions ----------
    head_scale = 1.0
    # q_v = wv[v] / (head_scale * sqrt(32));  need |q| small so filler stays real
    q = wv / (head_scale * np.sqrt(D))
    qmax = float(np.max(np.abs(q))) if V > 0 else 0.0
    if qmax > 0.5:                         # enlarge head to keep directions valid
        head_scale = head_scale * (qmax / 0.5) * 1.05
        q = wv / (head_scale * np.sqrt(D))

    E = np.zeros((V, D), dtype=np.float64)
    nfill = D - 2                          # dims 2..D-1 used as filler
    for v in range(V):
        p = 0.0
        qv = q[v]
        # remaining nfill dims must sum to -(p+qv) and have squared-sum 1 - p^2 - qv^2
        R = -(p + qv)
        S = 1.0 - p * p - qv * qv
        if S < 0:
            S = 0.0
        u = R / nfill
        rem_sq = S - nfill * u * u
        if rem_sq < 0:
            rem_sq = 0.0
        tt = np.sqrt(rem_sq / nfill)
        fill = np.full(nfill, u, dtype=np.float64)
        fill[0::2] += tt
        fill[1::2] -= tt
        E[v, 0] = p
        E[v, 1] = qv
        E[v, 2:] = fill

    # ---------- 4. write tensors ----------
    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"]   = np.zeros_like(w["pos.weight"], dtype=np.float32)

    head = np.zeros((2, D), dtype=np.float32)
    head[0, 0] = head_scale
    head[1, 1] = head_scale
    w["head.weight"] = head
    w["head.bias"]   = np.array([0.0, b0], dtype=np.float32)

    # keep blocks as identity (all-zero weights) but ensure shapes/finiteness
    for k in list(w.keys()):
        if k.startswith("block"):
            w[k] = np.zeros_like(w[k], dtype=np.float32)
    for k in ("norm.weight", "norm.bias"):
        if k in w:
            w[k] = w[k].astype(np.float32)

    # ---------- diagnostics ----------
    logits = Xb @ np.stack([np.zeros(V), wv], axis=1) + np.array([0.0, b0])
    pred = np.argmax(logits, axis=1)
    acc = float((pred == y).mean())
    diag = {
        "train_acc": acc,
        "head_scale": float(head_scale),
        "w_absmax": float(np.max(np.abs(wv))),
        "w_norm": float(np.linalg.norm(wv)),
        "bias": float(b0),
        "nonzero_embed_rows": int(np.sum(np.any(E != 0, axis=1))),
    }
    return {"weights": w, "diagnostics": diag}
