import numpy as np
import math

def edit(w, f):
    # ---- read feedback -------------------------------------------------
    x_all = np.asarray(f["x"], dtype=np.int64)          # [4000,64]
    y_all = np.asarray(f["y"], dtype=np.int64)          # [4000]
    feats = np.asarray(f["features"], dtype=np.float32) # [4000,32]
    N, L = x_all.shape
    V = w["embed.weight"].shape[0]                      # 2000

    # ---- 1. closed-form ridge token scores on normalised counts --------
    # Build sparse count matrix implicitly: for each example, mean of one-hots
    # is just count / length.  We fit  s = argmin ||X s - y||^2 + lam||s||^2
    # where X[i, token] = (#occurrences of token in doc i) / len(doc i).
    lam = 1e-2
    # accumulate Gram matrix G = X^T X  and rhs r = X^T y  without dense X
    G = np.zeros((V, V), dtype=np.float64)
    r = np.zeros(V, dtype=np.float64)
    for i in range(N):
        ids = x_all[i]
        ids = ids[ids != 0]
        if ids.size == 0:
            continue
        cnt = np.bincount(ids, minlength=V).astype(np.float64)
        cnt /= ids.size
        nz = np.nonzero(cnt)[0]
        c = cnt[nz]
        G[np.ix_(nz, nz)] += np.outer(c, c)
        r[nz] += c * (1.0 if y_all[i] == 1 else -1.0)
    G[np.diag_indices(V)] += lam
    try:
        s = np.linalg.solve(G, r)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(G, r, rcond=None)[0]
    s = np.asarray(s, dtype=np.float32)
    s = np.clip(s, -5.0, 5.0)

    # ---- 2. build embedding:  [score, const, 0...] ---------------------
    emb = np.zeros((V, 32), dtype=np.float32)
    emb[:, 0] = s
    emb[:, 1] = 0.1          # small constant so LN keeps a usable direction
    w["embed.weight"] = emb
    # zero out positional embedding so the block stays an identity
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # ---- 3. compute pooled features for the whole pool ----------------
    # forward: h = embed(ids); per-token LN; masked mean
    gamma = w["norm.weight"].astype(np.float32)   # [32]
    beta  = w["norm.bias"].astype(np.float32)     # [32]
    eps = 1e-5
    P = np.zeros((N, 32), dtype=np.float32)
    for i in range(N):
        ids = x_all[i]
        mask = ids != 0
        if not mask.any():
            continue
        e = emb[ids]                              # [64,32]
        mu = e.mean(axis=1, keepdims=True)
        var = ((e - mu) ** 2).mean(axis=1, keepdims=True)
        ln = (e - mu) / np.sqrt(var + eps) * gamma + beta
        P[i] = ln[mask].mean(axis=0)

    # ---- 4. closed-form ridge head on the exact pooled features --------
    # target: +1 for positive, -1 for negative ; then map to 2 logits
    t = np.where(y_all == 1, 1.0, -1.0).astype(np.float64)
    Pf = P.astype(np.float64)
    lam_h = 1e-3
    A = Pf.T @ Pf + lam_h * np.eye(32)
    b = Pf.T @ t
    try:
        wh = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        wh = np.linalg.lstsq(A, b, rcond=None)[0]
    wh = np.asarray(wh, dtype=np.float32)
    # head rows: class0 = -wh, class1 = +wh  (so logit1-logit0 = 2*wh·feat)
    head_w = np.zeros((2, 32), dtype=np.float32)
    head_w[0] = -wh
    head_w[1] =  wh
    w["head.weight"] = head_w
    w["head.bias"] = np.zeros(2, dtype=np.float32)

    # ---- diagnostics ---------------------------------------------------
    logits = P @ wh
    pred = (logits > 0).astype(np.int64)
    acc = float((pred == y_all).mean())
    diag = {
        "train_acc": acc,
        "score_mean": float(s.mean()),
        "score_std": float(s.std()),
        "head_norm": float(np.linalg.norm(wh)),
        "pooled_std": float(P.std()),
    }
    return {"weights": w, "diagnostics": diag}
