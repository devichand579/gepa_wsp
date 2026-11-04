import numpy as np

def edit(w, f):
    # ---- 0. constants -------------------------------------------------
    V, D = w["embed.weight"].shape          # 2000 x 32
    eps = 1e-5

    X = f["x"].astype(np.int64)             # [N,64]
    y = f["y"].astype(np.int64)             # [N]
    N = X.shape[0]

    # ---- 1. bag-of-words counts --------------------------------------
    # count[v] = number of documents containing token v (binary presence),
    # plus total occurrence count.  We use presence (more robust for short text).
    mask = (X != 0)                          # ignore PAD
    # build presence matrix efficiently
    presence = np.zeros((N, V), dtype=np.float32)
    rows = np.repeat(np.arange(N), X.shape[1])
    cols = X.reshape(-1)
    valid = (cols != 0)
    np.add.at(presence, (rows[valid], cols[valid]), 1.0)
    presence = (presence > 0).astype(np.float32)

    # document length (non-pad tokens)
    length = mask.sum(axis=1).astype(np.float32)
    length = np.maximum(length, 1.0)

    # ---- 2. closed-form linear classifier on token presence ----------
    # We fit  s_v  such that  sign( sum_v presence[v] * s_v + b ) = y.
    # Use ridge-regularised least squares on the (N x V) presence matrix.
    # Centre the labels to +-1.
    t = 2.0 * y.astype(np.float32) - 1.0     # {-1,+1}

    # Feature scaling: divide each column by its std to condition the solve.
    col_std = presence.std(axis=0)
    col_std[col_std < 1e-6] = 1.0
    P = presence / col_std                    # [N,V]

    # Ridge regression:  s = (P^T P + lam I)^-1 P^T t
    # V=2000, N=4000 -> solve in the smaller (N x N) or (V x V) space.
    lam = 1.0
    # Use the dual (N x N) form since N > V is false here (4000 > 2000),
    # so primal is smaller:  V x V = 2000 x 2000.
    PtP = P.T @ P
    PtP[np.diag_indices_from(PtP)] += lam
    Pty = P.T @ t
    try:
        s = np.linalg.solve(PtP, Pty)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(PtP, Pty, rcond=None)[0]

    # undo column scaling
    s = s / col_std

    # ---- 3. normalise scores to [-1, 1] ------------------------------
    # The construction below requires |score| <= 1.
    smax = np.max(np.abs(s))
    if smax < 1e-8:
        smax = 1.0
    score = (s / smax).astype(np.float32)
    score[0] = 0.0                            # PAD has no score

    # ---- 4. build zero-sum unit vectors w_t with delta . w_t = score_t
    # delta is a fixed zero-sum unit vector in R^32.
    delta = np.ones(D, dtype=np.float32)
    delta[0] = -(D - 1)
    delta /= np.linalg.norm(delta)            # unit, zero-sum

    # a fixed unit vector orthogonal to delta and to the all-ones vector
    # (so that it also lies in the zero-sum subspace)
    perp = np.zeros(D, dtype=np.float32)
    perp[0] = 1.0
    perp[1] = -1.0
    # remove component along delta and along ones
    perp -= (perp @ delta) * delta
    ones = np.ones(D, dtype=np.float32) / np.sqrt(D)
    perp -= (perp @ ones) * ones
    nperp = np.linalg.norm(perp)
    if nperp < 1e-8:
        # fallback: any other orthogonal direction
        perp = np.zeros(D, dtype=np.float32)
        perp[2] = 1.0
        perp -= (perp @ delta) * delta
        perp -= (perp @ ones) * ones
        nperp = np.linalg.norm(perp)
    perp /= nperp

    # w_t = score_t * delta + sqrt(1 - score_t^2) * perp
    sc = np.clip(score, -0.999, 0.999)
    wvec = sc[:, None] * delta[None, :] + \
           np.sqrt(1.0 - sc ** 2)[:, None] * perp[None, :]
    # ensure exact zero-sum / unit norm (numerical clean-up)
    wvec -= wvec.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(wvec, axis=1, keepdims=True)
    nrm[nrm < 1e-8] = 1.0
    wvec = wvec / nrm
    wvec[0] = 0.0                             # PAD stays zero

    # ---- 5. write embedding ------------------------------------------
    w["embed.weight"] = wvec.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # ---- 6. identity blocks ------------------------------------------
    for blk in ["block0", "block1"]:
        for name in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{blk}.{name}"
            if key in w:
                w[key] = np.zeros_like(w[key], dtype=np.float32)
        for name in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
            key = f"{blk}.{name}"
            if key in w:
                if name.endswith("weight"):
                    w[key] = np.ones_like(w[key], dtype=np.float32)
                else:
                    w[key] = np.zeros_like(w[key], dtype=np.float32)

    # ---- 7. final LayerNorm = identity -------------------------------
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # ---- 8. head: head1 - head0 = delta ------------------------------
    head = np.zeros_like(w["head.weight"], dtype=np.float32)
    head[0] = 0.0
    head[1] = delta
    w["head.weight"] = head

    # ---- 9. bias: choose threshold so that training accuracy is max --
    # logit1 - logit0 = (1/n) sum_i score_{t_i} + b
    # We pick b to maximise accuracy on the feedback pool.
    # Compute the mean score per document under the new embedding.
    # pooled_1 - pooled_0 = mean over tokens of w_t  (since gamma=1, beta=0)
    # and delta . pooled = mean over tokens of (delta . w_t) = mean score.
    doc_score = (presence @ score) / length        # [N]
    # choose b by scanning a fine grid (closed-form, no iteration on weights)
    order = np.argsort(doc_score)
    ds = doc_score[order]
    ty = t[order]
    # cumulative sums for fast threshold scan
    cum = np.cumsum(ty)
    total = cum[-1]
    # accuracy if we predict +1 for doc_score > thr:  correct = (total - cum_before) for +1 part ...
    # simpler: evaluate a dense grid
    grid = np.linspace(ds.min() - 0.1, ds.max() + 0.1, 401)
    best_b = 0.0
    best_acc = -1.0
    for b in grid:
        pred = (doc_score + b) > 0
        acc = np.mean((2 * pred.astype(np.int64) - 1) == t)
        if acc > best_acc:
            best_acc = acc
            best_b = b
    w["head.bias"] = np.array([0.0, -best_b], dtype=np.float32)

    # ---- 10. diagnostics ---------------------------------------------
    diag = {
        "train_acc": float(best_acc),
        "best_bias": float(best_b),
        "score_absmax": float(np.max(np.abs(score))),
        "score_mean": float(np.mean(score)),
        "n_nonzero_scores": int(np.count_nonzero(score)),
        "embed_norm_mean": float(np.mean(np.linalg.norm(wvec, axis=1))),
        "doc_score_std": float(np.std(doc_score)),
        "delta_norm": float(np.linalg.norm(delta)),
    }
    return {"weights": w, "diagnostics": diag}
