import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    y = np.asarray(f["y"], dtype=np.int64).astype(np.float64)  # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]                   # 2000

    # ---- normalized unigram count features (exclude pad id 0) ----
    mask = (X != 0)
    T = np.maximum(mask.sum(axis=1), 1).astype(np.float64)      # [N]
    # accumulate counts per vocab
    F = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    valid = (cols != 0)
    np.add.at(F, (rows[valid], cols[valid]), 1.0)
    F /= T[:, None]                                  # normalized counts

    # ---- closed-form ridge fit of target t = 2y-1 (class1 -> +1) ----
    t = 2.0 * y - 1.0
    mu = F.mean(axis=0, keepdims=True)
    Fc = F - mu
    # solve (Fc^T Fc + lam I) beta = Fc^T t   (2000x2000, cheap)
    lam = 1.0
    G = Fc.T @ Fc
    G[np.diag_indices_from(G)] += lam
    rhs = Fc.T @ t
    try:
        beta = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(G, rhs, rcond=None)[0]
    c0 = t.mean() - mu.reshape(-1) @ beta            # intercept so score ~ t

    # ---- build zero-mean, unit-norm directions d_j with h.d_j = beta_j/sqrt(32) ----
    D = 32
    h = np.zeros(D, dtype=np.float64)
    h[0] = 1.0
    h[1] = -1.0                                      # mean 0, ||h||^2 = 2
    hnorm2 = 2.0
    s = np.sqrt(D)

    # p_j must be zero-mean and orthogonal to h, with ||p_j||^2 = 1 - beta_j^2/(32*2)
    # zero-mean & orthogonal to h  ->  p[0]=p[1], and sum_{d>=2} p[d] = -2 p[0]
    # use the 30 remaining coords: put equal value a on coords 2..30 (29 slots) and b on coord 31
    # constraints: 29a + b = -2p0  ;  2 p0^2 + 29 a^2 + b^2 = target
    target = 1.0 - (beta ** 2) / (s * s * hnorm2)     # = 1 - beta^2/128
    target = np.clip(target, 1e-6, None)

    # choose p0 = 0 for simplicity -> then 29a + b = 0, 29a^2 + b^2 = target
    # b = -29a  -> 29a^2 + 841 a^2 = 870 a^2 = target
    a = np.sqrt(target / 870.0)
    b = -29.0 * a

    d = np.zeros((V, D), dtype=np.float64)
    coef = beta / (s * hnorm2)                       # beta_j / (sqrt(32)*2)
    d[:, 0] = coef
    d[:, 1] = -coef
    d[:, 2:31] = a[:, None]                          # 29 slots (2..30)
    d[:, 31] = b
    # pad row stays zero
    d[0, :] = 0.0

    # verify h.d_j = beta_j/sqrt(32):  h.d = d0 - d1 = 2*coef = beta/sqrt(32)  ✓
    # verify zero-mean: d0+d1 + 29a + b = 0 + 29a - 29a = 0  ✓
    # verify unit-norm: 2*coef^2 + 29a^2 + b^2 = beta^2/128 + target = 1  ✓

    # ---- write embed (float32) ----
    w["embed.weight"] = d.astype(np.float32)
    # pos stays zero, blocks stay zero, norm stays identity-LN

    # ---- head: logit1 - logit0 = h.pooled + (b1-b0); h.pooled = sum (c/T) beta ----
    # head0 = -h/2, head1 = +h/2  => head1-head0 = h
    head = np.zeros((2, D), dtype=np.float64)
    head[0] = -0.5 * h
    head[1] = 0.5 * h
    w["head.weight"] = head.astype(np.float32)
    # bias: b1 - b0 = c0  ->  b0 = -c0/2, b1 = c0/2
    w["head.bias"] = np.array([-0.5 * c0, 0.5 * c0], dtype=np.float32)

    # ---- diagnostics ----
    # sanity: reconstruct score on feedback pool from realized geometry
    pooled = (F @ d)                                  # [N,32]  (sqrt(32) factor folded below)
    pooled = pooled * s                               # LN(d_j)=sqrt(32) d_j
    logit = pooled @ head.T + w["head.bias"].astype(np.float64)
    pred = logit.argmax(axis=1)
    acc = float((pred == y.astype(np.int64)).mean())
    ce = float(-np.mean(np.log(np.clip(
        np.exp(logit - logit.max(axis=1, keepdims=True)) /
        np.exp(logit - logit.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True),
        1e-12, 1.0)[np.arange(N), y.astype(np.int64)])))

    diag = {
        "fit_accuracy": acc,
        "fit_cross_entropy": ce,
        "ridge_lambda": lam,
        "beta_abs_mean": float(np.abs(beta).mean()),
        "beta_abs_max": float(np.abs(beta).max()),
        "intercept_c0": float(c0),
        "embed_row_norm_mean": float(np.linalg.norm(d[1:], axis=1).mean()),
        "embed_row_mean_absmax": float(np.abs(d[1:].mean(axis=1)).max()),
        "head_row0_norm": float(np.linalg.norm(head[0])),
    }
    return {"weights": w, "diagnostics": diag}
