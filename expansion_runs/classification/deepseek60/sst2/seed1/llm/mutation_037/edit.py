import numpy as np

def edit(w, f):
    X = f["x"]; Y = f["y"]                      # [4000,64], [4000]
    N, L = X.shape
    V = w["embed.weight"].shape[0]              # 2000
    d = w["embed.weight"].shape[1]              # 32

    # ---- 1. length-normalized bag-of-words features ----
    mask = (X != 0).astype(np.float32)          # [N,64]
    lens = mask.sum(1).clip(min=1.0)            # [N]
    BOW = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        m = mask[i]
        np.add.at(BOW[i], row[m > 0], 1.0)
    BOW /= lens[:, None]                        # mean over tokens of onehot

    # ---- 2. closed-form ridge fit of per-token scores + bias ----
    t = (2.0 * Y.astype(np.float64) - 1.0)      # {-1,+1}
    lam = 5.0
    A = BOW.T @ BOW + lam * np.eye(V)           # [V,V]
    rhs = BOW.T @ t                             # [V]
    g = np.linalg.solve(A, rhs)                 # per-token score
    b = float(t.mean() - BOW.mean(0) @ g)       # bias

    # ---- 3. temperature calibration (grid over a few values) ----
    raw = BOW @ g + b
    best_T, best_ce = 1.0, 1e18
    for T in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        z = raw / T
        p1 = 1.0 / (1.0 + np.exp(-z))
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        ce = -np.mean(Y * np.log(p1) + (1 - Y) * np.log(1 - p1))
        if ce < best_ce:
            best_ce, best_T = ce, T
    T = best_T

    # ---- 4. realize g(v) through embed + head ----
    # want  head1 . LN(embed[v]) - head0 . LN(embed[v]) = g(v)
    # choose head1 = c*e1, head0 = -c*e1  =>  g(v) = 2c * t(v)[0]
    # t(v) must be zero-mean and ||t(v)||^2 = d
    gmax = np.abs(g).max() + 1e-9
    c = gmax / (2.0 * np.sqrt(d - 1.0))         # keeps |t0| <= sqrt(d-1)
    c = max(c, 1e-6)

    t0 = g / (2.0 * c)                          # target first coordinate
    t0 = np.clip(t0, -np.sqrt(d - 1.0), np.sqrt(d - 1.0))
    s = -t0                                     # required sum of remaining coords
    Q = d - t0 ** 2                             # required sum of squares remaining
    rem = d - 1                                 # number of remaining coords
    base = s / rem
    dsum = Q - (s ** 2) / rem
    dsum = np.maximum(dsum, 0.0)
    # distribute the residual energy on two coordinates
    r = np.sqrt(dsum)
    Tvec = np.zeros((V, d), dtype=np.float64)
    Tvec[:, 0] = t0
    Tvec[:, 1] = base + r
    Tvec[:, 2] = base - r
    Tvec[:, 3:] = base
    # fix any tiny drift so sum is exactly 0 and norm exactly sqrt(d)
    Tvec[:, 1] += -Tvec.sum(1) / (d - 1)
    nrm = np.linalg.norm(Tvec, axis=1, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    Tvec = Tvec * (np.sqrt(d) / nrm)

    # ---- 5. write weights ----
    E = np.zeros_like(w["embed.weight"], dtype=np.float32)
    E[:] = Tvec.astype(np.float32)
    E[0] = 0.0                                  # pad row unused (masked)
    w["embed.weight"] = E

    head = np.zeros_like(w["head.weight"], dtype=np.float32)
    head[1, 0] = c
    head[0, 0] = -c
    w["head.weight"] = head

    # bias: logits1 - logits0 = raw/T ; logits1+logits0 = 0
    # so logits1 = raw/(2T), logits0 = -raw/(2T)
    # head.bias must reproduce that: b1-b0 = b/T, b1+b0 = 0
    b1 = (b / T) / 2.0
    w["head.bias"] = np.array([-b1, b1], dtype=np.float32)

    # keep everything else as-is (already zero / identity)
    for k in ["pos.weight"]:
        w[k] = np.zeros_like(w[k], dtype=np.float32)

    diag = {
        "ridge_lambda": lam,
        "temperature": float(T),
        "train_ce": float(best_ce),
        "train_acc": float(np.mean((raw > 0).astype(np.int64) == Y)),
        "g_absmax": float(gmax),
        "head_scale_c": float(c),
        "bias": float(b),
    }
    return {"weights": w, "diagnostics": diag}
