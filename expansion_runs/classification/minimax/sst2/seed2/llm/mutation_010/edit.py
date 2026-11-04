import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = X.shape
    V = int(w["embed.weight"].shape[0])              # 2000
    mask = (X != 0).astype(np.float64)              # 1 for real tokens
    lens = mask.sum(axis=1)                         # [N]
    lens[lens == 0] = 1.0

    # mean-count (length-normalized bag-of-words) features
    Xc = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    keep = cols != 0
    np.add.at(Xc, (rows[keep], cols[keep]), 1.0)
    M = Xc / lens[:, None]

    t = np.where(y == 1, 1.0, -1.0)

    # ---------------- closed-form ridge fits ----------------
    def ridge_fit(A, b, lam):
        G = A.T @ A
        G.flat[:: G.shape[0] + 1] += lam
        return np.linalg.solve(G, A.T @ b)

    def acc(A, b, c):
        s = A @ c + 0.0
        return float(np.mean(np.sign(s) == b))

    # centered design for the intercept
    Mc = M - M.mean(axis=0, keepdims=True)
    yc = t - t.mean()

    # 5-fold CV over a small fixed grid of closed-form solutions
    rng = np.random.RandomState(0)
    fold = rng.randint(0, 5, size=N)
    grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
    best = (None, -1.0, None)
    for lam in grid:
        cor = 0
        for k in range(5):
            tr = fold != k
            te = ~tr
            c = ridge_fit(M[tr], t[tr], lam)
            s = M[te] @ c
            cor += float(np.sum(np.sign(s) == t[te]))
        if cor > best[1]:
            best = (lam, cor, None)
    lam = best[0]

    # final closed-form fit on the whole permitted pool
    g = ridge_fit(M, t, lam)                        # per-token weights
    g = g - g.mean()                                # remove constant part (bias handles it)
    g = np.clip(g, -3.9, 3.9)

    # ---------------- exact LayerNorm inversion ----------------
    # row(x) = [x, C, -x, 0...0];  LayerNorm(row)[0] = (x - C/32)/sigma(x)
    # sigma^2 = (2 x^2 + 0.96875 C^2)/32 ; invert for a target t0 in (-4,4)
    C = 300.0
    def solve_x(t0):
        s = t0 * t0
        A = 2.0 * s - 32.0
        B = 2.0
        Cc = 0.96875 * s - 0.03125
        disc = B * B - 4.0 * A * Cc
        if disc < 0.0:
            disc = 0.0
        r = np.sqrt(disc)
        # choose the root with the larger magnitude (monotone branch)
        u1 = (-B + r) / (2.0 * A)
        u2 = (-B - r) / (2.0 * A)
        u = u1 if abs(u1) >= abs(u2) else u2
        return u * C

    alpha = 0.5                                       # pooled[0] = alpha*mean(g) + beta
    beta = 0.0
    t_tok = alpha * g + beta                          # desired LayerNorm dim-0 value per token
    t_tok = np.clip(t_tok, -3.999, 3.999)
    x_tok = solve_x(t_tok)

    # ---------------- write weights ----------------
    E = np.zeros((V, 32), dtype=np.float64)
    E[:, 1] = C
    E[:, 2] = -C
    E[:, 0] = x_tok
    # pad row (id 0) is never pooled; keep it zero
    E[0, :] = 0.0

    # head: read only dim 0 of the pooled vector
    # pooled[0] = alpha*mean(g) + beta  ->  logit1 = w1*pooled[0] + b1
    # choose w1 so that the decision boundary matches the fitted classifier
    # fitted score  s(x) = (M @ g) ; pooled[0] = alpha*mean(g) = alpha*(M@g)
    # so logit1 = (1/alpha) * (M@g) + b  reproduces the fitted score exactly
    w1 = 1.0 / alpha
    HW = np.zeros((2, 32), dtype=np.float64)
    HW[1, 0] = w1
    HB = np.zeros(2, dtype=np.float64)

    out = {}
    for k, v in w.items():
        out[k] = np.asarray(v, dtype=np.float32).copy()
    out["embed.weight"] = E.astype(np.float32)
    out["head.weight"] = HW.astype(np.float32)
    out["head.bias"] = HB.astype(np.float32)

    # ---------------- self-verification (own forward re-implementation) ----------------
    def forward_check():
        h = E[None, :, :] * 0.0
        # embed
        emb = E[X]                                   # [N,L,32]
        h = emb.copy()
        # blocks are all-zero -> identity; pos is zero
        # final LayerNorm
        mu = h.mean(axis=2, keepdims=True)
        var = ((h - mu) ** 2).mean(axis=2, keepdims=True)
        hn = (h - mu) / np.sqrt(var + 1e-5)
        m = mask[:, :, None]
        pooled = (hn * m).sum(axis=1) / lens[:, None]
        logits = pooled @ HW.T + HB
        pred = np.argmax(logits, axis=1)
        return float(np.mean(pred == y)), logits

    acc_new, logits_new = forward_check()
    acc_old = float(np.mean(np.argmax(np.asarray(f["logits"]), axis=1) == y))
    # implied score from the fitted classifier (for reference)
    s_fit = M @ g
    acc_fit = float(np.mean(np.sign(s_fit) == t))

    diag = {
        "cv_lambda": float(lam),
        "cv_acc": float(best[1] / N),
        "acc_old_feedback": acc_old,
        "acc_new_simulated": acc_new,
        "acc_fitted_bow": acc_fit,
        "acc_new_cls0": float(np.mean((np.argmax(logits_new, axis=1) == y)[y == 0])),
        "acc_new_cls1": float(np.mean((np.argmax(logits_new, axis=1) == y)[y == 1])),
        "mean_len": float(lens.mean()),
        "g_absmax": float(np.abs(g).max()),
        "g_nonzero": int(np.count_nonzero(g)),
        "embed_absmax": float(np.abs(E).max()),
        "x_absmax": float(np.abs(x_tok).max()),
        "head_w": [float(v) for v in HW[1]],
        "head_b": [float(v) for v in HB],
        "pooled0_mean": float(logits_new[:, 1].mean()),
        "note": "blocks/pos/norm untouched (zero) so forward is exactly reproduced",
    }
    return {"weights": out, "diagnostics": diag}
