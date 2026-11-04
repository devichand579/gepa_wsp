import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    Y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]                  # 32

    # ---- 1. Fit a bag-of-words linear model on the pool (closed form) ----
    # Build sparse count matrix (unigram counts, length-normalized) for the pool.
    # Use a dense [N, V] float32 matrix (2000 cols) -> 4000x2000 = 8M floats, fine.
    C = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    np.add.at(C, (rows, cols), 1.0)
    # length-normalized counts (exclude pad id 0 from length)
    L = np.maximum((X != 0).sum(axis=1), 1).astype(np.float64)
    Cn = C / L[:, None]

    t = (Y.astype(np.float64) * 2.0 - 1.0)           # {-1,+1}

    # Ridge normal equations on centered features:  (Cn^T Cn + lam I) s = Cn^T t
    # Center columns to remove the intercept coupling.
    mu = Cn.mean(axis=0, keepdims=True)
    Cc = Cn - mu
    tc = t - t.mean()
    lam = 1.0
    A = Cc.T @ Cc + lam * np.eye(V)
    bvec = Cc.T @ tc
    s_ridge = np.linalg.solve(A, bvec)

    # Naive-Bayes log-odds per token (class-conditional counts)
    pos = Cn[t > 0].sum(axis=0)
    neg = Cn[t < 0].sum(axis=0)
    s_nb = np.log((pos + 1.0) / (neg + 1.0))

    # ---- 2. Pick the estimator by explicit forward-pass training accuracy ----
    def train_acc(s):
        score = Cn @ s
        pred = (score > np.median(score)).astype(np.int64)  # threshold ~ balanced
        return (pred == Y).mean()

    cands = {"ridge": s_ridge, "nb": s_nb, "ridge+nb": s_ridge + s_nb}
    accs = {k: float(train_acc(v)) for k, v in cands.items()}
    best_key = max(accs, key=accs.get)
    s = cands[best_key].astype(np.float64)

    # ---- 3. Realize s_t in embed so that d·LN(embed[t]) = s_t ----
    # Choose head direction d = c * u with u = e_0 - mean(e_0) (mean-zero, unit-ish).
    # Then d·w = c * w[0] (since mean(w)=0).  We need w_t[0] = s_t / c.
    # LN constraint: sum(w_t)=0, |w_t|^2 = D.  With w_t[0]=a_t, set the remaining
    # D-1 coords equal to q_t so sum = a_t + (D-1) q_t = 0  =>  q_t = -a_t/(D-1).
    # Then |w_t|^2 = a_t^2 + (D-1) q_t^2 = a_t^2 (1 + 1/(D-1)) = a_t^2 * D/(D-1).
    # Require = D  =>  a_t = ±sqrt(D-1).  That fixes |a_t|, so instead we let the
    # perp component absorb the magnitude: use a fixed unit perp vector p (mean 0,
    # p[0]=0, |p|=1) and set  w_t = a_t * e0 + beta_t * p + gamma_t * 1.
    # Solve the 3 constraints (sum=0, |w|^2=D, w[0]=a_t) with beta_t free.
    u = np.zeros(D); u[0] = 1.0
    u = u - u.mean()                                 # mean-zero direction
    u = u / np.linalg.norm(u)
    # perp basis: pick p orthogonal to u and to 1, mean-zero
    one = np.ones(D) / np.sqrt(D)
    # build p in the orthogonal complement of span{u, one}
    p = np.zeros(D); p[1] = 1.0; p[2] = -1.0
    p = p - (p @ u) * u - (p @ one) * one
    p = p / (np.linalg.norm(p) + 1e-12)

    # We want d·w_t = s_t with d = c*u  =>  c * (u·w_t) = s_t.
    # Write w_t = alpha_t*u + beta_t*p + g_t*one.  u·w_t = alpha_t (u·u=1, others 0).
    # So alpha_t = s_t / c.  Constraints: sum(w_t)=0 -> g_t*sqrt(D)=0 -> g_t=0.
    # |w_t|^2 = alpha_t^2 + beta_t^2 = D  ->  beta_t = sqrt(D - alpha_t^2).
    # Need |alpha_t| <= sqrt(D).  Choose c = max|s_t| / (sqrt(D)*0.9) so it fits.
    smax = float(np.max(np.abs(s))) + 1e-9
    c = smax / (np.sqrt(D) * 0.9)
    alpha = s / c
    alpha = np.clip(alpha, -np.sqrt(D) * 0.999, np.sqrt(D) * 0.999)
    beta = np.sqrt(np.maximum(D - alpha ** 2, 0.0))

    W = alpha[:, None] * u[None, :] + beta[:, None] * p[None, :]
    # enforce exact constraints numerically
    W = W - W.mean(axis=1, keepdims=True)
    nrm = np.sqrt((W ** 2).sum(axis=1, keepdims=True)) + 1e-12
    W = W * (np.sqrt(D) / nrm)

    w["embed.weight"] = W.astype(np.float32)

    # ---- 4. Recompute pooled features with the new embed (exact forward) ----
    # h = embed[x]; LN per token; masked mean pool.
    E = w["embed.weight"].astype(np.float64)         # [V,D]
    H = E[X]                                         # [N,64,D]
    m = H.mean(axis=2, keepdims=True)
    v = H.var(axis=2, keepdims=True)
    Hn = (H - m) / np.sqrt(v + 1e-5)
    mask = (X != 0).astype(np.float64)               # [N,64]
    pooled = (Hn * mask[:, :, None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1.0)

    # ---- 5. Refit head in closed form on the recomputed pooled features ----
    # Multi-class ridge: solve for W_head [2,D], b_head [2].
    P = pooled                                      # [N,D]
    Pm = P.mean(axis=0, keepdims=True)
    Pc = P - Pm
    Yoh = np.zeros((N, 2)); Yoh[np.arange(N), Y] = 1.0
    Yc = Yoh - Yoh.mean(axis=0, keepdims=True)
    lamh = 1.0
    Ah = Pc.T @ Pc + lamh * np.eye(D)
    Bh = Pc.T @ Yc
    Wh = np.linalg.solve(Ah, Bh).T                  # [2,D]
    bh = Yoh.mean(axis=0) - Wh @ Pm.ravel()
    w["head.weight"] = Wh.astype(np.float32)
    w["head.bias"] = bh.astype(np.float32)

    # ---- diagnostics ----
    logits = P @ Wh.T + bh
    pred = logits.argmax(axis=1)
    acc = float((pred == Y).mean())
    diag = {
        "method": best_key,
        "cand_train_acc": accs,
        "train_acc_after": acc,
        "s_absmax": smax,
        "c": float(c),
        "head_w_norm": float(np.linalg.norm(Wh)),
        "embed_row_std": float(w["embed.weight"].std()),
        "embed_row_mean": float(w["embed.weight"].mean()),
    }
    return {"weights": w, "diagnostics": diag}
