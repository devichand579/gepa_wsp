def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = x.shape
    V = w["embed.weight"].shape[0]

    mask = (x != 0)
    n = np.maximum(mask.sum(1), 1).astype(np.float64)   # [N]

    # counts per token id (bag of words), length-normalised
    Xc = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), mask.sum(1))
    cols = x[mask]
    np.add.at(Xc, (rows, cols), 1.0)
    X = Xc / n[:, None]

    z = 2.0 * y.astype(np.float64) - 1.0            # +-1 targets
    Xa = np.hstack([X, np.ones((N, 1))])             # intercept column
    G = Xa.T @ Xa                                   # [V+1, V+1]
    rhs = Xa.T @ z                                  # [V+1]

    # ---------------- closed-form ridge over a few lambdas (single batched solve) ----
    lams = np.array([0.003, 0.01, 0.03, 0.1, 0.3], dtype=np.float64)
    P = V + 1
    Gs = G[None, :, :].repeat(len(lams), 0).copy()
    idx = np.arange(P)
    Gs[:, idx, idx] += lams[:, None] * np.concatenate([[1.0], np.zeros(P - 1)])[None, :]
    W = np.linalg.solve(Gs, rhs[None, :].repeat(len(lams), 0))   # [nl,P]
    sc = Xa @ W.T                                            # [N,nl]
    accs = ((sc > 0).astype(np.int64) == y[:, None]).mean(0)
    best = int(np.argmax(accs))
    wv = W[best]
    s_tok = wv[:V].copy()
    b_int = float(wv[V])
    lam_best = float(lams[best])

    # ---------------- design the per-token embedding so LayerNorm keeps magnitude ----
    # e_t = (s_t, c + delta_1..delta_31), sum(delta)=0, large spread A
    c = 0.0
    delta = np.zeros(31, dtype=np.float64)
    half = 15
    delta[:half] = -1.0
    delta[half:] = 1.0
    delta *= 1.0
    A = float((delta ** 2).sum())          # = 30
    # choose spread so that |s-c| << sqrt(A/0.96875) over the score range
    s_absmax = float(np.abs(s_tok).max()) if np.any(s_tok != 0) else 1.0
    spread = max(1.0, 4.0 * s_absmax)
    delta = delta * (spread / np.sqrt(A))  # keep A, rescale magnitudes
    A = float((delta ** 2).sum())

    E = np.zeros((V, 32), dtype=np.float64)
    E[:, 0] = s_tok
    E[:, 1:] = (c + delta)[None, :]

    # ---------------- forward verification (exact deployed net) ----------------
    def forward(Ew, headw, headb, posw):
        h = Ew[x]                                  # [N,L,32]
        h = h + posw[None, :, :]
        mu = h.mean(-1, keepdims=True)
        var = h.var(-1, keepdims=True)
        z = (h - mu) / np.sqrt(var + 1e-5)
        pooled = (z * mask[:, :, None]).sum(1) / n[:, None]
        logits = pooled @ headw.T + headb
        return logits, pooled

    def metrics(logits):
        pred = logits.argmax(1)
        acc = float((pred == y).mean())
        p = np.exp(logits - logits.max(1, keepdims=True))
        p = p / p.sum(1, keepdims=True)
        ce = float(-np.log(np.clip(p[np.arange(N), y], 1e-12, None)).mean())
        return acc, ce

    headw = np.array([[-1.0] + [0.0] * 31, [1.0] + [0.0] * 31], dtype=np.float64)
    posw = np.zeros((64, 32), dtype=np.float64)

    # choose head bias so the decision threshold matches the fitted intercept
    # pooled0 ~ kappa * (Xw + b_int);  kappa = (31/32)*sqrt(32)/sqrt(A) approx
    kappa = (31.0 / 32.0) * np.sqrt(32.0) / np.sqrt(A)
    # we want logit1-logit0 = 2*kappa*(score) -> threshold at score=0 handled by b_int
    # pooled0 ~ kappa*(Xw) + const ; set bias so that decision uses score+b_int
    headb = np.array([-b_int * kappa, b_int * kappa], dtype=np.float64)

    # ---------------- write parameters ----------------
    w["embed.weight"][:] = E.astype(np.float32)
    w["pos.weight"][:] = posw.astype(np.float32)
    w["head.weight"][:] = headw.astype(np.float32)
    w["head.bias"][:] = headb.astype(np.float32)

    # ---------------- diagnostics ----------------
    lg_new, pooled_new = forward(E, headw, headb, posw)
    acc_new, ce_new = metrics(lg_new)
    lg_old = np.asarray(f["logits"], dtype=np.float64)
    acc_old, ce_old = metrics(lg_old)
    pred_new = lg_new.argmax(1)
    cls_acc = [float((pred_new[y == c] == c).mean()) for c in (0, 1)]

    diag = {
        "lambda": lam_best,
        "ridge_train_acc": float(accs[best]),
        "pool_acc_before": acc_old,
        "pool_ce_before": ce_old,
        "pool_acc_after": acc_new,
        "pool_ce_after": ce_new,
        "acc_delta": acc_new - acc_old,
        "class_acc_after": cls_acc,
        "intercept_b": b_int,
        "kappa": float(kappa),
        "A_spread": A,
        "s_absmax": s_absmax,
        "n_tokens_nonzero": int((np.abs(s_tok) > 1e-9).sum()),
        "embed_absmax": float(np.abs(E).max()),
        "pooled0_std_after": float(pooled_new[:, 0].std()),
        "pooled_other_absmax": float(np.abs(pooled_new[:, 1:]).max()),
        "top_tokens_by_abs_weight": [
            [int(t), round(float(s_tok[t]), 4)]
            for t in np.argsort(-np.abs(s_tok))[:15]
        ],
    }
    return {"weights": w, "diagnostics": diag}
