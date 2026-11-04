def edit(w, f):
    X = np.asarray(f["x"], dtype=np.float64).reshape(len(f["x"]), -1)   # [N,784]
    y = np.asarray(f["y"]).astype(np.int64)
    N, D = X.shape
    K = 10

    # ---------- class moments ----------
    mu_bar = X.mean(axis=0)
    Xc = X - mu_bar
    mu_c = np.zeros((K, D))
    cnt = np.bincount(y, minlength=K).astype(np.float64)
    for c in range(K):
        m = y == c
        if m.any():
            mu_c[c] = X[m].mean(axis=0)
    # pooled within-class scatter:  Xc^T Xc  -  sum_c n_c mu_c mu_c^T
    Sw = (Xc.T @ Xc - (mu_c.T * cnt) @ mu_c) / max(N - K, 1)
    tr = np.trace(Sw) / D
    Sw = 0.5 * (Sw + Sw.T)

    # ---------- shrinkage + whitening (within-class metric) ----------
    alpha = 0.30
    Swr = Sw + (alpha * tr + 1e-8) * np.eye(D)
    evals, evecs = np.linalg.eigh(Swr)
    evals = np.maximum(evals, 1e-8)
    Wwh = (evecs / np.sqrt(evals)).astype(np.float64)      # [784,784]
    Z = Xc @ Wwh                                        # whitened pixels
    dmu = (mu_c - mu_bar) @ Wwh                         # [10,784] whitened class-mean diffs
    norms = np.linalg.norm(dmu, axis=1)
    keep = norms > 1e-9
    dmu_n = np.zeros_like(dmu)
    dmu_n[keep] = dmu[keep] / norms[keep, None]

    # ---------- hidden layer design (fc1) ----------
    H = 128
    rows = []          # [784] weight vectors
    biases = []        # [1]
    # (a) class-anchor splines: relu(s_c - t) for 5 thresholds per class
    for c in range(K):
        if not keep[c]:
            continue
        s = Z @ dmu_n[c]                                # LDA-ish score
        qs = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
        for t in qs:
            rows.append(dmu_n[c])
            biases.append(-float(t))
    # (b) random directions in the whitened space
    rng = np.random.default_rng(12345)
    n_rand = H - len(rows)
    if n_rand > 0:
        R = rng.standard_normal((n_rand, D))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        for r in R:
            rows.append(r)
            biases.append(0.0)
    A = np.asarray(rows)                                 # [H,784]
    b = np.asarray(biases)                               # [H]
    W1 = A.T                                             # [784,128] (fc1.weight = W1.T)

    # ---------- hidden activations on the full pool ----------
    Hf = np.maximum(0.0, X @ W1 + b)                     # [N,128]

    # ---------- closed-form ridge on one-hot targets, lambda by PRESS/LOO ----------
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1.0
    Y = Y - 1.0 / K
    Y = Y * K                                             # scale so targets are ~+-1
    HtH = Hf.T @ Hf
    HtY = Hf.T @ Y
    I = np.eye(Hf.shape[1])
    trH = np.trace(HtH) / Hf.shape[1]
    lam_grid = np.array([1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]) * max(trH, 1e-12)
    best = (np.inf, None, None)
    for lam in lam_grid:
        Areg = HtH + lam * I
        try:
            W2 = np.linalg.solve(Areg, HtY)              # [128,10]
        except np.linalg.LinAlgError:
            continue
        R = Y - Hf @ W2
        Hinv = np.linalg.inv(Areg)
        # PRESS / leave-one-out residual for ridge (closed form)
        Hd = Hf @ Hinv
        d = 1.0 - np.sum(Hf * (Hd @ np.eye(Hf.shape[1])).T, axis=1) if False else 1.0 - np.einsum('ij,ji->i', Hf, Hinv.T @ Hf.T.T) * 0
        # exact: h_ii = sum_j Hf[i,j] * Hinv[j,:] @ Hf[i,:]
        hdiag = np.einsum('ij,jk,ik->i', Hf, Hinv, Hf)
        den = np.maximum(1.0 - hdiag, 1e-6)
        loo = R / den[:, None]
        err = float(np.mean(np.sum(loo * loo, axis=1)))
        if err < best[0]:
            best = (err, lam, W2)
    err_loo, lam, W2 = best

    # ---------- assemble weights ----------
    fc1w = np.ascontiguousarray(W1.T.astype(np.float32))          # [128,784]
    fc1b = np.ascontiguousarray(b.astype(np.float32))             # [128]
    fc2w = np.ascontiguousarray(W2.T.astype(np.float32))          # [10,128]
    # calibrate bias: mean logit over pool -> centered logits
    Hf2 = np.maximum(0.0, X @ W1 + b)
    mean_logit = (Hf2 @ W2).mean(axis=0)
    fc2b = np.ascontiguousarray((-mean_logit).astype(np.float32))  # [10]

    # safety: finite, bounded
    def _clip(a, lim=1.0):
        a = np.nan_to_num(a, nan=0.0, posinf=lim, neginf=-lim)
        return np.clip(a, -lim, lim).astype(np.float32)

    fc1w = _clip(fc1w, 1.0)
    fc1b = _clip(fc1b, 1.0)
    fc2w = _clip(fc2w, 10.0)
    fc2b = _clip(fc2b, 10.0)

    w["fc1.weight"] = fc1w
    w["fc1.bias"] = fc1b
    w["fc2.weight"] = fc2w
    w["fc2.bias"] = fc2b

    # ---------- diagnostics ----------
    Hf3 = np.maximum(0.0, X @ W1 + b)
    logits = Hf3 @ W2 - mean_logit
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    mx = logits.max(axis=1, keepdims=True)
    lse = mx + np.log(np.exp(logits - mx).sum(axis=1, keepdims=True))
    ce = float(np.mean(lse - logits[np.arange(N), y]))
    accs = {}
    for c in range(K):
        m = y == c
        if m.any():
            accs[str(c)] = round(float((pred[m] == c).mean()), 4)
    diag = {
        "method": "within-class whitening + class-anchor/random relu features + closed-form ridge (PRESS lambda)",
        "n_train": int(N),
        "train_acc": round(acc, 4),
        "train_ce": round(ce, 4),
        "loo_mse": round(float(err_loo), 5),
        "lambda_rel": float(lam / max(trH, 1e-12)),
        "per_class_acc": accs,
        "fc1_absmax": round(float(np.abs(fc1w).max()), 5),
        "fc2_absmax": round(float(np.abs(fc2w).max()), 5),
        "hidden_active_frac": round(float((Hf3 > 0).mean()), 4),
        "hidden_mean": round(float(Hf3.mean()), 4),
        "logit_std": round(float(logits.std()), 4),
        "old_acc": 0.098,
    }
    return {"weights": w, "diagnostics": diag}
