import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)   # [N,32] pooled LN features
    y = np.asarray(f["y"], dtype=np.int64)
    N, D = X.shape

    # ---- class statistics ----
    m0 = X[y == 0].mean(axis=0)
    m1 = X[y == 1].mean(axis=0)
    Xc0 = X[y == 0] - m0
    Xc1 = X[y == 1] - m1
    C0 = (Xc0.T @ Xc0) / max(len(Xc0) - 1, 1)
    C1 = (Xc1.T @ Xc1) / max(len(Xc1) - 1, 1)
    C = 0.5 * (C0 + C1)

    # ---- shrinkage toward diagonal (closed form) ----
    lam = 0.15
    tr = np.trace(C) / D
    Cs = (1.0 - lam) * C + lam * tr * np.eye(D)

    # ---- LDA direction ----
    d = m1 - m0
    try:
        wdir = np.linalg.solve(Cs + 1e-6 * np.eye(D), d)
    except np.linalg.LinAlgError:
        wdir = np.linalg.pinv(Cs) @ d

    # normalize direction for numerical safety
    nrm = np.linalg.norm(wdir)
    if nrm < 1e-12:
        wdir = np.zeros(D); wdir[1] = 1.0
        nrm = 1.0
    wdir = wdir / nrm

    # ---- pick threshold on training scores ----
    scores = X @ wdir
    # midpoint of class means
    thr = 0.5 * (scores[y == 0].mean() + scores[y == 1].mean())

    # ---- realize in head: logit1 - logit0 = 2c * wdir . pooled + b ----
    # head.weight row0 = -c*wdir, row1 = +c*wdir ; bias0=0, bias1 = 2c*thr
    c = 1.0
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[0] = (-c * wdir).astype(np.float32)
    head_w[1] = ( c * wdir).astype(np.float32)
    head_b = np.zeros(2, dtype=np.float32)
    head_b[1] = float(2.0 * c * thr)

    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # ---- self-check: reproduce logits from cached features ----
    logits = X @ head_w.T + head_b
    pred = np.argmax(logits, axis=1)
    train_acc = float((pred == y).mean())

    # cross-entropy
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    ce = float(-np.mean(np.log(np.clip(p[np.arange(N), y], 1e-12, 1.0))))

    diag = {
        "train_accuracy": train_acc,
        "train_ce": ce,
        "lda_shrinkage": lam,
        "wdir_norm_before": float(nrm),
        "thr": float(thr),
        "head_bias1": float(head_b[1]),
        "feature_std_min": float(X.std(axis=0).min()),
        "feature_std_max": float(X.std(axis=0).max()),
        "n_samples": int(N),
    }
    return {"weights": w, "diagnostics": diag}
