import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)   # [4000,32] pooled features
    y = np.asarray(f["y"], dtype=np.int64)            # [4000]

    # ---- class means / pooled covariance with ridge shrinkage ----
    m0 = X[y == 0].mean(axis=0)
    m1 = X[y == 1].mean(axis=0)
    Xc = X - np.where(y[:, None] == 0, m0[None, :], m1[None, :])
    n, d = X.shape
    S = (Xc.T @ Xc) / max(n - 2, 1)
    lam = 1e-3 * np.trace(S) / d + 1e-8
    S_reg = S + lam * np.eye(d)

    # ---- LDA direction w ∝ S^{-1}(m1-m0) ----
    diff = m1 - m0
    wdir = np.linalg.solve(S_reg, diff)

    # ---- choose scale so logits are well-conditioned (unit-variance scores) ----
    scores = X @ wdir
    s_std = scores.std() + 1e-8
    scale = 1.0 / s_std

    # ---- bias: midpoint of class-mean scores (in scaled units) ----
    c0 = (m0 @ wdir) * scale
    c1 = (m1 @ wdir) * scale
    thr = 0.5 * (c0 + c1)

    # head.weight rows: [negative, positive]; positive row = +wdir*scale
    head_w = np.zeros((2, d), dtype=np.float32)
    head_w[1] = (wdir * scale).astype(np.float32)
    head_w[0] = (-wdir * scale).astype(np.float32)
    head_b = np.array([thr, -thr], dtype=np.float32)

    # ---- write back, preserving names/shapes and finiteness ----
    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # ---- diagnostics ----
    logits = X @ head_w.T + head_b[None, :]
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, 1.0)).mean())

    diag = {
        "train_accuracy": acc,
        "train_cross_entropy": ce,
        "lda_scale": float(scale),
        "w_norm": float(np.linalg.norm(wdir)),
        "head_w_norm": float(np.linalg.norm(head_w)),
        "bias": [float(head_b[0]), float(head_b[1])],
        "ridge_lambda": float(lam),
    }
    return {"weights": w, "diagnostics": diag}
