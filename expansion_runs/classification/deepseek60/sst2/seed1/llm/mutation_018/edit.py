import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)   # [N,32] pooled features
    y = np.asarray(f["y"], dtype=np.int64)            # [N]
    N, D = X.shape

    # --- center / scale features (guard constant dims) ---
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd

    # --- class means and pooled covariance ---
    m0 = Z[y == 0].mean(axis=0)
    m1 = Z[y == 1].mean(axis=0)
    Z0 = Z[y == 0] - m0
    Z1 = Z[y == 1] - m1
    C = (Z0.T @ Z0 + Z1.T @ Z1) / max(N - 2, 1)

    # --- shrinkage toward diagonal (closed form) ---
    tr = np.trace(C) / D
    lam = 0.15
    Cs = (1.0 - lam) * C + lam * tr * np.eye(D)

    # --- LDA direction ---
    d = np.linalg.solve(Cs, (m1 - m0))

    # scores on training data
    s = Z @ d
    s0 = s[y == 0]
    s1 = s[y == 1]
    thr = 0.5 * (s0.mean() + s1.mean())

    # --- map back to original feature space: s = (X-mu)/sd @ d = X @ a + c ---
    a = d / sd
    c = -float(mu @ a)

    # head: logit0 = 0, logit1 = a @ features + (c - thr)
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[1, :] = a.astype(np.float32)
    head_b = np.zeros(2, dtype=np.float32)
    head_b[1] = np.float32(c - thr)

    # scale so magnitudes stay well within bounds
    mx = float(np.max(np.abs(head_w)))
    if mx > 100.0:
        head_w *= np.float32(100.0 / mx)
        head_b *= np.float32(100.0 / mx)

    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # diagnostics: training accuracy of the new head
    logits = X @ head_w.T + head_b
    pred = np.argmax(logits, axis=1)
    acc = float((pred == y).mean())

    return {
        "weights": w,
        "diagnostics": {
            "train_accuracy": acc,
            "lda_shrinkage": lam,
            "head_row1_norm": float(np.linalg.norm(head_w[1])),
            "head_bias1": float(head_b[1]),
            "thr": float(thr),
            "feature_std_min": float(sd.min()),
            "feature_std_max": float(sd.max()),
        },
    }
