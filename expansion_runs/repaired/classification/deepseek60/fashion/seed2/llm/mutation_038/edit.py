import numpy as np

def edit(w, f):
    W = np.array(w["head.weight"], dtype=np.float64)   # [10,1568]
    b = np.array(w["head.bias"], dtype=np.float64)     # [10]
    H = np.asarray(f["features"], dtype=np.float64)    # [N,1568]
    y = np.asarray(f["y"], dtype=np.int64)             # [N]
    logits = np.array(f["logits"], dtype=np.float64)   # [N,10]

    N, D = H.shape
    C = W.shape[0]
    pred = np.argmax(logits, axis=1)
    wrong = list(np.where(pred != y)[0])

    diag = {"n_wrong_before": int(len(wrong)), "fixed": []}

    for i in wrong:
        t = int(y[i]); p = int(pred[i])
        h = H[i]
        norm2 = float(h @ h)
        if norm2 < 1e-12:
            continue

        # current margin of the violated sample (t vs p)
        m_i = float(logits[i, t] - logits[i, p])
        # need m_i + 2*alpha*norm2 > 0  ->  lower bound
        alpha_lo = max(0.0, -m_i / (2.0 * norm2))

        # correlation of h with every feature row
        s = H @ h  # [N]

        # upper bound: no currently-correct sample may flip
        alpha_hi = np.inf
        for j in range(N):
            if j == i:
                continue
            yj = int(y[j])
            sj = float(s[j])
            if sj == 0.0:
                continue
            base = (1.0 if yj == t else 0.0) - (1.0 if yj == p else 0.0)
            lj = logits[j]
            myj = lj[yj]
            for c in range(C):
                if c == yj:
                    continue
                coef = base - (1.0 if c == t else 0.0) + (1.0 if c == p else 0.0)
                A = coef * sj
                if A < 0.0:
                    ub = -(myj - lj[c]) / A
                    if ub < alpha_hi:
                        alpha_hi = ub

        if not np.isfinite(alpha_hi):
            alpha_hi = alpha_lo + max(1e-3, 0.1 * alpha_lo + 1e-3)
        if alpha_hi <= alpha_lo:
            alpha = alpha_lo + 1e-4
        else:
            alpha = 0.5 * (alpha_lo + alpha_hi)

        # closed-form rank-1 correction (minimal-norm solution of the 2-row system)
        W[t] += alpha * h
        W[p] -= alpha * h
        # propagate logit changes for subsequent iterations
        logits[:, t] += alpha * s
        logits[:, p] -= alpha * s

        diag["fixed"].append({
            "idx": int(i), "true": t, "pred": p,
            "alpha": float(alpha), "alpha_lo": float(alpha_lo),
            "alpha_hi": float(alpha_hi), "norm2": float(norm2),
            "margin_before": float(m_i),
            "margin_after": float(m_i + 2.0 * alpha * norm2),
        })

    W = np.clip(W, -1000.0, 1000.0)
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # verification on the full permitted pool
    new_logits = H @ W.T + b
    new_pred = np.argmax(new_logits, axis=1)
    diag["acc_after"] = float(np.mean(new_pred == y))
    diag["n_wrong_after"] = int(np.sum(new_pred != y))
    diag["max_abs_W"] = float(np.max(np.abs(W)))

    return {"weights": w, "diagnostics": diag}
