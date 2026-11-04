import numpy as np

def edit(w, f):
    # Extract feedback arrays
    X = f["features"].astype(np.float64)          # [2000, 1568]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = X.shape
    K = 10

    # Current head and logits
    W_cur = w["head.weight"].astype(np.float64)   # [10, 1568]
    b_cur = w["head.bias"].astype(np.float64)     # [10]
    logits_cur = f["logits"].astype(np.float64)   # [2000, 10]

    def softmax_ce(logits, labels):
        # stable softmax CE
        m = np.max(logits, axis=1, keepdims=True)
        z = logits - m
        logsumexp = np.log(np.sum(np.exp(z), axis=1, keepdims=True))
        logp = z - logsumexp
        return -np.mean(logp[np.arange(len(labels)), labels])

    ce_cur = softmax_ce(logits_cur, y)

    # Build one-hot targets
    Y = np.zeros((N, K), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with bias column
    Xa = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # [2000, 1569]

    # Ridge regression: solve (Xa^T Xa + lambda I) W = Xa^T Y
    # Use a small lambda for numerical stability.  We try a couple of fixed
    # closed-form candidates and keep the best by full-pool CE.
    candidates = []

    for lam in (1e-3, 1e-1, 1.0):
        A = Xa.T @ Xa + lam * np.eye(D + 1, dtype=np.float64)
        B = Xa.T @ Y
        W_full = np.linalg.solve(A, B)          # [1569, 10]
        W_new = W_full[:-1, :].T                # [10, 1568]
        b_new = W_full[-1, :]                   # [10]

        # Scale to max abs <= 1000
        max_abs = max(np.max(np.abs(W_new)), np.max(np.abs(b_new)), 1e-12)
        scale = 1000.0 / max_abs
        W_new *= scale
        b_new *= scale

        logits_new = X @ W_new.T + b_new
        ce_new = softmax_ce(logits_new, y)
        acc_new = np.mean(np.argmax(logits_new, axis=1) == y)
        candidates.append((ce_new, acc_new, W_new, b_new, lam))

    # Also consider simply rescaling the current head to max 1000
    max_cur = max(np.max(np.abs(W_cur)), np.max(np.abs(b_cur)), 1e-12)
    scale_cur = 1000.0 / max_cur
    W_cur_scaled = W_cur * scale_cur
    b_cur_scaled = b_cur * scale_cur
    logits_cur_scaled = X @ W_cur_scaled.T + b_cur_scaled
    ce_cur_scaled = softmax_ce(logits_cur_scaled, y)
    acc_cur_scaled = np.mean(np.argmax(logits_cur_scaled, axis=1) == y)
    candidates.append((ce_cur_scaled, acc_cur_scaled, W_cur_scaled, b_cur_scaled, "current_rescaled"))

    # Pick candidate with best CE, tie-break by accuracy
    best = min(candidates, key=lambda t: (t[0], -t[1]))
    ce_best, acc_best, W_best, b_best, tag = best

    # Only replace if it improves CE and does not hurt accuracy
    if ce_best < ce_cur and acc_best >= 0.999:
        w["head.weight"] = W_best.astype(np.float32)
        w["head.bias"] = b_best.astype(np.float32)

    # Diagnostics
    diag = {
        "ce_current": float(ce_cur),
        "ce_best": float(ce_best),
        "acc_best": float(acc_best),
        "chosen": str(tag),
        "max_abs_head_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_head_bias": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diag}
