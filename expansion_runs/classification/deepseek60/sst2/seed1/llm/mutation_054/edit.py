import numpy as np
import math

def edit(w, f):
    # Extract feedback data
    x = f["x"]          # (4000, 64) int64
    y = f["y"]          # (4000,) int64, 0/1
    N, L = x.shape
    V = 2000            # vocabulary size from embed.weight shape

    # Convert labels to ±1
    y_signed = 2.0 * y.astype(np.float32) - 1.0

    # Document lengths (non-padding tokens)
    lengths = (x != 0).sum(axis=1).astype(np.float32)
    lengths[lengths == 0] = 1.0

    # Build count matrix C (N, V) and mean feature matrix X = C / lengths
    C = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        ids = x[i][x[i] != 0]
        if ids.size > 0:
            C[i] = np.bincount(ids, minlength=V)
    X = C / lengths[:, None]

    mean_X = X.mean(axis=0)

    # Candidate 1: Ridge regression on ±1 targets
    best_w = None
    best_b = None
    best_acc = -1.0

    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        A = X.T @ X + lam * np.eye(V, dtype=np.float32)
        b_vec = X.T @ y_signed
        try:
            w_ridge = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            continue
        bias_ridge = - w_ridge @ mean_X
        scores = X @ w_ridge + bias_ridge
        preds = (scores > 0).astype(np.int64)
        acc = (preds == y).mean()
        if acc > best_acc:
            best_acc = acc
            best_w = w_ridge
            best_b = bias_ridge

    # Candidate 2: Naive Bayes log-odds
    pos_counts = C[y == 1].sum(axis=0)
    neg_counts = C[y == 0].sum(axis=0)
    alpha = 1.0
    pos_total = pos_counts.sum() + alpha * V
    neg_total = neg_counts.sum() + alpha * V
    w_nb = np.log((pos_counts + alpha) / pos_total) - np.log((neg_counts + alpha) / neg_total)
    bias_nb = - w_nb @ mean_X
    scores_nb = X @ w_nb + bias_nb
    preds_nb = (scores_nb > 0).astype(np.int64)
    acc_nb = (preds_nb == y).mean()
    if acc_nb > best_acc:
        best_acc = acc_nb
        best_w = w_nb
        best_b = bias_nb

    # Fallback: if something went wrong, use current head
    if best_w is None:
        best_w = np.zeros(V, dtype=np.float32)
        best_b = 0.0

    # Realize best_w and best_b in the network.
    # We want score = mean_j w_{x_j} + b.
    # Set head difference u = (c, 0, ..., 0), and embed rows z_j with
    # LN(z_j)[0] = w_j / c, mean 0, variance 1.
    max_abs_w = np.max(np.abs(best_w))
    if max_abs_w < 1e-8:
        c = 1.0
    else:
        c = max_abs_w / math.sqrt(31.0) * 1.05  # slight margin

    a = best_w / c  # desired z_j[0]

    # Construct zero-mean unit-variance vectors z_j of shape (V, 32)
    # z_j[0] = a_j, and we add a fixed orthogonal component to satisfy variance.
    # Base vector: [a_j, -a_j/31, ..., -a_j/31]
    z = np.zeros((V, 32), dtype=np.float32)
    z[:, 0] = a
    z[:, 1:] = -a[:, None] / 31.0

    # Fixed direction d with d[0]=0, sum(d)=0, norm(d)=1
    d = np.zeros(32, dtype=np.float32)
    d[1] = 1.0 / math.sqrt(2.0)
    d[2] = -1.0 / math.sqrt(2.0)

    # Scale factor s_j = sqrt(32 * (1 - a_j^2 / 31))
    s = np.sqrt(np.maximum(0.0, 32.0 * (1.0 - a**2 / 31.0)))
    z += s[:, None] * d[None, :]

    # Write embed weights
    w["embed.weight"][:] = z.astype(np.float32)

    # Set head weights: head[1] - head[0] = (c, 0, ..., 0)
    head_w = np.zeros((2, 32), dtype=np.float32)
    head_w[1, 0] = c
    head_w[0, 0] = 0.0
    w["head.weight"][:] = head_w

    # Set head bias: difference = best_b
    head_b = np.zeros(2, dtype=np.float32)
    head_b[1] = best_b / 2.0
    head_b[0] = -best_b / 2.0
    w["head.bias"][:] = head_b

    # Ensure all other parameters are finite and within bounds (they are already zero/one)
    for k in w:
        if not np.all(np.isfinite(w[k])):
            w[k] = np.nan_to_num(w[k], nan=0.0, posinf=1000.0, neginf=-1000.0)
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "method": "closed_form_ridge_or_nb_bag_of_words",
        "best_train_acc": float(best_acc),
        "c": float(c),
        "max_abs_w": float(max_abs_w),
        "bias": float(best_b),
    }
    return {"weights": w, "diagnostics": diagnostics}
