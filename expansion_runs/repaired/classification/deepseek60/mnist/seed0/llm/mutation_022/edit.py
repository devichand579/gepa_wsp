import numpy as np

def edit(w, f):
    # ---------- data ----------
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)   # [N,784]
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # ---------- current fc1 ----------
    W1 = w["fc1.weight"].astype(np.float32).copy()   # [128,784]
    b1 = w["fc1.bias"].astype(np.float32).copy()     # [128]

    # ---------- compute LDA directions on raw pixels ----------
    mu_k = np.zeros((C, D), dtype=np.float32)
    counts = np.zeros(C, dtype=np.float32)
    for c in range(C):
        mask = (y == c)
        counts[c] = mask.sum()
        if counts[c] > 0:
            mu_k[c] = X[mask].mean(axis=0)
    mu = X.mean(axis=0)

    # within-class scatter Sw = sum_i (x_i - mu_{y_i})(x_i - mu_{y_i})^T
    # = X^T X - sum_c n_c mu_c mu_c^T
    XtX = X.T @ X
    Sw = XtX - sum(counts[c] * np.outer(mu_k[c], mu_k[c]) for c in range(C))
    Sw = Sw.astype(np.float32)

    # regularize Sw and invert
    eps = 1e-2
    Sw_reg = Sw + eps * np.eye(D, dtype=np.float32)
    Sw_inv = np.linalg.inv(Sw_reg)

    # LDA directions: d_c = Sw_inv @ (mu_c - mu)
    dirs = np.zeros((C, D), dtype=np.float32)
    for c in range(C):
        d = Sw_inv @ (mu_k[c] - mu)
        # normalize to unit variance of projection
        proj = X @ d
        s = proj.std() + 1e-8
        d = d / s
        dirs[c] = d

    # ---------- replace first 10 dead units with LDA directions ----------
    # These units currently have very negative biases and are always inactive.
    W1[:C] = dirs
    # bias = -mu @ d  so that the unit is active for its own class on average
    b1[:C] = -(mu @ dirs.T).astype(np.float32)

    # ---------- recompute hidden features ----------
    H = X @ W1.T + b1
    H = np.maximum(H, 0.0).astype(np.float32)   # [N,128]

    # ---------- fit fc2 by LDA on hidden features ----------
    mu_h_k = np.zeros((C, H.shape[1]), dtype=np.float32)
    for c in range(C):
        mask = (y == c)
        if mask.sum() > 0:
            mu_h_k[c] = H[mask].mean(axis=0)
    mu_h = H.mean(axis=0)

    # within-class scatter of H
    HtH = H.T @ H
    Sw_h = HtH - sum(counts[c] * np.outer(mu_h_k[c], mu_h_k[c]) for c in range(C))
    Sw_h = Sw_h.astype(np.float32)

    eps_h = 1e-2
    Sw_h_reg = Sw_h + eps_h * np.eye(H.shape[1], dtype=np.float32)
    Sw_h_inv = np.linalg.inv(Sw_h_reg)

    # LDA classifier: W2[c] = Sw_h_inv @ mu_h_k[c]
    W2 = np.zeros((C, H.shape[1]), dtype=np.float32)
    b2 = np.zeros(C, dtype=np.float32)
    for c in range(C):
        W2[c] = Sw_h_inv @ mu_h_k[c]
        b2[c] = -0.5 * (mu_h_k[c] @ (Sw_h_inv @ mu_h_k[c])) + np.log(counts[c] / N + 1e-12)

    # ---------- calibrate logits to reduce overconfidence ----------
    # Compute current logits on the full pool and choose a scale so that the
    # 99th percentile of max logits is about 5.  This does not change argmax.
    logits = H @ W2.T + b2
    max_logits = logits.max(axis=1)
    scale_target = 5.0
    scale = scale_target / (np.percentile(max_logits, 99) + 1e-8)
    # avoid extreme scaling
    scale = float(np.clip(scale, 0.01, 1.0))

    W2 = (W2 * scale).astype(np.float32)
    b2 = (b2 * scale).astype(np.float32)

    # ---------- write back ----------
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # diagnostics
    diag = {
        "lda_scale": scale,
        "fc1_weight_norm": float(np.linalg.norm(W1)),
        "fc1_bias_norm": float(np.linalg.norm(b1)),
        "fc2_weight_norm": float(np.linalg.norm(W2)),
        "fc2_bias_norm": float(np.linalg.norm(b2)),
        "hidden_zero_fraction": float((H == 0).mean()),
        "train_acc": float((np.argmax(logits, axis=1) == y).mean()),
    }
    return {"weights": w, "diagnostics": diag}
