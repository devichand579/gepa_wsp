import numpy as np

def edit(w, f):
    # Read feedback data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Basic statistics
    mu_all = X.mean(axis=0)
    Xc = X - mu_all

    # Class means
    mu = np.zeros((C, D), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        counts[c] = idx.sum()
        if counts[c] > 0:
            mu[c] = X[idx].mean(axis=0)

    # Ridge linear classifier on raw pixels
    # W_lin: (D, C)
    lam_lin = 1e-2
    XtX = Xc.T @ Xc
    XtX_reg = XtX + lam_lin * np.eye(D)
    XtY = Xc.T @ (Y - Y.mean(axis=0, keepdims=True))
    W_lin = np.linalg.solve(XtX_reg, XtY)  # (D, C)
    W_lin = W_lin.T  # (C, D)

    # LDA directions via class-mean span
    A = mu - mu_all  # (C, D)
    # Within-class scatter projected onto class-mean span
    X_centered = X.copy()
    for c in range(C):
        idx = (y == c)
        if counts[c] > 0:
            X_centered[idx] -= mu[c]
    Z = X_centered @ A.T  # (N, C)
    S_w_small = Z.T @ Z   # (C, C)
    S_b_small = A @ A.T   # (C, C)
    S_w_small += 1e-3 * np.eye(C)
    # Solve S_b u = lambda S_w u
    M = np.linalg.solve(S_w_small, S_b_small)
    evals, evecs = np.linalg.eig(M)
    # Sort by real part of eigenvalues
    order = np.argsort(-np.real(evals))
    evecs = np.real(evecs[:, order])
    # Take up to 9 LDA directions
    n_lda = min(9, C - 1)
    V_lda = (A.T @ evecs[:, :n_lda]).T  # (n_lda, D)
    # Normalize
    for i in range(n_lda):
        nrm = np.linalg.norm(V_lda[i])
        if nrm > 1e-12:
            V_lda[i] /= nrm

    # Class-mean directions (centered)
    V_mean = A.copy()  # (C, D)
    for i in range(C):
        nrm = np.linalg.norm(V_mean[i])
        if nrm > 1e-12:
            V_mean[i] /= nrm

    # Assemble weight rows
    rows = []
    # Linear classifier: positive and negative parts
    for i in range(C):
        rows.append(W_lin[i])
        rows.append(-W_lin[i])
    # LDA: positive and negative parts
    for i in range(n_lda):
        rows.append(V_lda[i])
        rows.append(-V_lda[i])
    # Class means
    for i in range(C):
        rows.append(V_mean[i])

    n_fixed = len(rows)
    n_random = 128 - n_fixed
    if n_random < 0:
        # Should not happen, but truncate if needed
        rows = rows[:128]
        n_random = 0

    # Random ReLU features
    rng = np.random.RandomState(1234)
    for _ in range(n_random):
        v = rng.randn(D)
        nrm = np.linalg.norm(v)
        if nrm > 1e-12:
            v /= nrm
        rows.append(v)

    W1 = np.array(rows, dtype=np.float64)  # (128, D)

    # Center each unit's pre-activation by subtracting its mean
    # Compute projections on training data
    P = X @ W1.T  # (N, 128)
    b1 = -P.mean(axis=0)

    # Hidden features
    H = np.maximum(0.0, P + b1)  # (N, 128)

    # Ridge regression for final layer
    # Augment with bias by centering H and Y
    H_mean = H.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Hc = H - H_mean
    Yc = Y - Y_mean

    lam_fc2 = 1e-2
    HtH = Hc.T @ Hc
    HtH_reg = HtH + lam_fc2 * np.eye(128)
    HtY = Hc.T @ Yc
    W2 = np.linalg.solve(HtH_reg, HtY)  # (128, 10)
    b2 = Y_mean - H_mean @ W2

    # Write back as float32
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.T.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    logits = H @ W2 + b2
    pred = np.argmax(logits, axis=1)
    acc = float((pred == y).mean())
    diagnostics = {
        "train_accuracy": acc,
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc1_bias_norm": float(np.linalg.norm(w["fc1.bias"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "hidden_mean": float(H.mean()),
        "hidden_std": float(H.std()),
        "hidden_zero_fraction": float((H == 0).mean()),
    }
    return {"weights": w, "diagnostics": diagnostics}
