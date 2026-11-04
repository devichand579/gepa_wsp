import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                  # (2000,)
    N = X.shape[0]
    C = 10

    # Current weights
    W1_old = w["fc1.weight"].copy()   # (128, 784)
    b1_old = w["fc1.bias"].copy()     # (128,)
    W2_old = w["fc2.weight"].copy()   # (10, 128)

    # One-hot labels
    Y = np.eye(C, dtype=np.float32)[y]  # (N, 10)

    # ---------- Ridge directions on raw pixels ----------
    X_mean = X.mean(axis=0)
    Xc = X - X_mean
    Y_mean = Y.mean(axis=0)
    Yc = Y - Y_mean

    alpha_ridge = 1.0
    XtX = Xc.T @ Xc
    XtY = Xc.T @ Yc
    W_ridge = np.linalg.solve(XtX + alpha_ridge * np.eye(X.shape[1]), XtY)  # (784, 10)

    # ---------- LDA directions ----------
    m = X.mean(axis=0)
    S_b = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    S_w = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for c in range(C):
        Xc_c = X[y == c]
        n_c = Xc_c.shape[0]
        m_c = Xc_c.mean(axis=0)
        diff = m_c - m
        S_b += n_c * np.outer(diff, diff)
        Xc_c_centered = Xc_c - m_c
        S_w += Xc_c_centered.T @ Xc_c_centered

    reg_lda = 1.0
    S_w_reg = S_w + reg_lda * np.eye(X.shape[1])
    L = np.linalg.cholesky(S_w_reg)
    L_inv = np.linalg.inv(L)
    M = L_inv @ S_b @ L_inv.T
    vals, vecs = np.linalg.eigh(M)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx[:9]]
    W_lda = L_inv.T @ vecs  # (784, 9)
    # Normalize LDA directions
    for i in range(9):
        nrm = np.linalg.norm(W_lda[:, i])
        if nrm > 1e-8:
            W_lda[:, i] /= nrm

    # ---------- Build new hidden features ----------
    # Replace 38 least important units by fc2 weight norm
    importance = np.linalg.norm(W2_old, axis=0)  # (128,)
    replace_idx = np.argsort(importance)[:38]

    new_weights = []
    new_biases = []

    # 10 ridge directions, each as positive/negative ReLU pair
    for c in range(C):
        d = W_ridge[:, c]
        b = -d @ X_mean
        new_weights.append(d)
        new_biases.append(b)
        new_weights.append(-d)
        new_biases.append(-b)

    # 9 LDA directions, each as positive/negative ReLU pair
    for i in range(9):
        d = W_lda[:, i]
        b = -d @ m
        new_weights.append(d)
        new_biases.append(b)
        new_weights.append(-d)
        new_biases.append(-b)

    new_weights = np.array(new_weights, dtype=np.float32)  # (38, 784)
    new_biases = np.array(new_biases, dtype=np.float32)    # (38,)

    W1_new = W1_old.copy()
    b1_new = b1_old.copy()
    for i, idx in enumerate(replace_idx):
        W1_new[idx] = new_weights[i]
        b1_new[idx] = new_biases[i]

    # ---------- Compute new hidden features on full training pool ----------
    H = X @ W1_new.T + b1_new  # (N, 128)
    H = np.maximum(H, 0.0)

    # Standardize hidden features
    mean_H = H.mean(axis=0)
    std_H = H.std(axis=0)
    std_H[std_H < 1e-6] = 1.0
    H_std = (H - mean_H) / std_H

    # ---------- Closed-form ridge fit for fc2 ----------
    alpha2 = 1.0
    HtH = H_std.T @ H_std
    HtY = H_std.T @ Yc
    W2_std = np.linalg.solve(HtH + alpha2 * np.eye(H.shape[1]), HtY)  # (128, 10)
    W2_std = W2_std.T  # (10, 128)

    # Convert back to raw hidden features
    W2_raw = W2_std / std_H          # (10, 128)
    b2_raw = Y_mean - W2_raw @ mean_H  # (10,)

    # ---------- Write back ----------
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2_raw.astype(np.float32)
    w["fc2.bias"] = b2_raw.astype(np.float32)

    # Safety clip
    for k in w:
        if isinstance(w[k], np.ndarray):
            w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diag = {
        "replaced_units": int(len(replace_idx)),
        "ridge_dirs": 10,
        "lda_dirs": 9,
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diag}
