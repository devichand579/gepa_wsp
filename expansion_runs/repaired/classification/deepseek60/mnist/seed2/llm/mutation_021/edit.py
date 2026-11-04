import numpy as np

def edit(w, f):
    # ---- extract feedback data ----
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    N = X.shape[0]
    C = 10

    # ---- current fc1 parameters ----
    W1 = w["fc1.weight"].astype(np.float64)  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float64)    # (128,)

    # ---- pre-activations without bias ----
    A = X @ W1.T  # (2000, 128)

    # ---- standardize each hidden unit over the feedback pool ----
    mean_j = A.mean(axis=0)          # (128,)
    std_j  = A.std(axis=0) + 1e-8    # (128,)

    W1_new = W1 / std_j[:, None]     # (128, 784)
    b1_new = -mean_j / std_j         # (128,)

    # ---- refreshed features ----
    H = np.maximum(0.0, A / std_j + b1_new)  # (2000, 128)

    # ---- closed-form LDA readout ----
    # class means
    mu = np.zeros((C, 128), dtype=np.float64)
    for c in range(C):
        mu[c] = H[y == c].mean(axis=0)

    # within-class scatter
    Sw = np.zeros((128, 128), dtype=np.float64)
    for c in range(C):
        Hc = H[y == c]
        diff = Hc - mu[c]
        Sw += diff.T @ diff

    # regularize Sw (scale by average variance)
    alpha = 1e-2
    Sw_reg = Sw + alpha * np.trace(Sw) / 128.0 * np.eye(128)

    # LDA weights and biases
    W2 = np.linalg.solve(Sw_reg, mu.T).T          # (10, 128)
    b2 = -0.5 * np.sum(mu * W2, axis=1)           # (10,)

    # ---- write back as float32 ----
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"]   = b1_new.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"]   = b2.astype(np.float32)

    # ---- diagnostics ----
    logits = H @ W2.T + b2
    pred = np.argmax(logits, axis=1)
    acc = float((pred == y).mean())
    diag = {
        "fc1_bias_mean": float(b1_new.mean()),
        "fc1_bias_std":  float(b1_new.std()),
        "fc1_bias_min":  float(b1_new.min()),
        "fc1_bias_max":  float(b1_new.max()),
        "fc2_weight_norm": float(np.linalg.norm(W2)),
        "fc2_bias_norm":   float(np.linalg.norm(b2)),
        "train_accuracy":  acc,
    }
    return {"weights": w, "diagnostics": diag}
