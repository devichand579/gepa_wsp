import numpy as np

def edit(w, f):
    # Extract current weights
    fc1_w = w["fc1.weight"].copy()
    fc1_b = w["fc1.bias"].copy()
    fc2_w = w["fc2.weight"].copy()
    fc2_b = w["fc2.bias"].copy()

    # Full feedback pool
    x = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)

    # Re-center fc1 biases so that mean pre-activation over the pool is zero
    z = x @ fc1_w.T + fc1_b          # (2000, 128)
    mean_z = z.mean(axis=0)          # (128,)
    fc1_b_new = fc1_b - mean_z
    z_new = z - mean_z
    h = np.maximum(0.0, z_new)       # refreshed ReLU features (2000, 128)

    # Closed-form regularized LDA on the refreshed features
    C = 10
    mu = h.mean(axis=0)              # (128,)
    mu_c = np.zeros((C, 128), dtype=np.float64)
    for c in range(C):
        mu_c[c] = h[y == c].mean(axis=0)

    Sw = np.zeros((128, 128), dtype=np.float64)
    for c in range(C):
        diff = h[y == c] - mu_c[c]
        Sw += diff.T @ diff

    # Try two small ridge values and pick the one with better training accuracy
    best_acc = -1.0
    best_W = None
    best_b = None
    for alpha in (0.01, 0.001):
        Sw_reg = Sw + alpha * np.eye(128, dtype=np.float64)
        W_lda = np.linalg.solve(Sw_reg, mu_c.T).T   # (10, 128)
        b_lda = -W_lda @ mu                         # (10,)
        logits = h @ W_lda.T + b_lda
        pred = np.argmax(logits, axis=1)
        acc = (pred == y).mean()
        if acc > best_acc:
            best_acc = acc
            best_W = W_lda
            best_b = b_lda

    # Update weights
    w["fc1.bias"] = fc1_b_new.astype(np.float32)
    w["fc2.weight"] = best_W.astype(np.float32)
    w["fc2.bias"] = best_b.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "train_accuracy": float(best_acc),
        "fc1_bias_mean": float(fc1_b_new.mean()),
        "fc1_bias_std": float(fc1_b_new.std()),
        "fc2_weight_norm": float(np.linalg.norm(best_W)),
        "fc2_bias_norm": float(np.linalg.norm(best_b)),
        "alpha": 0.01 if best_acc == (h @ (np.linalg.solve(Sw + 0.01*np.eye(128), mu_c.T).T).T + (-np.linalg.solve(Sw + 0.01*np.eye(128), mu_c.T).T @ mu)).argmax(axis=1).__eq__(y).mean() else 0.001
    }
    return {"weights": w, "diagnostics": diagnostics}
