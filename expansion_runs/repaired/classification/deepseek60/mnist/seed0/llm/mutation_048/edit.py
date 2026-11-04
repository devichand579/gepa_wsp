import numpy as np
import math

def edit(w, f):
    # Extract full feedback pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    F_old = f["features"].astype(np.float32)                    # (2000, 128)

    # Compute class means and global mean
    classes = np.arange(10)
    mean_c = np.zeros((10, X.shape[1]), dtype=np.float32)
    for c in classes:
        mean_c[c] = X[y == c].mean(axis=0)
    mean_all = X.mean(axis=0)

    # Build new weights and biases for the first 10 dead units
    W_new = (mean_c - mean_all).astype(np.float32)              # (10, 784)
    b_new = (-0.5 * (np.sum(mean_c**2, axis=1) - np.sum(mean_all**2))).astype(np.float32)  # (10,)

    # Compute new activations for these 10 units on the full pool
    A_new = np.maximum(0.0, X @ W_new.T + b_new)                # (2000, 10)

    # Combine with the old features (first 10 columns were dead, so we replace them)
    F_new = F_old.copy()
    F_new[:, :10] = A_new

    # Fit fc2 via ridge regression on the new features
    # Augment with a bias column
    F_aug = np.hstack([F_new, np.ones((F_new.shape[0], 1), dtype=np.float32)])  # (2000, 129)
    Y_onehot = np.eye(10, dtype=np.float32)[y]                                  # (2000, 10)

    # Ridge solution: W_aug = (F_aug^T F_aug + lambda I)^-1 F_aug^T Y
    # Use a small lambda for numerical stability and mild regularization
    lam = 1e-3
    A = F_aug.T @ F_aug + lam * np.eye(F_aug.shape[1], dtype=np.float32)
    B = F_aug.T @ Y_onehot
    W_aug = np.linalg.solve(A, B).T                                             # (10, 129)

    W2_new = W_aug[:, :128].astype(np.float32)
    b2_new = W_aug[:, 128].astype(np.float32)

    # Write back to w
    w["fc1.weight"][:10] = W_new
    w["fc1.bias"][:10] = b_new
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new

    # Optional: clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    # Diagnostics
    diag = {
        "method": "revive_dead_units_ridge",
        "lambda": lam,
        "fc1_bias_first10": w["fc1.bias"][:10].tolist(),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diag}
