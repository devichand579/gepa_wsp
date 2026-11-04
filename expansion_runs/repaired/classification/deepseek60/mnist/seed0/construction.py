import numpy as np

def edit(w, f):
    # Extract the entire training pool features and labels
    H = f["features"].astype(np.float64)   # (2000, 128)
    y = f["y"].astype(np.int64)            # (2000,)
    N, D = H.shape                         # D = 128
    C = 10                                 # number of classes

    # One-hot encode labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment features with a bias column
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float64)])  # (N, D+1)

    # Ridge regression closed-form solution
    # We solve (H_aug^T H_aug + λ I) W = H_aug^T Y
    lam = 1e-3
    A = H_aug.T @ H_aug + lam * np.eye(D + 1, dtype=np.float64)
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B)          # (D+1, C)

    # Extract weights and bias for fc2
    fc2_weight = W_aug[:D, :].T.astype(np.float32)   # (C, D)
    fc2_bias   = W_aug[D, :].astype(np.float32)      # (C,)

    # Update the weights dictionary
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"]   = fc2_bias

    # Optional diagnostics: compute training accuracy on the provided pool
    logits = H @ fc2_weight.T + fc2_bias
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)

    diagnostics = {
        "train_accuracy": float(acc),
        "lambda": lam,
        "fc2_weight_mean": float(np.mean(fc2_weight)),
        "fc2_weight_std": float(np.std(fc2_weight)),
        "fc2_bias_mean": float(np.mean(fc2_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
