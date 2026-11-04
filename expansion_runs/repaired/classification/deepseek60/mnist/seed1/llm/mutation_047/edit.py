import numpy as np

def edit(w, f):
    # Extract full training pool
    H = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = H.shape
    C = 10

    # Standardise features (handle dead units)
    mu = H.mean(axis=0)
    sd = H.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Hs = (H - mu) / sd

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment with bias column
    Hs_aug = np.concatenate([Hs, np.ones((N, 1))], axis=1)   # (N, D+1)

    # Ridge regression: W = (X^T X + λ I)^{-1} X^T Y
    lam = 1e-3
    A = Hs_aug.T @ Hs_aug + lam * np.eye(D + 1)
    B = Hs_aug.T @ Y
    W = np.linalg.solve(A, B)                     # (D+1, C)

    # Convert back to original feature space
    # logits = W^T [ (h-mu)/sd ; 1 ] = sum_j W_j (h_j - mu_j)/sd_j + W_bias
    new_fc2_weight = (W[:D, :] / sd[:, None]).T   # (C, D)
    new_fc2_bias = W[D, :] - np.sum(W[:D, :] * (mu / sd)[:, None], axis=0)

    w["fc2.weight"] = new_fc2_weight.astype(np.float32)
    w["fc2.bias"] = new_fc2_bias.astype(np.float32)

    # Diagnostics: training accuracy on the feedback pool
    logits = Hs_aug @ W
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == y))

    diag = {
        "train_acc": acc,
        "lambda": lam,
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
    }
    return {"weights": w, "diagnostics": diag}
