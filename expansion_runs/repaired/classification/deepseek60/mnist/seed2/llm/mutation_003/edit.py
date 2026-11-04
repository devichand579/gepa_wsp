import numpy as np

def edit(w, f):
    # Read current parameters
    fc1_w = w["fc1.weight"]
    fc1_b = w["fc1.bias"]
    fc2_w = w["fc2.weight"]
    fc2_b = w["fc2.bias"]

    # Full feedback pool
    X = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    n, d = X.shape
    C = 10

    # One-hot targets
    Y = np.zeros((n, C), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Center features and targets for a stable ridge fit with intercept
    X_mean = X.mean(axis=0, keepdims=True)
    Y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # Ridge regression: solve (Xc^T Xc + alpha I) W^T = Xc^T Yc
    # Use a modest alpha to avoid overfitting the 2000-example pool.
    alpha = 1e-2
    A = Xc.T @ Xc + alpha * np.eye(d, dtype=np.float64)
    B = Xc.T @ Yc
    Wt = np.linalg.solve(A, B)                    # [128, 10]
    W_new = Wt.T                                  # [10, 128]
    b_new = (Y_mean - X_mean @ W_new.T).ravel()   # [10]

    # Optional: blend with the current fc2 to reduce variance.
    # A 50/50 blend often helps when the refit is close in training accuracy.
    blend = 0.5
    W_final = blend * W_new + (1.0 - blend) * fc2_w.astype(np.float64)
    b_final = blend * b_new + (1.0 - blend) * fc2_b.astype(np.float64)

    # Ensure finite float32 and bounded
    W_final = np.clip(W_final, -1000.0, 1000.0).astype(np.float32)
    b_final = np.clip(b_final, -1000.0, 1000.0).astype(np.float32)

    # Write back
    w["fc2.weight"] = W_final
    w["fc2.bias"] = b_final

    # Diagnostics
    # Compute training accuracy of the new readout on the full pool
    logits_new = X @ W_final.T + b_final
    pred_new = np.argmax(logits_new, axis=1)
    acc_new = float((pred_new == y).mean())

    # Current accuracy for comparison
    logits_old = X @ fc2_w.T + fc2_b
    pred_old = np.argmax(logits_old, axis=1)
    acc_old = float((pred_old == y).mean())

    diagnostics = {
        "acc_old_full": acc_old,
        "acc_new_full": acc_new,
        "alpha": alpha,
        "blend": blend,
        "fc2_weight_norm": float(np.linalg.norm(W_final)),
        "fc2_bias_norm": float(np.linalg.norm(b_final)),
        "fc2_weight_mean": float(W_final.mean()),
        "fc2_weight_std": float(W_final.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
