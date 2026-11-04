def edit(w, f):
    import numpy as np

    # Extract full feedback features and labels
    H = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    n, d = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((n, C), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Current fc2 parameters
    W_old = w["fc2.weight"].astype(np.float64)    # (10, 128)
    b_old = w["fc2.bias"].astype(np.float64)      # (10,)

    # Center features and targets to handle intercept cleanly
    H_mean = H.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Hc = H - H_mean
    Yc = Y - Y_mean

    # Closed-form ridge solution on the full feedback pool
    alpha = 0.01
    A = Hc.T @ Hc + alpha * np.eye(d)
    W_ridge = np.linalg.solve(A, Hc.T @ Yc).T     # (10, 128)
    b_ridge = Y_mean - W_ridge @ H_mean           # (10,)

    # Analytically choose a single blend coefficient between old and new logits
    Z_new = H @ W_ridge.T + b_ridge
    Z_old = f["logits"].astype(np.float64)
    D = Z_new - Z_old
    num = np.sum(D * (Y - Z_old))
    den = np.sum(D * D)
    beta = num / den if den > 1e-12 else 0.5
    beta = float(np.clip(beta, 0.0, 1.0))

    # Blend weights and biases
    W_new = beta * W_ridge + (1.0 - beta) * W_old
    b_new = beta * b_ridge + (1.0 - beta) * b_old

    # Write back as float32
    w["fc2.weight"] = W_new.astype(np.float32)
    w["fc2.bias"] = b_new.astype(np.float32)

    diagnostics = {
        "alpha": float(alpha),
        "beta": float(beta),
        "W_new_norm": float(np.linalg.norm(W_new)),
        "b_new_norm": float(np.linalg.norm(b_new)),
        "W_old_norm": float(np.linalg.norm(W_old)),
        "b_old_norm": float(np.linalg.norm(b_old)),
        "W_new_max_abs": float(np.max(np.abs(W_new))),
        "b_new_max_abs": float(np.max(np.abs(b_new))),
    }
    return {"weights": w, "diagnostics": diagnostics}
