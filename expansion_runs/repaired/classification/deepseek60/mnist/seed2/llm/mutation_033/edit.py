import numpy as np

def edit(w, f):
    # Extract full feedback pool
    x = f["x"].reshape(-1, 784).astype(np.float32)   # (2000, 784)
    y = f["y"].astype(np.int64)                      # (2000,)

    # Current fc1 parameters
    W1 = w["fc1.weight"].astype(np.float32)          # (128, 784)
    b1 = w["fc1.bias"].astype(np.float32)            # (128,)

    # Compute pre-activations for the full pool
    z = x @ W1.T + b1                                # (2000, 128)

    # Per-unit standardization
    mu = z.mean(axis=0)                              # (128,)
    sigma = z.std(axis=0) + 1e-5                     # (128,)

    W1_new = W1 / sigma[:, None]
    b1_new = (b1 - mu) / sigma

    # New hidden features
    H = np.maximum(0.0, x @ W1_new.T + b1_new)       # (2000, 128)

    # One-hot targets
    Y = np.zeros((2000, 10), dtype=np.float32)
    Y[np.arange(2000), y] = 1.0

    # Center features and targets
    H_mean = H.mean(axis=0)                          # (128,)
    Y_mean = Y.mean(axis=0)                          # (10,)
    H_c = H - H_mean
    Y_c = Y - Y_mean

    # Closed-form ridge regression for fc2
    alpha = 1e-3
    A = H_c.T @ H_c + alpha * np.eye(128, dtype=np.float32)
    B = H_c.T @ Y_c
    W2 = np.linalg.solve(A, B).T                     # (10, 128)
    b2 = Y_mean - H_mean @ W2.T                      # (10,)

    # Compute new logits and rescale to match original logit std
    Z = H @ W2.T + b2                                # (2000, 10)
    std_new = Z.std()
    std_cur = f["logits"].std()
    scale = std_cur / (std_new + 1e-8)
    W2 *= scale
    b2 *= scale

    # Clip to safe range (should already be well within)
    W1_new = np.clip(W1_new, -1000.0, 1000.0)
    b1_new = np.clip(b1_new, -1000.0, 1000.0)
    W2 = np.clip(W2, -1000.0, 1000.0)
    b2 = np.clip(b2, -1000.0, 1000.0)

    # Write back
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    diag = {
        "scale": float(scale),
        "std_new": float(std_new),
        "std_cur": float(std_cur),
        "fc1_weight_norm": float(np.linalg.norm(W1_new)),
        "fc1_bias_norm": float(np.linalg.norm(b1_new)),
        "fc2_weight_norm": float(np.linalg.norm(W2)),
        "fc2_bias_norm": float(np.linalg.norm(b2)),
        "fc1_weight_min": float(W1_new.min()),
        "fc1_weight_max": float(W1_new.max()),
        "fc1_bias_min": float(b1_new.min()),
        "fc1_bias_max": float(b1_new.max()),
        "fc2_weight_min": float(W2.min()),
        "fc2_weight_max": float(W2.max()),
        "fc2_bias_min": float(b2.min()),
        "fc2_bias_max": float(b2.max()),
    }
    return {"weights": w, "diagnostics": diag}
