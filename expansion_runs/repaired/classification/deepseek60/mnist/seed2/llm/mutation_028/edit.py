import numpy as np

def edit(w, f):
    # Read current weights
    fc1_w = w["fc1.weight"].astype(np.float64)
    fc1_b = w["fc1.bias"].astype(np.float64)
    fc2_w = w["fc2.weight"].astype(np.float64)
    fc2_b = w["fc2.bias"].astype(np.float64)

    X = f["features"].astype(np.float64)   # [2000,128] relu features
    y = f["y"].astype(np.int64)            # [2000]
    N, D = X.shape
    C = 10

    # 1) Feature conditioning: compute per-feature mean/std on the pool
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    # We cannot directly normalize after ReLU, but we can scale the pre-activation
    # by a positive factor and shift the bias so that the post-ReLU feature is
    # approximately scaled.  For a positive scale s, relu(s*z + t) = s*relu(z + t/s)
    # if t is chosen so that the active set is preserved.  We use a mild scale
    # to improve conditioning without destroying the learned representation.
    scale = 1.0 / np.maximum(sd, 1e-3)      # per-unit scale
    scale = np.clip(scale, 0.1, 10.0)       # avoid extreme rescaling
    # Keep the mean roughly the same by shifting the bias accordingly.
    # new_bias = old_bias * scale + (mu * (1 - scale))  (approximate)
    new_fc1_w = fc1_w * scale[:, None]
    new_fc1_b = fc1_b * scale + mu * (1.0 - scale)

    # 2) Recompute hidden features with the new fc1 on the whole pool.
    #    We have the raw images in f["x"].
    Ximg = f["x"].astype(np.float64).reshape(N, -1)   # [2000,784]
    Z = Ximg @ new_fc1_w.T + new_fc1_b[None, :]       # [2000,128]
    H = np.maximum(Z, 0.0)                            # new features

    # 3) Closed-form ridge refit of fc2 on the new features.
    #    Use one-hot targets and solve (H^T H + alpha I) W^T = H^T Y.
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Center features to help the bias term
    H_mean = H.mean(axis=0)
    Hc = H - H_mean[None, :]

    # Choose alpha by a simple closed-form heuristic: alpha = trace(Hc^T Hc)/N * 1e-3
    # (a small ridge to stabilize the normal equations)
    alpha = 1e-3 * (Hc * Hc).sum() / N
    alpha = max(alpha, 1e-6)

    # Solve for weights (without bias first)
    A = Hc.T @ Hc + alpha * np.eye(D)
    B = Hc.T @ Y
    W_new = np.linalg.solve(A, B).T          # [C, D]
    b_new = (Y.mean(axis=0) - W_new @ H_mean)  # [C]

    # 4) Optionally blend with the original fc2 to avoid catastrophic changes.
    #    A small blend keeps some of the original decision boundaries.
    blend = 0.0   # use pure refit; if unstable, could set to 0.1
    fc2_w_new = (1.0 - blend) * W_new + blend * fc2_w
    fc2_b_new = (1.0 - blend) * b_new + blend * fc2_b

    # 5) Write back as float32, clip to safe range.
    w["fc1.weight"] = np.clip(new_fc1_w, -1000, 1000).astype(np.float32)
    w["fc1.bias"]   = np.clip(new_fc1_b, -1000, 1000).astype(np.float32)
    w["fc2.weight"] = np.clip(fc2_w_new, -1000, 1000).astype(np.float32)
    w["fc2.bias"]   = np.clip(fc2_b_new, -1000, 1000).astype(np.float32)

    # Diagnostics
    diag = {
        "scale_mean": float(scale.mean()),
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "alpha": float(alpha),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_min": float(w["fc2.weight"].min()),
        "fc2_weight_max": float(w["fc2.weight"].max()),
        "fc2_bias_min": float(w["fc2.bias"].min()),
        "fc2_bias_max": float(w["fc2.bias"].max()),
    }
    return {"weights": w, "diagnostics": diag}
