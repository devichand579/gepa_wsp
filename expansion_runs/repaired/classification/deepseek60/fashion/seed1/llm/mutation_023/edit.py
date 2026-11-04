import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)          # (2000, 1568)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = X.shape
    C = 10

    # Add bias column
    Xb = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # (2000, 1569)

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Ridge regression closed-form solution
    lambda_reg = 1e-4
    A = Xb.T @ Xb + lambda_reg * np.eye(D + 1)
    B = Xb.T @ Y
    W = np.linalg.solve(A, B)                     # (1569, 10)

    head_weight = W[:-1, :].T                     # (10, 1568)
    head_bias = W[-1, :]                          # (10,)

    # Scale weights so that the maximum absolute value is 999 (<=1000)
    max_abs = max(np.max(np.abs(head_weight)), np.max(np.abs(head_bias)))
    if max_abs > 0:
        scale = 999.0 / max_abs
        head_weight *= scale
        head_bias *= scale
    else:
        scale = 1.0

    # Write back to the parameter dictionary
    w["head.weight"] = head_weight.astype(np.float32)
    w["head.bias"] = head_bias.astype(np.float32)

    diagnostics = {
        "lambda": lambda_reg,
        "scale": float(scale),
        "max_abs": float(max_abs * scale),
        "head_weight_mean": float(np.mean(head_weight)),
        "head_weight_std": float(np.std(head_weight)),
        "head_bias_mean": float(np.mean(head_bias)),
    }

    return {"weights": w, "diagnostics": diagnostics}
