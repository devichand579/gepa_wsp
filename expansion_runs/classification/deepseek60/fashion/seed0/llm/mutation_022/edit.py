import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"]          # shape (2000, 1568)
    y = f["y"]                 # shape (2000,)
    N, D = X.shape
    C = 10

    # One-hot encode labels
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Add bias column to features
    Xb = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)

    # Ridge regression: W = (X^T X + lambda I)^-1 X^T Y
    # Use a moderate lambda for regularization
    lam = 100.0
    A = Xb.T @ Xb + lam * np.eye(D + 1, dtype=np.float64)
    B = Xb.T @ Y
    W = np.linalg.solve(A, B)  # (1569, 10)

    # Extract head weights and bias
    head_weight = W[:-1, :].T   # (10, 1568)
    head_bias = W[-1, :]        # (10,)

    # Update the head parameters
    w["head.weight"][:] = head_weight.astype(np.float32)
    w["head.bias"][:] = head_bias.astype(np.float32)

    # Optional: scale logits to reduce cross-entropy without changing argmax
    # Compute current logits on feedback to find a safe scale factor
    logits = X @ head_weight.T + head_bias  # (2000, 10)
    # Center bias to allow larger scaling without exceeding bounds
    bias_center = head_bias.mean()
    head_bias_centered = head_bias - bias_center
    w["head.bias"][:] = head_bias_centered.astype(np.float32)

    # Recompute logits with centered bias
    logits_centered = X @ head_weight.T + head_bias_centered
    # Find maximum absolute logit
    max_abs_logit = np.max(np.abs(logits_centered))
    # We can scale up to keep max_abs <= 1000
    scale = min(1000.0 / (max_abs_logit + 1e-9), 10.0)  # cap at 10 for safety
    if scale > 1.0:
        w["head.weight"][:] = (head_weight * scale).astype(np.float32)
        w["head.bias"][:] = (head_bias_centered * scale).astype(np.float32)

    # Diagnostics
    diagnostics = {
        "ridge_lambda": lam,
        "max_abs_head_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_head_bias": float(np.max(np.abs(w["head.bias"]))),
        "scale_factor": float(scale),
    }

    return {"weights": w, "diagnostics": diagnostics}
