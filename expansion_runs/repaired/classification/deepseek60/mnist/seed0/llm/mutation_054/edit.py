import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    H = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Standardize features (avoid division by zero)
    mu = H.mean(axis=0)
    sigma = H.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    Z = (H - mu) / sigma

    # Center Z and Y
    Zc = Z - Z.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    # Ridge regression: W_z = (Zc^T Zc + lambda I)^{-1} Zc^T Yc
    # Use a small lambda for numerical stability and mild regularization.
    # Since Z is standardized, Zc^T Zc has diagonal ~ N-1 = 1999.
    # lambda = 1.0 gives a very small amount of shrinkage.
    lambda_reg = 1.0
    A = Zc.T @ Zc + lambda_reg * np.eye(D)
    W_z = np.linalg.solve(A, Zc.T @ Yc)           # (D, C)
    b_z = Y.mean(axis=0) - Z.mean(axis=0) @ W_z   # (C,)

    # Convert back to original feature space:
    # logits = Z @ W_z + b_z = H @ (W_z / sigma) - (mu/sigma) @ W_z + b_z
    fc2_weight = (W_z.T / sigma[None, :]).astype(np.float32)   # (C, D)
    fc2_bias = (b_z - (mu / sigma) @ W_z).astype(np.float32)   # (C,)

    # Write back into the weight dictionary
    w["fc2.weight"][...] = fc2_weight
    w["fc2.bias"][...] = fc2_bias

    # Optional diagnostics
    diagnostics = {
        "method": "standardized_ridge_fc2",
        "lambda_reg": float(lambda_reg),
        "fc2_weight_norm": float(np.linalg.norm(fc2_weight)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_bias)),
        "train_accuracy": float(np.mean(np.argmax(H @ fc2_weight.T + fc2_bias, axis=1) == y)),
    }
    return {"weights": w, "diagnostics": diagnostics}
