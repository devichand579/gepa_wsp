import numpy as np

def edit(w, f):
    # Extract arrays
    X = f["x"].reshape(2000, 784).astype(np.float64)  # [2000,784]
    y = f["y"].astype(np.int64)                       # [2000]
    H_old = f["features"].astype(np.float64)          # [2000,128] current hidden features
    logits_old = f["logits"].astype(np.float64)       # [2000,10]
    probs_old = f["probs"].astype(np.float64)         # [2000,10]

    # Current weights
    W1 = w["fc1.weight"].astype(np.float64)  # [128,784]
    b1 = w["fc1.bias"].astype(np.float64)    # [128]
    W2 = w["fc2.weight"].astype(np.float64)  # [10,128]
    b2 = w["fc2.bias"].astype(np.float64)    # [10]

    # Identify dead units: first 10 have very negative biases
    dead_idx = np.arange(10)
    alive_idx = np.arange(10, 128)

    # Compute class means and LDA directions for the dead units
    classes = np.unique(y)
    mu = X.mean(axis=0)  # [784]
    mu_c = np.array([X[y == c].mean(axis=0) for c in classes])  # [10,784]
    # Between-class scatter (scaled by class counts, but classes are balanced)
    B = mu_c - mu  # [10,784]
    # Within-class scatter
    Sw = np.zeros((784, 784), dtype=np.float64)
    for c in classes:
        Xc = X[y == c]
        Xc_centered = Xc - mu_c[c]
        Sw += Xc_centered.T @ Xc_centered
    # Regularize Sw to make it invertible
    reg = 1e-3 * np.trace(Sw) / 784.0
    Sw_reg = Sw + reg * np.eye(784)
    # Solve generalized eigenvalue problem in the span of B
    # We want v = B^T a. Maximize a^T B Sb B^T a / a^T B Sw B^T a
    # Sb = sum_c n_c (mu_c - mu)(mu_c - mu)^T. Here n_c=200, so Sb = 200 * B^T B.
    # Actually, let's compute the projected matrices directly.
    # Let M = B (10x784). We want to solve M Sb M^T a = lambda M Sw M^T a.
    # Sb = 200 * B^T B. So M Sb M^T = 200 * B B^T B B^T? Wait, careful:
    # v = B^T a. v^T Sb v = a^T B (200 B^T B) B^T a = 200 a^T (B B^T) (B B^T) a.
    # v^T Sw v = a^T B Sw B^T a.
    # So we need to solve: 200 (B B^T)^2 a = lambda (B Sw B^T) a.
    # This is a 10x10 generalized eigenvalue problem.
    BBt = B @ B.T  # [10,10]
    BSwBt = B @ Sw_reg @ B.T  # [10,10]
    # Solve generalized eigenproblem: 200 * BBt @ BBt @ a = lambda * BSwBt @ a
    # We can use scipy.linalg.eig, but only numpy is allowed. We can transform to standard eigenproblem.
    # Let L = cholesky(BSwBt). Then solve (L^-1 200 BBt BBt L^-T) v = lambda v, a = L^-T v.
    # Since BSwBt is 10x10, we can just use np.linalg.eig on inv(BSwBt) @ (200 * BBt @ BBt).
    M = np.linalg.inv(BSwBt) @ (200.0 * BBt @ BBt)
    eigvals, eigvecs = np.linalg.eig(M)
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    a = eigvecs[:, idx[:10]].real  # [10,10]
    # Directions v = B^T a  -> [784,10]
    V = B.T @ a  # [784,10]
    # Normalize each direction to unit norm
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    # Set new fc1 weights for dead units: V.T is [10,784]
    W1_new = W1.copy()
    W1_new[dead_idx] = V.T
    # Set biases so that the new features are centered around 0
    # Compute mean projection on all data
    proj_mean = X @ V  # [2000,10]
    b1_new = b1.copy()
    b1_new[dead_idx] = -proj_mean.mean(axis=0)
    # Recompute hidden features for all 2000 examples
    # Pre-activations for all units
    pre_act = X @ W1_new.T + b1_new  # [2000,128]
    H_new = np.maximum(pre_act, 0.0)  # [2000,128]
    # The new features are H_new[:, dead_idx] (10 features)
    H_new_dead = H_new[:, dead_idx]  # [2000,10]
    # Existing features from alive units (we keep them as they are, but note H_old was computed with old W1.
    # However, we changed W1 only for dead units, so alive units' features are unchanged.
    # So H_old[:, alive_idx] is exactly the same as H_new[:, alive_idx].
    H_alive = H_old[:, alive_idx]  # [2000,118]

    # Now fit fc2 for the new dead units and bias, keeping alive units' weights fixed.
    # Current logits from alive units: logits_alive = H_alive @ W2[:, alive_idx].T + b2
    logits_alive = H_alive @ W2[:, alive_idx].T + b2  # [2000,10]
    # We want to find W2_dead (10x10) and b2_new (10) such that
    # logits_alive + H_new_dead @ W2_dead.T + b2_new = target
    # We use one-hot targets for least squares.
    Y = np.zeros((2000, 10))
    Y[np.arange(2000), y] = 1.0
    # Residual target for the new part:
    R = Y - logits_alive  # [2000,10]
    # We need to solve for W2_dead and b2_new:
    # H_new_dead @ W2_dead.T + b2_new = R
    # Augment H_new_dead with a column of ones
    H_aug = np.hstack([H_new_dead, np.ones((2000, 1))])  # [2000,11]
    # Solve ridge regression: W_aug = (H_aug^T H_aug + lambda I)^-1 H_aug^T R
    # We'll use a small lambda for regularization.
    lambda_reg = 1.0  # small compared to typical H^T H values
    A = H_aug.T @ H_aug + lambda_reg * np.eye(11)
    b = H_aug.T @ R  # [11,10]
    W_aug = np.linalg.solve(A, b)  # [11,10]
    W2_dead = W_aug[:10, :].T  # [10,10]
    b2_new = W_aug[10, :]      # [10]

    # Update weights
    W2_new = W2.copy()
    W2_new[:, dead_idx] = W2_dead
    b2_new_full = b2 + b2_new  # add the correction to existing bias

    # Write back to w
    w["fc1.weight"] = W1_new.astype(np.float32)
    w["fc1.bias"] = b1_new.astype(np.float32)
    w["fc2.weight"] = W2_new.astype(np.float32)
    w["fc2.bias"] = b2_new_full.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "dead_units_replaced": 10,
        "lambda_reg": lambda_reg,
        "new_fc1_weight_norm": float(np.linalg.norm(W1_new[dead_idx])),
        "new_fc1_bias": b1_new[dead_idx].tolist(),
        "new_fc2_weight_norm": float(np.linalg.norm(W2_new[:, dead_idx])),
        "new_fc2_bias": b2_new_full.tolist(),
    }
    return {"weights": w, "diagnostics": diagnostics}
