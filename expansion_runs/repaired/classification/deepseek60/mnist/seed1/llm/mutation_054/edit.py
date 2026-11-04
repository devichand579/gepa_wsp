import numpy as np

def edit(w, f):
    # Extract full feedback pool
    H = f["features"]          # (2000, 128)
    y = f["y"]                 # (2000,)
    N, D = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Standardise features (column-wise)
    mu = H.mean(axis=0)
    sigma = H.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    Z = (H - mu) / sigma

    # Augment with bias column
    Z_aug = np.hstack([Z, np.ones((N, 1), dtype=np.float32)])

    # Current fc2 in the original space
    W_old = w["fc2.weight"].T.astype(np.float32)   # (128, 10)
    b_old = w["fc2.bias"].astype(np.float32)       # (10,)

    # Convert current classifier to standardised space:
    # logits = H @ W_old + b_old = Z @ W_z + b_z
    # Z = (H - mu) / sigma  =>  H = Z * sigma + mu
    # H @ W_old = (Z * sigma + mu) @ W_old = Z @ (sigma[:,None] * W_old) + mu @ W_old
    W_z_old = sigma[:, None] * W_old               # (128, 10)
    b_z_old = b_old + mu @ W_old                   # (10,)

    # Ridge regression with prior mean = current classifier
    lambda_reg = 1.0   # moderate shrinkage toward current weights
    A = Z_aug.T @ Z_aug + lambda_reg * np.eye(D + 1)
    # Right-hand side: Z_aug^T Y + lambda * prior_aug
    prior_aug = np.vstack([W_z_old, b_z_old[None, :]])  # (D+1, 10)
    B = Z_aug.T @ Y + lambda_reg * prior_aug
    W_aug = np.linalg.solve(A, B)                  # (D+1, 10)

    W_z = W_aug[:D, :]                             # (128, 10)
    b_z = W_aug[D, :]                              # (10,)

    # Convert back to original feature space
    W_new = W_z / sigma[:, None]                   # (128, 10)
    b_new = b_z - (W_z * mu[:, None] / sigma[:, None]).sum(axis=0)  # (10,)

    new_fc2_weight = W_new.T.astype(np.float32)    # (10, 128)
    new_fc2_bias = b_new.astype(np.float32)        # (10,)

    # Evaluate on the feedback pool
    logits_new = H @ new_fc2_weight.T + new_fc2_bias
    pred_new = np.argmax(logits_new, axis=1)
    acc_new = float(np.mean(pred_new == y))

    logits_old = f["logits"]
    pred_old = np.argmax(logits_old, axis=1)
    acc_old = float(np.mean(pred_old == y))

    # Only replace if the new classifier is at least as good on the pool
    if acc_new >= acc_old:
        w["fc2.weight"] = new_fc2_weight
        w["fc2.bias"] = new_fc2_bias
        used_new = True
    else:
        used_new = False

    diagnostics = {
        "acc_old": acc_old,
        "acc_new": acc_new,
        "used_new": used_new,
        "lambda_reg": lambda_reg,
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    return {"weights": w, "diagnostics": diagnostics}
