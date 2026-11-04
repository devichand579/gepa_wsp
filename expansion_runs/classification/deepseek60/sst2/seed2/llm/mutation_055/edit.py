import numpy as np

def edit(w, f):
    # Read current features and labels for the entire feedback pool.
    X = np.asarray(f["features"], dtype=np.float64)   # [4000, 32]
    y = np.asarray(f["y"], dtype=np.int64)            # [4000]
    N, D = X.shape

    # Standardize features using pool statistics. This is a closed-form
    # affine transform and can be absorbed into the final head.
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd

    # Build class-conditional means and pooled covariance.
    Z0 = Z[y == 0]
    Z1 = Z[y == 1]
    m0 = Z0.mean(axis=0)
    m1 = Z1.mean(axis=0)

    # Pooled within-class covariance, with a small ridge for stability.
    C0 = np.cov(Z0, rowvar=False, bias=True)
    C1 = np.cov(Z1, rowvar=False, bias=True)
    C = 0.5 * (C0 + C1)
    C += 1e-3 * np.eye(D)

    # Fisher direction: C^{-1} (m1 - m0).
    diff = m1 - m0
    try:
        direction = np.linalg.solve(C, diff)
    except np.linalg.LinAlgError:
        direction = np.linalg.pinv(C) @ diff

    # Normalize direction for numerical stability.
    nrm = np.linalg.norm(direction)
    if nrm > 1e-12:
        direction = direction / nrm

    # Choose intercept so that the decision boundary is at the midpoint
    # of the projected class means.
    proj0 = Z0 @ direction
    proj1 = Z1 @ direction
    c0 = proj0.mean()
    c1 = proj1.mean()
    intercept = -0.5 * (c0 + c1)

    # We need logits = head @ pooled_features + bias.
    # Since Z = (X - mu)/sd, the effective linear map on X is:
    #   direction @ Z = (direction / sd) @ X - direction @ (mu/sd)
    # So set head.weight[1] - head.weight[0] = direction / sd,
    # and adjust bias accordingly.
    w_dir = direction / sd.ravel()
    w_bias = intercept - np.dot(w_dir, mu.ravel())

    # Use symmetric two-row head: row0 = -0.5 * w_dir, row1 = +0.5 * w_dir.
    # Then logit1 - logit0 = w_dir @ X + w_bias.
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[0] = -0.5 * w_dir
    head_w[1] =  0.5 * w_dir

    head_b = np.zeros(2, dtype=np.float32)
    head_b[0] = -0.5 * w_bias
    head_b[1] =  0.5 * w_bias

    # Write back only the head. Keep all other tensors unchanged so that
    # the cached features remain exactly valid.
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"]   = head_b.astype(np.float32)

    # Ensure final norm is identity-LN, which is already true, but set explicitly.
    w["norm.weight"] = np.ones(D, dtype=np.float32)
    w["norm.bias"]   = np.zeros(D, dtype=np.float32)

    # Diagnostics: in-sample accuracy and simple statistics.
    logits = X @ head_w.T + head_b
    pred = np.argmax(logits, axis=1)
    acc = float((pred == y).mean())

    diagnostics = {
        "in_sample_accuracy": acc,
        "direction_norm": float(np.linalg.norm(w_dir)),
        "intercept": float(w_bias),
        "feature_mean_absmax": float(np.max(np.abs(mu))),
        "feature_std_min": float(np.min(sd)),
        "feature_std_max": float(np.max(sd)),
    }

    return {"weights": w, "diagnostics": diagnostics}
