import numpy as np

def edit(w, f):
    # Extract arrays
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # [2000, 784]
    y = f["y"].astype(np.int64)
    H = f["features"].astype(np.float64)  # [2000, 128] current hidden features
    N = X.shape[0]
    C = 10
    D = X.shape[1]

    # Current fc1 and fc2
    fc1_w = w["fc1.weight"].astype(np.float64).copy()  # [128, 784]
    fc1_b = w["fc1.bias"].astype(np.float64).copy()    # [128]
    fc2_w = w["fc2.weight"].astype(np.float64).copy()  # [10, 128]
    fc2_b = w["fc2.bias"].astype(np.float64).copy()    # [10]

    # Identify dead units: those with very negative bias (first 10 in this network)
    dead_idx = np.where(fc1_b < -10.0)[0]
    if len(dead_idx) == 0:
        # No dead units, just refit fc2 with ridge towards current
        pass
    else:
        # Compute LDA directions on raw pixels for the dead units
        # Class means
        mu = np.zeros((C, D))
        for c in range(C):
            mu[c] = X[y == c].mean(axis=0)
        mu_all = X.mean(axis=0)
        # Pooled within-class covariance
        Sigma = np.zeros((D, D))
        for c in range(C):
            Xc = X[y == c] - mu[c]
            Sigma += Xc.T @ Xc
        Sigma /= (N - C)
        # Regularize covariance for stability
        Sigma += np.eye(D) * 1e-3
        # LDA directions: Sigma^{-1} (mu_c - mu_all)
        # We need 10 directions. Solve Sigma W = (mu - mu_all)^T
        # Use pseudo-inverse or solve
        try:
            W_lda = np.linalg.solve(Sigma, (mu - mu_all).T).T  # [10, 784]
        except np.linalg.LinAlgError:
            W_lda = np.linalg.lstsq(Sigma, (mu - mu_all).T, rcond=None)[0].T
        # Normalize each direction to have unit norm (optional)
        norms = np.linalg.norm(W_lda, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        W_lda = W_lda / norms

        # Replace dead units with LDA directions
        n_dead = len(dead_idx)
        for i, idx in enumerate(dead_idx):
            if i < W_lda.shape[0]:
                fc1_w[idx] = W_lda[i]
                fc1_b[idx] = 0.0  # Set bias to 0 so unit is active
            else:
                # If more dead units than classes, fill with zeros or random? Just set to zero
                fc1_w[idx] = 0.0
                fc1_b[idx] = 0.0

    # Recompute hidden features with new fc1
    # h = relu(X @ fc1_w.T + fc1_b)
    H_new = np.maximum(0.0, X @ fc1_w.T + fc1_b)  # [2000, 128]

    # Refit fc2 using ridge regression towards current fc2
    # Augment with bias
    H_aug = np.hstack([H_new, np.ones((N, 1))])  # [2000, 129]
    # One-hot targets
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    # Current fc2 augmented
    W0 = np.hstack([fc2_w, fc2_b.reshape(-1, 1)])  # [10, 129]

    # Ridge solution with prior: W = (Y^T H_aug + lambda W0) (H_aug^T H_aug + lambda I)^{-1}
    # Choose lambda based on scale of H_aug^T H_aug
    # Compute trace of H_aug^T H_aug
    HtH = H_aug.T @ H_aug
    trace_HtH = np.trace(HtH)
    # Use lambda = 1e-3 * trace_HtH / 129 to keep regularization moderate
    lam = 1e-3 * trace_HtH / 129.0
    if lam < 1e-6:
        lam = 1e-6

    # Solve for W_aug
    # W_aug = (Y^T H_aug + lam * W0) @ inv(HtH + lam * I)
    # Use solve for stability
    A = HtH + lam * np.eye(129)
    B = Y.T @ H_aug + lam * W0  # [10, 129]
    try:
        W_aug = np.linalg.solve(A, B.T).T  # [10, 129]
    except np.linalg.LinAlgError:
        W_aug = B @ np.linalg.pinv(A)

    fc2_w_new = W_aug[:, :128]
    fc2_b_new = W_aug[:, 128]

    # Compute new logits on full pool
    logits_new = H_new @ fc2_w_new.T + fc2_b_new  # [2000, 10]

    # Scale logits to match current logit standard deviation
    # Current logits std from f["logits"]
    current_logits = f["logits"].astype(np.float64)
    current_std = np.std(current_logits)
    new_std = np.std(logits_new)
    if new_std > 1e-8:
        scale = current_std / new_std
    else:
        scale = 1.0
    # Apply scale to weights and bias
    fc2_w_new *= scale
    fc2_b_new *= scale

    # Update weights
    w["fc1.weight"] = fc1_w.astype(np.float32)
    w["fc1.bias"] = fc1_b.astype(np.float32)
    w["fc2.weight"] = fc2_w_new.astype(np.float32)
    w["fc2.bias"] = fc2_b_new.astype(np.float32)

    # Diagnostics
    # Compute training accuracy with new weights
    logits_final = H_new @ fc2_w_new.T + fc2_b_new
    preds = np.argmax(logits_final, axis=1)
    acc = np.mean(preds == y)
    # Compute cross-entropy
    probs = np.exp(logits_final - logits_final.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    diagnostics = {
        "dead_units_replaced": int(len(dead_idx)),
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "fc1_weight_norm": float(np.linalg.norm(fc1_w)),
        "fc1_bias_norm": float(np.linalg.norm(fc1_b)),
        "fc2_weight_norm": float(np.linalg.norm(fc2_w_new)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_b_new)),
        "logit_scale": float(scale),
        "lambda": float(lam)
    }

    return {"weights": w, "diagnostics": diagnostics}
