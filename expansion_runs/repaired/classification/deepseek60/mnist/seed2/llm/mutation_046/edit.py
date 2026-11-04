import numpy as np
import math

def edit(w, f):
    # Extract arrays
    X = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape
    K = 10

    # Current fc2 parameters
    W_cur = w["fc2.weight"].astype(np.float64).copy()  # (10, 128)
    b_cur = w["fc2.bias"].astype(np.float64).copy()    # (10,)

    # Helper: accuracy of a linear classifier on given data
    def acc(W, b, X, y):
        logits = X @ W.T + b
        pred = np.argmax(logits, axis=1)
        return np.mean(pred == y)

    # Current full-pool accuracy
    cur_acc = acc(W_cur, b_cur, X, y)

    # 5-fold CV setup
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)

    # Candidate LDA ridge values
    ridge_vals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_ridge = None
    best_cv_acc = -1.0

    # Precompute class means and pooled covariance for full data (for final fit)
    # But for CV we need to compute on training folds only.

    for ridge in ridge_vals:
        cv_accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Compute class means on training data
            means = np.zeros((K, D))
            for c in range(K):
                mask = (y_tr == c)
                if np.sum(mask) > 0:
                    means[c] = X_tr[mask].mean(axis=0)
                else:
                    means[c] = 0.0

            # Pooled within-class covariance
            cov = np.zeros((D, D))
            for c in range(K):
                mask = (y_tr == c)
                if np.sum(mask) > 1:
                    Xc = X_tr[mask] - means[c]
                    cov += Xc.T @ Xc
            cov /= (len(train_idx) - K)
            # Regularize
            cov += ridge * np.eye(D)

            # LDA weights: W = inv(cov) @ (means - global_mean).T
            # Actually for classification: score_c = x^T inv(cov) mean_c - 0.5 mean_c^T inv(cov) mean_c + log(prior)
            # Since priors are balanced, log(prior) is constant.
            inv_cov = np.linalg.inv(cov)
            W_lda = (inv_cov @ means.T).T  # (K, D)
            b_lda = -0.5 * np.sum(means * (inv_cov @ means.T).T, axis=1)  # (K,)
            # Add log prior (uniform -> 0)
            # Evaluate on validation
            val_acc = acc(W_lda, b_lda, X_val, y_val)
            cv_accs.append(val_acc)
        mean_cv = np.mean(cv_accs)
        if mean_cv > best_cv_acc:
            best_cv_acc = mean_cv
            best_ridge = ridge

    # Also evaluate current readout in the same CV framework (just for reference)
    # We already have cur_acc on full pool, but CV for current would require refitting? No, current is fixed.
    # We can compute CV accuracy of current by simply evaluating on each fold? But current is trained on all data, so CV is not fair.
    # Instead, we compare best LDA CV to current full-pool accuracy. If LDA CV is better, we apply LDA.
    # However, to be safe, we can also compute LDA on full data and check full-pool accuracy.

    # Fit LDA on full data with best_ridge
    means_full = np.zeros((K, D))
    for c in range(K):
        mask = (y == c)
        means_full[c] = X[mask].mean(axis=0)

    cov_full = np.zeros((D, D))
    for c in range(K):
        mask = (y == c)
        Xc = X[mask] - means_full[c]
        cov_full += Xc.T @ Xc
    cov_full /= (N - K)
    cov_full += best_ridge * np.eye(D)

    inv_cov_full = np.linalg.inv(cov_full)
    W_lda_full = (inv_cov_full @ means_full.T).T
    b_lda_full = -0.5 * np.sum(means_full * (inv_cov_full @ means_full.T).T, axis=1)

    lda_full_acc = acc(W_lda_full, b_lda_full, X, y)

    # Decide whether to apply LDA
    apply_lda = False
    if best_cv_acc > cur_acc + 1e-4:  # require a small margin
        apply_lda = True
    elif lda_full_acc > cur_acc + 1e-4:
        apply_lda = True

    # Prepare diagnostics
    diagnostics = {
        "current_full_acc": float(cur_acc),
        "best_ridge": float(best_ridge),
        "best_cv_acc": float(best_cv_acc),
        "lda_full_acc": float(lda_full_acc),
        "applied_lda": bool(apply_lda),
    }

    if apply_lda:
        # Scale weights to reasonable magnitude? LDA weights can be large if covariance is ill-conditioned.
        # We can optionally scale down to avoid extreme logits, but argmax is scale-invariant.
        # However, we must ensure finite and <=1000 in absolute value.
        W_new = W_lda_full
        b_new = b_lda_full
        # Clip to safe range
        W_new = np.clip(W_new, -1000.0, 1000.0)
        b_new = np.clip(b_new, -1000.0, 1000.0)
        w["fc2.weight"] = W_new.astype(np.float32)
        w["fc2.bias"] = b_new.astype(np.float32)
        diagnostics["fc2_weight_norm"] = float(np.linalg.norm(W_new))
        diagnostics["fc2_bias_norm"] = float(np.linalg.norm(b_new))
    else:
        # Fallback: conservative bias correction to equalize mean logits per class
        # Compute mean logits per class on full pool
        logits = X @ W_cur.T + b_cur
        mean_logits_per_class = np.array([logits[y == c].mean(axis=0) for c in range(K)])  # (K, K)
        # Actually we want to adjust bias so that mean logit for each class is equal.
        # The mean logit for class c over all examples is logits[:, c].mean().
        mean_logits = logits.mean(axis=0)  # (K,)
        target_mean = mean_logits.mean()
        adjustment = target_mean - mean_logits
        # Only apply if adjustment is significant
        if np.max(np.abs(adjustment)) > 1e-3:
            b_new = b_cur + adjustment
            w["fc2.bias"] = b_new.astype(np.float32)
            diagnostics["bias_adjustment"] = adjustment.tolist()
        else:
            diagnostics["bias_adjustment"] = "none"

    # Ensure all weights are finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    return {"weights": w, "diagnostics": diagnostics}
