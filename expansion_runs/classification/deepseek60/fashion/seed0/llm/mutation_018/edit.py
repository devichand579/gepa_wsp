import numpy as np

def edit(w, f):
    # Extract features and labels from feedback
    F = f["features"]          # (2000, 1568)
    y = f["y"]                 # (2000,)
    N, D = F.shape
    C = 10

    # Standardize features (scale-invariant regularization)
    mu = F.mean(axis=0)
    sigma = F.std(axis=0) + 1e-6
    F_std = (F - mu) / sigma

    # Augment with bias column
    F_aug = np.hstack([F_std, np.ones((N, 1))])  # (2000, 1569)

    # One-hot targets
    Y = np.eye(C)[y]  # (2000, 10)

    # Small grid for ridge lambda via 5-fold CV (closed-form, no iterative training)
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    best_lambda = 1.0
    best_acc = -1.0

    # 5-fold CV
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    folds = np.array_split(indices, 5)

    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            F_tr = F_aug[tr_idx]
            Y_tr = Y[tr_idx]
            F_val = F_aug[val_idx]
            Y_val = Y[val_idx]
            # Ridge solution: (F^T F + lam I) W = F^T Y
            A = F_tr.T @ F_tr + lam * np.eye(D + 1)
            B = F_tr.T @ Y_tr
            W_aug = np.linalg.solve(A, B)
            # Predict
            logits_val = F_val @ W_aug
            pred = np.argmax(logits_val, axis=1)
            true = np.argmax(Y_val, axis=1)
            acc = np.mean(pred == true)
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Fit on full feedback with best lambda
    lam = best_lambda
    A = F_aug.T @ F_aug + lam * np.eye(D + 1)
    B = F_aug.T @ Y
    W_aug = np.linalg.solve(A, B)

    # Split into weights and bias in standardized space
    W_std = W_aug[:D, :].T   # (10, 1568)
    b_std = W_aug[D, :]      # (10,)

    # Convert back to original feature space
    W = W_std / sigma[None, :]                     # (10, 1568)
    b = b_std - W_std @ (mu / sigma)               # (10,)

    # Compute logits on full feedback to find margin
    logits = F @ W.T + b  # (2000, 10)
    true_logits = logits[np.arange(N), y]
    # Mask out correct class to find max incorrect
    logits_masked = logits.copy()
    logits_masked[np.arange(N), y] = -np.inf
    max_incorrect = logits_masked.max(axis=1)
    margins = true_logits - max_incorrect
    min_margin = margins.min()

    # Center bias (does not affect softmax)
    b_centered = b - b.mean()

    # Determine scaling factor k to sharpen logits while respecting |weight| <= 1000
    max_abs_W = np.max(np.abs(W))
    max_abs_b = np.max(np.abs(b_centered))
    k_max = 1000.0 / max(max_abs_W, max_abs_b, 1e-9)
    target_margin = 30.0
    if min_margin > 1e-9:
        k_target = target_margin / min_margin
    else:
        k_target = k_max
    k = min(k_target, k_max)
    # Ensure k is at least 1.0 (if not, use 1.0)
    k = max(k, 1.0)

    # Apply scaled weights and bias
    new_head_weight = (k * W).astype(np.float32)
    new_head_bias = (k * b_centered).astype(np.float32)

    # Clip to safe range just in case
    new_head_weight = np.clip(new_head_weight, -1000.0, 1000.0)
    new_head_bias = np.clip(new_head_bias, -1000.0, 1000.0)

    # Update weights
    w["head.weight"] = new_head_weight
    w["head.bias"] = new_head_bias

    # Diagnostics
    diagnostics = {
        "best_lambda": float(lam),
        "cv_accuracy": float(best_acc),
        "min_margin_before_scale": float(min_margin),
        "scale_k": float(k),
        "max_abs_head_weight": float(np.max(np.abs(new_head_weight))),
        "max_abs_head_bias": float(np.max(np.abs(new_head_bias))),
        "feedback_accuracy": float(np.mean(np.argmax(logits, axis=1) == y)),
        "feedback_ce": float(-np.mean(np.log(np.exp(logits[np.arange(N), y]) / np.exp(logits).sum(axis=1) + 1e-12))),
    }

    return {"weights": w, "diagnostics": diagnostics}
