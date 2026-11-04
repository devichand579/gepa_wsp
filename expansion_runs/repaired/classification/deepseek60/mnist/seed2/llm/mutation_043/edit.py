import numpy as np

def edit(w, f):
    # Extract feedback data
    F = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    N, D = F.shape
    C = 10
    Y = np.eye(C, dtype=np.float64)[y]            # (2000, 10)

    # Current fc2 parameters (PyTorch linear: logits = F @ W.T + b)
    W0 = w["fc2.weight"].astype(np.float64)       # (10, 128)
    b0 = w["fc2.bias"].astype(np.float64)         # (10,)

    # Current logits and accuracy on the full pool
    logits0 = F @ W0.T + b0
    acc0 = np.mean(np.argmax(logits0, axis=1) == y)

    # Helper: fit ridge with prior toward (W0, b0)
    def fit_ridge(F_tr, Y_tr, lam_w, lam_b, W_prior, b_prior):
        N_tr, D_tr = F_tr.shape
        F_aug = np.hstack([F_tr, np.ones((N_tr, 1))])
        A0 = np.vstack([W_prior.T, b_prior.reshape(1, -1)])
        Lambda = np.ones(D_tr + 1) * lam_w
        Lambda[-1] = lam_b
        M = F_aug.T @ F_aug + np.diag(Lambda)
        rhs = F_aug.T @ Y_tr + np.diag(Lambda) @ A0
        A = np.linalg.solve(M, rhs)
        W = A[:D_tr, :].T
        b = A[D_tr, :]
        return W, b

    # Helper: bias-only correction for fixed W
    def fit_bias_only(F_tr, Y_tr, W_fixed):
        # Optimal b = mean(Y - F @ W.T) over training samples
        resid = Y_tr - F_tr @ W_fixed.T
        b = resid.mean(axis=0)
        return W_fixed, b

    # 5-fold CV
    rng = np.random.RandomState(42)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)

    # Candidate methods: (name, lambda_w, lambda_b)
    # lambda_w large -> stay close to W0; lambda_b large -> stay close to b0
    candidates = [
        ("bias_only", None, None),
        ("ridge_prior_1e2_1e2", 1e2, 1e2),
        ("ridge_prior_1e1_1e1", 1e1, 1e1),
        ("ridge_prior_1e0_1e0", 1e0, 1e0),
        ("ridge_prior_1e-1_1e-1", 1e-1, 1e-1),
        ("ridge_prior_1e-2_1e-2", 1e-2, 1e-2),
        ("ridge_noprior_1e-2", 1e-2, 0.0),
        ("ridge_noprior_1e-1", 1e-1, 0.0),
        ("ridge_noprior_1e0", 1e0, 0.0),
    ]

    cv_results = {}
    for name, lam_w, lam_b in candidates:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            F_tr, Y_tr = F[tr_idx], Y[tr_idx]
            F_val, y_val = F[val_idx], y[val_idx]

            if name == "bias_only":
                W_new, b_new = fit_bias_only(F_tr, Y_tr, W0)
            else:
                # If no prior, use zero prior
                if "noprior" in name:
                    W_prior = np.zeros_like(W0)
                    b_prior = np.zeros_like(b0)
                else:
                    W_prior = W0
                    b_prior = b0
                W_new, b_new = fit_ridge(F_tr, Y_tr, lam_w, lam_b, W_prior, b_prior)

            logits_val = F_val @ W_new.T + b_new
            acc = np.mean(np.argmax(logits_val, axis=1) == y_val)
            accs.append(acc)
        cv_results[name] = np.mean(accs)

    # Also evaluate current fc2 on the same CV folds (it is fixed)
    accs_cur = []
    for i in range(5):
        val_idx = folds[i]
        F_val, y_val = F[val_idx], y[val_idx]
        logits_val = F_val @ W0.T + b0
        accs_cur.append(np.mean(np.argmax(logits_val, axis=1) == y_val))
    cv_results["current"] = np.mean(accs_cur)

    # Pick best method
    best_name = max(cv_results, key=cv_results.get)
    best_cv_acc = cv_results[best_name]

    # Refit best method on all 2000 examples
    if best_name == "current":
        W_new, b_new = W0.copy(), b0.copy()
    elif best_name == "bias_only":
        W_new, b_new = fit_bias_only(F, Y, W0)
    else:
        # find the candidate entry
        for name, lam_w, lam_b in candidates:
            if name == best_name:
                if "noprior" in name:
                    W_prior = np.zeros_like(W0)
                    b_prior = np.zeros_like(b0)
                else:
                    W_prior = W0
                    b_prior = b0
                W_new, b_new = fit_ridge(F, Y, lam_w, lam_b, W_prior, b_prior)
                break

    # Scale logits to match current logit standard deviation for reasonable CE
    logits_new = F @ W_new.T + b_new
    std_new = np.std(logits_new)
    std_cur = np.std(logits0)
    if std_new > 1e-8:
        scale = std_cur / std_new
    else:
        scale = 1.0
    W_new *= scale
    b_new *= scale

    # Ensure finite and within [-1000, 1000]
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)

    # Update weights
    w["fc2.weight"] = W_new
    w["fc2.bias"] = b_new

    diagnostics = {
        "cv_results": {k: float(v) for k, v in cv_results.items()},
        "best_method": best_name,
        "best_cv_acc": float(best_cv_acc),
        "current_full_acc": float(acc0),
        "scale": float(scale),
        "fc2_weight_norm": float(np.linalg.norm(W_new)),
        "fc2_bias_norm": float(np.linalg.norm(b_new)),
        "fc2_weight_min": float(W_new.min()),
        "fc2_weight_max": float(W_new.max()),
        "fc2_bias_min": float(b_new.min()),
        "fc2_bias_max": float(b_new.max()),
    }
    return {"weights": w, "diagnostics": diagnostics}
