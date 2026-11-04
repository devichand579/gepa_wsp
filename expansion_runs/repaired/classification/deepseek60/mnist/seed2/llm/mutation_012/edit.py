import numpy as np
import math

def edit(w, f):
    # Extract feedback data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    y = f["y"].astype(np.int64)                                 # (2000,)
    N = X.shape[0]
    C = 10

    # Current fc1 parameters
    W1 = w["fc1.weight"].astype(np.float32)  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float32)    # (128,)

    # Compute pre-activations and mean to re-center biases
    Z = X @ W1.T + b1                        # (2000, 128)
    mean_z = Z.mean(axis=0)                  # (128,)
    b1_new = -mean_z

    # Helper: compute ReLU features for a given bias
    def get_features(bias):
        Z = X @ W1.T + bias
        return np.maximum(0, Z)

    # LDA fitting with cross-validation for alpha and standardization
    def lda_cv(H, y, alpha, standardize):
        # 5-fold CV
        rng = np.random.RandomState(42)
        indices = rng.permutation(N)
        folds = np.array_split(indices, 5)
        accs = []
        for i in range(5):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_train = H[train_idx]
            y_train = y[train_idx]
            H_test = H[test_idx]
            y_test = y[test_idx]

            if standardize:
                m = H_train.mean(axis=0)
                s = H_train.std(axis=0) + 1e-6
                H_train_std = (H_train - m) / s
                H_test_std = (H_test - m) / s
            else:
                H_train_std = H_train
                H_test_std = H_test

            D = H_train_std.shape[1]
            Mu = np.zeros((C, D), dtype=np.float64)
            for c in range(C):
                Mu[c] = H_train_std[y_train == c].mean(axis=0)

            S = np.zeros((D, D), dtype=np.float64)
            for c in range(C):
                Hc = H_train_std[y_train == c]
                if len(Hc) > 1:
                    Hc_centered = Hc - Mu[c]
                    S += Hc_centered.T @ Hc_centered
            S /= (len(y_train) - C)
            S_reg = S + alpha * np.eye(D)

            W_lda = np.linalg.solve(S_reg, Mu.T).T  # (C, D)
            priors = np.array([np.mean(y_train == c) for c in range(C)])
            b_lda = -0.5 * np.sum(Mu * W_lda, axis=1) + np.log(priors)

            scores = H_test_std @ W_lda.T + b_lda
            preds = np.argmax(scores, axis=1)
            accs.append(np.mean(preds == y_test))
        cv_acc = np.mean(accs)

        # Fit on full data
        if standardize:
            m_full = H.mean(axis=0)
            s_full = H.std(axis=0) + 1e-6
            H_std = (H - m_full) / s_full
        else:
            H_std = H
            m_full = None
            s_full = None

        Mu_full = np.zeros((C, D), dtype=np.float64)
        for c in range(C):
            Mu_full[c] = H_std[y == c].mean(axis=0)

        S_full = np.zeros((D, D), dtype=np.float64)
        for c in range(C):
            Hc = H_std[y == c]
            if len(Hc) > 1:
                Hc_centered = Hc - Mu_full[c]
                S_full += Hc_centered.T @ Hc_centered
        S_full /= (N - C)
        S_reg_full = S_full + alpha * np.eye(D)

        W_lda_full = np.linalg.solve(S_reg_full, Mu_full.T).T  # (C, D)
        priors_full = np.array([np.mean(y == c) for c in range(C)])
        b_lda_full = -0.5 * np.sum(Mu_full * W_lda_full, axis=1) + np.log(priors_full)

        if standardize:
            W_orig = W_lda_full / s_full[None, :]
            b_orig = b_lda_full - (W_lda_full / s_full[None, :]) @ m_full
        else:
            W_orig = W_lda_full
            b_orig = b_lda_full

        return W_orig, b_orig, cv_acc

    # Search over bias options, standardization, and alpha
    bias_options = [b1, b1_new]
    best_cv = -1.0
    best_W = None
    best_b = None
    best_bias = None
    best_std = None
    best_alpha = None

    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    for bias in bias_options:
        H = get_features(bias)
        for standardize in [False, True]:
            for alpha in alphas:
                W_lda, b_lda, cv_acc = lda_cv(H, y, alpha, standardize)
                if cv_acc > best_cv:
                    best_cv = cv_acc
                    best_W = W_lda
                    best_b = b_lda
                    best_bias = bias
                    best_std = standardize
                    best_alpha = alpha

    # Apply best parameters
    w["fc1.bias"] = best_bias.astype(np.float32)
    w["fc2.weight"] = best_W.astype(np.float32)
    w["fc2.bias"] = best_b.astype(np.float32)

    # Clip to safe range
    for k in w:
        if isinstance(w[k], np.ndarray):
            w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "best_cv_accuracy": float(best_cv),
        "best_alpha": float(best_alpha),
        "standardize": bool(best_std),
        "bias_changed": bool(np.any(best_bias != b1)),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc1_bias_mean": float(np.mean(w["fc1.bias"])),
        "fc1_bias_std": float(np.std(w["fc1.bias"])),
    }

    return {"weights": w, "diagnostics": diagnostics}
