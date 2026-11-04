import numpy as np
import math

def edit(w, f):
    try:
        H = f["features"].astype(np.float64)          # (2000, 128)
        y = f["y"].astype(np.int64)                   # (2000,)
        N, D = H.shape
        C = 10

        H_aug = np.hstack([H, np.ones((N, 1))])       # (2000, 129)
        Y = np.eye(C)[y]                              # (2000, 10)

        def ridge_fit(H_aug, Y, lam):
            A = H_aug.T @ H_aug
            reg = lam * np.eye(H_aug.shape[1])
            reg[-1, -1] = 0.0                         # do not regularize bias
            W = np.linalg.solve(A + reg, H_aug.T @ Y)
            return W

        def lda_fit(H, y, alpha):
            N, D = H.shape
            C = 10
            mu = np.zeros((C, D))
            for c in range(C):
                mu[c] = H[y == c].mean(axis=0)
            Sigma = np.zeros((D, D))
            for c in range(C):
                diff = H[y == c] - mu[c]
                Sigma += diff.T @ diff
            Sigma /= (N - C)
            avg_var = np.trace(Sigma) / D
            Sigma_shrunk = (1 - alpha) * Sigma + alpha * np.eye(D) * avg_var
            W = np.linalg.solve(Sigma_shrunk, mu.T).T
            b = -0.5 * np.sum(mu * W, axis=1) + np.log(np.bincount(y, minlength=C) / N)
            return W, b

        def cv_ridge(lam, n_folds=5):
            indices = np.arange(N)
            rng = np.random.RandomState(42)
            rng.shuffle(indices)
            folds = np.array_split(indices, n_folds)
            accs = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                H_train = H_aug[train_idx]
                Y_train = Y[train_idx]
                H_val = H_aug[val_idx]
                y_val = y[val_idx]
                W = ridge_fit(H_train, Y_train, lam)
                logits = H_val @ W
                pred = np.argmax(logits, axis=1)
                accs.append(np.mean(pred == y_val))
            return np.mean(accs)

        def cv_lda(alpha, n_folds=5):
            indices = np.arange(N)
            rng = np.random.RandomState(42)
            rng.shuffle(indices)
            folds = np.array_split(indices, n_folds)
            accs = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                H_train = H[train_idx]
                y_train = y[train_idx]
                H_val = H[val_idx]
                y_val = y[val_idx]
                W, b = lda_fit(H_train, y_train, alpha)
                logits = H_val @ W.T + b
                pred = np.argmax(logits, axis=1)
                accs.append(np.mean(pred == y_val))
            return np.mean(accs)

        lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        ridge_accs = [cv_ridge(lam) for lam in lambdas]
        best_lam = lambdas[int(np.argmax(ridge_accs))]
        best_ridge_acc = float(np.max(ridge_accs))

        alphas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95]
        lda_accs = [cv_lda(alpha) for alpha in alphas]
        best_alpha = alphas[int(np.argmax(lda_accs))]
        best_lda_acc = float(np.max(lda_accs))

        if best_ridge_acc >= best_lda_acc:
            W_full = ridge_fit(H_aug, Y, best_lam)
            fc2_weight = W_full[:D, :].T
            fc2_bias = W_full[D, :]
            method = "ridge"
            best_param = best_lam
            best_cv_acc = best_ridge_acc
        else:
            W_full, b_full = lda_fit(H, y, best_alpha)
            fc2_weight = W_full
            fc2_bias = b_full
            method = "lda"
            best_param = best_alpha
            best_cv_acc = best_lda_acc

        w["fc2.weight"] = fc2_weight.astype(np.float32)
        w["fc2.bias"] = fc2_bias.astype(np.float32)

        diagnostics = {
            "method": method,
            "best_param": float(best_param),
            "best_cv_acc": float(best_cv_acc),
            "ridge_accs": {str(l): float(a) for l, a in zip(lambdas, ridge_accs)},
            "lda_accs": {str(a): float(acc) for a, acc in zip(alphas, lda_accs)},
            "fc2_weight_mean": float(np.mean(fc2_weight)),
            "fc2_weight_std": float(np.std(fc2_weight)),
            "fc2_bias_mean": float(np.mean(fc2_bias)),
            "fc2_bias_std": float(np.std(fc2_bias)),
        }
        return {"weights": w, "diagnostics": diagnostics}
    except Exception as e:
        return {"weights": w, "diagnostics": {"error": str(e)}}
