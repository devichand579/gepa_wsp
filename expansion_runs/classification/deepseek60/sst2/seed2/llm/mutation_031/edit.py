import numpy as np

def edit(w, f):
    # Extract data
    x = f["x"]  # [4000, 64] int64
    y = f["y"]  # [4000] int64
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build length-normalized BOW matrix: X[i,t] = count(i,t) / length(i)
    # Use bincount per example
    X = np.zeros((N, V), dtype=np.float64)
    lengths = np.zeros(N, dtype=np.float64)
    for i in range(N):
        tokens = x[i]
        # exclude padding (0)
        mask = tokens != 0
        if mask.sum() == 0:
            lengths[i] = 1.0
            continue
        lengths[i] = mask.sum()
        counts = np.bincount(tokens[mask], minlength=V)
        X[i] = counts / lengths[i]

    y_signed = 2.0 * y - 1.0  # {-1, 1}

    # Helper: 5-fold CV accuracy for a given method
    def cv_accuracy(method, param):
        idx = np.arange(N)
        np.random.seed(42)
        np.random.shuffle(idx)
        folds = np.array_split(idx, 5)
        accs = []
        for k in range(5):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            X_tr = X[train_idx]
            y_tr = y_signed[train_idx]
            X_te = X[test_idx]
            y_te = y_signed[test_idx]

            if method == "ridge":
                A = X_tr.T @ X_tr + param * np.eye(V)
                b_vec = X_tr.T @ y_tr
                w_lin = np.linalg.solve(A, b_vec)
                b_lin = np.mean(y_tr - X_tr @ w_lin)
            elif method == "lda":
                mu0 = X_tr[y_tr < 0].mean(axis=0)
                mu1 = X_tr[y_tr > 0].mean(axis=0)
                # pooled covariance
                X0 = X_tr[y_tr < 0]
                X1 = X_tr[y_tr > 0]
                n0, n1 = len(X0), len(X1)
                cov = (X0.T @ X0 + X1.T @ X1) / (n0 + n1) - np.outer((n0*mu0 + n1*mu1)/(n0+n1), (n0*mu0 + n1*mu1)/(n0+n1))
                A = cov + param * np.eye(V)
                w_lin = np.linalg.solve(A, mu1 - mu0)
                b_lin = -0.5 * (mu1 + mu0) @ w_lin
            elif method == "nb":
                # Naive Bayes with Laplace smoothing alpha=param
                alpha = param
                count_pos = X_tr[y_tr > 0].sum(axis=0)
                count_neg = X_tr[y_tr < 0].sum(axis=0)
                total_pos = count_pos.sum()
                total_neg = count_neg.sum()
                w_lin = np.log((count_pos + alpha) / (total_pos + alpha * V)) - np.log((count_neg + alpha) / (total_neg + alpha * V))
                b_lin = np.log(0.5) - np.log(0.5)  # equal priors
            else:
                raise ValueError(method)

            scores = X_te @ w_lin + b_lin
            preds = (scores > 0).astype(int)
            acc = np.mean(preds == (y_te > 0).astype(int))
            accs.append(acc)
        return np.mean(accs)

    # Evaluate several methods and parameters
    best_acc = -1.0
    best_method = None
    best_param = None
    best_w = None
    best_b = None

    # Ridge lambdas
    for lam in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        acc = cv_accuracy("ridge", lam)
        if acc > best_acc:
            best_acc = acc
            best_method = "ridge"
            best_param = lam

    # LDA lambdas
    for lam in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        acc = cv_accuracy("lda", lam)
        if acc > best_acc:
            best_acc = acc
            best_method = "lda"
            best_param = lam

    # Naive Bayes alphas
    for alpha in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]:
        acc = cv_accuracy("nb", alpha)
        if acc > best_acc:
            best_acc = acc
            best_method = "nb"
            best_param = alpha

    # Fit best method on full data
    if best_method == "ridge":
        A = X.T @ X + best_param * np.eye(V)
        b_vec = X.T @ y_signed
        w_lin = np.linalg.solve(A, b_vec)
        b_lin = np.mean(y_signed - X @ w_lin)
    elif best_method == "lda":
        mu0 = X[y_signed < 0].mean(axis=0)
        mu1 = X[y_signed > 0].mean(axis=0)
        X0 = X[y_signed < 0]
        X1 = X[y_signed > 0]
        n0, n1 = len(X0), len(X1)
        cov = (X0.T @ X0 + X1.T @ X1) / (n0 + n1) - np.outer((n0*mu0 + n1*mu1)/(n0+n1), (n0*mu0 + n1*mu1)/(n0+n1))
        A = cov + best_param * np.eye(V)
        w_lin = np.linalg.solve(A, mu1 - mu0)
        b_lin = -0.5 * (mu1 + mu0) @ w_lin
    elif best_method == "nb":
        alpha = best_param
        count_pos = X[y_signed > 0].sum(axis=0)
        count_neg = X[y_signed < 0].sum(axis=0)
        total_pos = count_pos.sum()
        total_neg = count_neg.sum()
        w_lin = np.log((count_pos + alpha) / (total_pos + alpha * V)) - np.log((count_neg + alpha) / (total_neg + alpha * V))
        b_lin = 0.0

    # Normalize w_lin to avoid overflow and ensure |a_t| <= 4
    max_abs = np.max(np.abs(w_lin))
    if max_abs < 1e-8:
        max_abs = 1.0
    R = max_abs / 4.0  # so |w_t / R| <= 4
    a = w_lin / R  # a_t in [-4, 4]

    # Construct embed.weight
    embed = np.zeros((V, 32), dtype=np.float32)
    for t in range(V):
        at = a[t]
        # first two coordinates: at, -at
        embed[t, 0] = at
        embed[t, 1] = -at
        # remaining 30 coordinates: pairs of +/- c
        c = np.sqrt((32.0 - 2.0 * at * at) / 30.0)
        for i in range(2, 32):
            embed[t, i] = c if (i % 2 == 0) else -c
    # Pad token 0: set to zero (doesn't matter, excluded from pooling)
    embed[0, :] = 0.0

    w["embed.weight"] = embed

    # Zero out positional embeddings
    w["pos.weight"] = np.zeros_like(w["pos.weight"])

    # Zero out all transformer block weights and biases
    for block in ["block0", "block1"]:
        for name in ["qkv", "proj", "ff1", "ff2"]:
            w[f"{block}.{name}.weight"] = np.zeros_like(w[f"{block}.{name}.weight"])
            w[f"{block}.{name}.bias"] = np.zeros_like(w[f"{block}.{name}.bias"])
        # Set LayerNorm weights to 1, biases to 0 (they don't matter but safe)
        w[f"{block}.norm1.weight"] = np.ones_like(w[f"{block}.norm1.weight"])
        w[f"{block}.norm1.bias"] = np.zeros_like(w[f"{block}.norm1.bias"])
        w[f"{block}.norm2.weight"] = np.ones_like(w[f"{block}.norm2.weight"])
        w[f"{block}.norm2.bias"] = np.zeros_like(w[f"{block}.norm2.bias"])

    # Final norm: identity (weight=1, bias=0)
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    # Set head to compute R * pooled[0] + b_lin
    head_w = np.zeros_like(w["head.weight"])  # [2, 32]
    head_w[0, 0] = -0.5 * R
    head_w[1, 0] = 0.5 * R
    w["head.weight"] = head_w

    head_b = np.zeros_like(w["head.bias"])  # [2]
    head_b[0] = -0.5 * b_lin
    head_b[1] = 0.5 * b_lin
    w["head.bias"] = head_b

    # Diagnostics
    diagnostics = {
        "best_method": best_method,
        "best_param": float(best_param),
        "cv_accuracy": float(best_acc),
        "R": float(R),
        "b_lin": float(b_lin),
        "w_lin_absmax": float(max_abs),
        "w_lin_std": float(np.std(w_lin)),
    }

    return {"weights": w, "diagnostics": diagnostics}
