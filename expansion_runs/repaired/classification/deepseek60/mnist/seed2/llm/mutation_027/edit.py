import numpy as np

def edit(w, f):
    # Load data
    X = f["x"].reshape(2000, 784).astype(np.float32)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    num_classes = 10

    # Helper: ridge regression
    def fit_ridge(H, y, alpha):
        H_aug = np.concatenate([H, np.ones((N, 1), dtype=np.float32)], axis=1)
        Y = np.zeros((N, num_classes), dtype=np.float32)
        Y[np.arange(N), y] = 1.0
        D = H_aug.shape[1]
        A = H_aug.T @ H_aug + alpha * np.eye(D, dtype=np.float32)
        B = H_aug.T @ Y
        W = np.linalg.solve(A, B)
        weight = W[:-1, :].T
        bias = W[-1, :]
        return weight, bias

    def evaluate(H, y, alpha, T=1.0):
        indices = np.arange(N)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)
        accs, ces = [], []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_train, y_train = H[train_idx], y[train_idx]
            H_val, y_val = H[val_idx], y[val_idx]
            weight, bias = fit_ridge(H_train, y_train, alpha)
            logits = H_val @ weight.T + bias
            logits = logits * T
            preds = np.argmax(logits, axis=1)
            acc = np.mean(preds == y_val)
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            ce = -np.mean(np.log(probs[np.arange(len(y_val)), y_val] + 1e-12))
            accs.append(acc)
            ces.append(ce)
        return np.mean(accs), np.mean(ces)

    # Candidate A: current fc1 features
    H_current = f["features"].astype(np.float32)

    # Candidate B: PCA + LDA features
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:128]  # (128, 784)
    Z = Xc @ V.T  # (2000, 128)

    mu_c = np.zeros((num_classes, 128), dtype=np.float32)
    for c in range(num_classes):
        mu_c[c] = Z[y == c].mean(axis=0)
    mu_all = Z.mean(axis=0)

    Sw = np.zeros((128, 128), dtype=np.float32)
    for c in range(num_classes):
        Zc = Z[y == c] - mu_c[c]
        Sw += Zc.T @ Zc
    Sb = np.zeros((128, 128), dtype=np.float32)
    for c in range(num_classes):
        diff = (mu_c[c] - mu_all).reshape(-1, 1)
        Sb += (y == c).sum() * (diff @ diff.T)

    Sw += 1e-3 * np.eye(128, dtype=np.float32)
    Sw_inv = np.linalg.inv(Sw)
    M = Sw_inv @ Sb
    eigvals, eigvecs = np.linalg.eig(M)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx].real
    W_lda = eigvecs[:, :9].T @ V  # (9, 784)
    W_rest = V[9:128]             # (119, 784)
    W_new = np.vstack([W_lda, W_rest])  # (128, 784)

    best_score = -np.inf
    best_params = None

    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    thresholds = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # Evaluate candidate A
    for alpha in alphas:
        acc, ce = evaluate(H_current, y, alpha, T=1.0)
        score = acc - 0.01 * ce
        if score > best_score:
            best_score = score
            best_params = ('current', alpha, None, 1.0)

    # Evaluate candidate B
    for th in thresholds:
        b_new = -mu @ W_new.T + th
        H_new = np.maximum(0, X @ W_new.T + b_new)
        for alpha in alphas:
            acc, ce = evaluate(H_new, y, alpha, T=1.0)
            score = acc - 0.01 * ce
            if score > best_score:
                best_score = score
                best_params = ('pca_lda', alpha, th, 1.0)

    # Refit on full data with best candidate
    if best_params[0] == 'current':
        H_best = H_current
        W_fc1 = w['fc1.weight'].copy()
        b_fc1 = w['fc1.bias'].copy()
        alpha = best_params[1]
    else:
        _, alpha, th, _ = best_params
        b_new = -mu @ W_new.T + th
        H_best = np.maximum(0, X @ W_new.T + b_new)
        W_fc1 = W_new.astype(np.float32)
        b_fc1 = b_new.astype(np.float32)

    weight, bias = fit_ridge(H_best, y, alpha)

    # Temperature scaling to minimize cross-entropy
    logits_full = H_best @ weight.T + bias
    best_T = 1.0
    best_ce = np.inf
    for T in np.linspace(0.1, 10.0, 100):
        logits_scaled = logits_full * T
        logits_max = logits_scaled.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_scaled - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        if ce < best_ce:
            best_ce = ce
            best_T = T

    # Apply temperature scaling to fc2 weights and bias
    weight = weight * best_T
    bias = bias * best_T

    # Update weights dictionary
    w['fc1.weight'] = W_fc1.astype(np.float32)
    w['fc1.bias'] = b_fc1.astype(np.float32)
    w['fc2.weight'] = weight.astype(np.float32)
    w['fc2.bias'] = bias.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "candidate": best_params[0],
        "alpha": float(alpha),
        "threshold": float(best_params[2]) if best_params[2] is not None else None,
        "temperature": float(best_T),
        "train_ce": float(best_ce),
        "fc1_weight_norm": float(np.linalg.norm(w['fc1.weight'])),
        "fc2_weight_norm": float(np.linalg.norm(w['fc2.weight'])),
        "fc2_bias_norm": float(np.linalg.norm(w['fc2.bias']))
    }
    return {"weights": w, "diagnostics": diagnostics}
