import numpy as np

def edit(w, f):
    # Copy current weights
    W1 = w["fc1.weight"].copy()
    b1 = w["fc1.bias"].copy()
    W2 = w["fc2.weight"].copy()
    b2 = w["fc2.bias"].copy()

    # Extract data
    X_raw = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # (N, 784)
    y = f["y"].astype(np.int64)
    X_feat = f["features"].astype(np.float64)  # (N, 128)
    N = len(y)
    classes = np.unique(y)
    n_classes = len(classes)

    # Current accuracy
    logits_cur = X_feat @ W2.T + b2
    pred_cur = np.argmax(logits_cur, axis=1)
    acc_cur = np.mean(pred_cur == y)

    def evaluate(W2_cand, b2_cand, X=X_feat):
        logits = X @ W2_cand.T + b2_cand
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == y)
        logits_shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shift)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        ce = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        return acc, ce

    best_acc, best_ce = evaluate(W2, b2)
    best_W2, best_b2 = W2, b2
    best_W1, best_b1 = W1, b1
    best_X = X_feat

    # --- Try LDA on current features ---
    mu = np.zeros((n_classes, X_feat.shape[1]))
    for i, c in enumerate(classes):
        mu[i] = np.mean(X_feat[y == c], axis=0)
    S = np.zeros((X_feat.shape[1], X_feat.shape[1]))
    for i, c in enumerate(classes):
        Xc = X_feat[y == c] - mu[i]
        S += Xc.T @ Xc
    S /= (N - n_classes)

    for alpha in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        S_reg = S + alpha * np.trace(S) / X_feat.shape[1] * np.eye(X_feat.shape[1])
        try:
            W_lda = np.linalg.solve(S_reg, mu.T).T
        except np.linalg.LinAlgError:
            continue
        b_lda = -0.5 * np.sum(mu * W_lda, axis=1) + np.log(np.bincount(y, minlength=n_classes) / N)
        acc, ce = evaluate(W_lda, b_lda)
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc, best_ce = acc, ce
            best_W2, best_b2 = W_lda, b_lda
            best_W1, best_b1 = W1, b1
            best_X = X_feat

    # --- Try ridge regression on current features ---
    Y_onehot = np.zeros((N, n_classes))
    Y_onehot[np.arange(N), y] = 1.0
    X_aug = np.hstack([X_feat, np.ones((N, 1))])
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        A = X_aug.T @ X_aug + alpha * np.eye(X_aug.shape[1])
        B = X_aug.T @ Y_onehot
        try:
            W_ridge = np.linalg.solve(A, B).T
        except np.linalg.LinAlgError:
            continue
        W2_cand = W_ridge[:, :128]
        b2_cand = W_ridge[:, 128]
        acc, ce = evaluate(W2_cand, b2_cand)
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc, best_ce = acc, ce
            best_W2, best_b2 = W2_cand, b2_cand
            best_W1, best_b1 = W1, b1
            best_X = X_feat

    # --- Replace dead fc1 units with informative filters ---
    dead_mask = np.all(np.abs(X_feat) < 1e-6, axis=0)
    n_dead = int(np.sum(dead_mask))
    if n_dead > 0:
        overall_mean = np.mean(X_raw, axis=0)
        mu_raw = np.zeros((n_classes, 784))
        for i, c in enumerate(classes):
            mu_raw[i] = np.mean(X_raw[y == c], axis=0)
        mu_centered = mu_raw - overall_mean
        norms = np.linalg.norm(mu_centered, axis=1, keepdims=True)
        norms[norms == 0] = 1
        mu_centered = mu_centered / norms

        diffs = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                d = mu_raw[i] - mu_raw[j]
                d = d / (np.linalg.norm(d) + 1e-12)
                diffs.append(d)
        diffs = np.array(diffs) if diffs else np.zeros((0, 784))

        X_raw_centered = X_raw - overall_mean
        U, S, Vt = np.linalg.svd(X_raw_centered, full_matrices=False)
        pca_filters = Vt[:128]
        pca_filters = pca_filters / (np.linalg.norm(pca_filters, axis=1, keepdims=True) + 1e-12)

        pool = np.vstack([mu_centered, diffs, pca_filters])

        new_W1 = W1.copy()
        new_b1 = b1.copy()
        dead_indices = np.where(dead_mask)[0]
        for idx, unit in enumerate(dead_indices):
            if idx < len(pool):
                filt = pool[idx]
            else:
                filt = np.random.randn(784)
                filt = filt / (np.linalg.norm(filt) + 1e-12)
            new_W1[unit] = filt
            pre = X_raw @ filt
            new_b1[unit] = -np.mean(pre)

        X_new = np.maximum(0, X_raw @ new_W1.T + new_b1)

        # Refit fc2 on new features using LDA and ridge
        mu_new = np.zeros((n_classes, X_new.shape[1]))
        for i, c in enumerate(classes):
            mu_new[i] = np.mean(X_new[y == c], axis=0)
        S_new = np.zeros((X_new.shape[1], X_new.shape[1]))
        for i, c in enumerate(classes):
            Xc = X_new[y == c] - mu_new[i]
            S_new += Xc.T @ Xc
        S_new /= (N - n_classes)

        for alpha in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            S_reg = S_new + alpha * np.trace(S_new) / X_new.shape[1] * np.eye(X_new.shape[1])
            try:
                W_lda = np.linalg.solve(S_reg, mu_new.T).T
            except np.linalg.LinAlgError:
                continue
            b_lda = -0.5 * np.sum(mu_new * W_lda, axis=1) + np.log(np.bincount(y, minlength=n_classes) / N)
            acc, ce = evaluate(W_lda, b_lda, X=X_new)
            if acc > best_acc or (acc == best_acc and ce < best_ce):
                best_acc, best_ce = acc, ce
                best_W2, best_b2 = W_lda, b_lda
                best_W1, best_b1 = new_W1, new_b1
                best_X = X_new

        X_aug_new = np.hstack([X_new, np.ones((N, 1))])
        for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            A = X_aug_new.T @ X_aug_new + alpha * np.eye(X_aug_new.shape[1])
            B = X_aug_new.T @ Y_onehot
            try:
                W_ridge = np.linalg.solve(A, B).T
            except np.linalg.LinAlgError:
                continue
            W2_cand = W_ridge[:, :128]
            b2_cand = W_ridge[:, 128]
            acc, ce = evaluate(W2_cand, b2_cand, X=X_new)
            if acc > best_acc or (acc == best_acc and ce < best_ce):
                best_acc, best_ce = acc, ce
                best_W2, best_b2 = W2_cand, b2_cand
                best_W1, best_b1 = new_W1, new_b1
                best_X = X_new

    # Update weights
    w["fc1.weight"] = best_W1.astype(np.float32)
    w["fc1.bias"] = best_b1.astype(np.float32)
    w["fc2.weight"] = best_W2.astype(np.float32)
    w["fc2.bias"] = best_b2.astype(np.float32)

    # Clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "acc_cur": float(acc_cur),
        "best_acc": float(best_acc),
        "best_ce": float(best_ce),
        "n_dead": n_dead,
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
