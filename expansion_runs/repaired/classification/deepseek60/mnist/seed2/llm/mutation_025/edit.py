import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    H = f["features"].astype(np.float64)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    n, d = H.shape
    n_classes = 10

    # One-hot targets
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    # Compute class means and overall mean
    mu = H.mean(axis=0)                           # (128,)
    class_means = np.zeros((n_classes, d), dtype=np.float64)
    class_counts = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        mask = (y == c)
        class_counts[c] = mask.sum()
        if class_counts[c] > 0:
            class_means[c] = H[mask].mean(axis=0)

    # Within-class scatter matrix Sw (128x128)
    Sw = np.zeros((d, d), dtype=np.float64)
    for c in range(n_classes):
        mask = (y == c)
        if class_counts[c] > 1:
            Xc = H[mask]
            Xc_centered = Xc - class_means[c]
            Sw += Xc_centered.T @ Xc_centered

    # Between-class scatter matrix Sb (128x128)
    Sb = np.zeros((d, d), dtype=np.float64)
    for c in range(n_classes):
        diff = class_means[c] - mu
        Sb += class_counts[c] * np.outer(diff, diff)

    # Cross-validation to choose shrinkage alpha for LDA
    # We'll use 5-fold CV, but we need to recompute Sw and Sb on each fold.
    # To save time, we can use the full Sw and Sb for CV? No, that would leak.
    # We'll do a proper 5-fold CV.
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    alphas = np.logspace(-4, 2, 20)  # 20 values from 1e-4 to 1e2
    best_alpha = 1.0
    best_acc = -1.0

    for alpha in alphas:
        acc_sum = 0.0
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            H_train = H[train_idx]
            y_train = y[train_idx]
            H_val = H[val_idx]
            y_val = y[val_idx]

            # Compute Sw and Sb on training fold
            mu_tr = H_train.mean(axis=0)
            class_means_tr = np.zeros((n_classes, d), dtype=np.float64)
            class_counts_tr = np.zeros(n_classes, dtype=np.float64)
            for c in range(n_classes):
                mask = (y_train == c)
                class_counts_tr[c] = mask.sum()
                if class_counts_tr[c] > 0:
                    class_means_tr[c] = H_train[mask].mean(axis=0)

            Sw_tr = np.zeros((d, d), dtype=np.float64)
            for c in range(n_classes):
                mask = (y_train == c)
                if class_counts_tr[c] > 1:
                    Xc = H_train[mask]
                    Xc_centered = Xc - class_means_tr[c]
                    Sw_tr += Xc_centered.T @ Xc_centered

            Sb_tr = np.zeros((d, d), dtype=np.float64)
            for c in range(n_classes):
                diff = class_means_tr[c] - mu_tr
                Sb_tr += class_counts_tr[c] * np.outer(diff, diff)

            # LDA with shrinkage: Sw_reg = Sw_tr + alpha * I
            Sw_reg = Sw_tr + alpha * np.eye(d)
            # Solve for weights: W = Sw_reg^{-1} @ class_means_tr.T
            # Then bias = -0.5 * diag(class_means_tr @ W) + log(prior)
            try:
                W = np.linalg.solve(Sw_reg, class_means_tr.T).T  # (10, 128)
            except np.linalg.LinAlgError:
                continue
            b = -0.5 * np.sum(class_means_tr * W, axis=1)
            # Add log prior (optional, but classes are balanced in MNIST)
            # priors = class_counts_tr / class_counts_tr.sum()
            # b += np.log(priors)

            # Predict on validation
            logits_val = H_val @ W.T + b
            preds = np.argmax(logits_val, axis=1)
            acc_sum += (preds == y_val).mean()

        avg_acc = acc_sum / 5.0
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_alpha = alpha

    # Now fit on all data with best_alpha
    Sw_reg = Sw + best_alpha * np.eye(d)
    W = np.linalg.solve(Sw_reg, class_means.T).T  # (10, 128)
    b = -0.5 * np.sum(class_means * W, axis=1)
    # priors = class_counts / class_counts.sum()
    # b += np.log(priors)

    # Update fc2 weights and bias
    w["fc2.weight"] = W.astype(np.float32)
    w["fc2.bias"] = b.astype(np.float32)

    # Optionally, we can also adjust fc1 biases? Not needed for this proposal.
    # We keep fc1 as is.

    diagnostics = {
        "best_alpha": float(best_alpha),
        "cv_accuracy": float(best_acc),
        "fc2_weight_norm": float(np.linalg.norm(W)),
        "fc2_bias_norm": float(np.linalg.norm(b)),
        "fc2_weight_min": float(W.min()),
        "fc2_weight_max": float(W.max()),
        "fc2_bias_min": float(b.min()),
        "fc2_bias_max": float(b.max()),
    }
    return {"weights": w, "diagnostics": diagnostics}
