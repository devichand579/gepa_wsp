import numpy as np

def edit(w, f):
    # Extract current fc1 parameters
    W1 = w["fc1.weight"].astype(np.float32)  # (128, 784)
    b1 = w["fc1.bias"].astype(np.float32)    # (128,)

    # Helper: compute hidden features for a batch of images
    def get_features(images):
        # images: (N, 1, 28, 28)
        N = images.shape[0]
        flat = images.reshape(N, -1).astype(np.float32)
        h = flat @ W1.T + b1
        return np.maximum(h, 0.0)

    # Helper: shift images by (dy, dx) with zero padding
    def shift_images(images, dy, dx):
        padded = np.pad(images, ((0,0),(0,0),(2,2),(2,2)), mode='constant')
        start_y = 2 - dy
        start_x = 2 - dx
        return padded[:, :, start_y:start_y+28, start_x:start_x+28]

    # Build augmented features and labels
    shifts = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
    F_aug_list = []
    y_aug_list = []
    for dy, dx in shifts:
        imgs = shift_images(f["x"], dy, dx)
        F_aug_list.append(get_features(imgs))
        y_aug_list.append(f["y"])
    F_aug = np.concatenate(F_aug_list, axis=0)  # (50000, 128)
    y_aug = np.concatenate(y_aug_list, axis=0)  # (50000,)

    N = len(f["y"])
    orig_idx_aug = np.arange(len(F_aug)) % N

    # Split original indices for validation
    rng = np.random.RandomState(42)
    indices = np.arange(N)
    rng.shuffle(indices)
    train_idx = indices[:1600]
    val_idx = indices[1600:]

    # Training mask: all augmented samples whose original index is in train_idx
    train_mask = np.isin(orig_idx_aug, train_idx)
    F_train = F_aug[train_mask]
    y_train = y_aug[train_mask]

    # Validation: original (unshifted) features for val_idx
    F_val = F_aug[val_idx]   # shift (0,0) is the first N rows
    y_val = y_aug[val_idx]

    # LDA fitting function
    def fit_lda(F, y, alpha):
        N_, D = F.shape
        classes = np.unique(y)
        C = len(classes)
        mu = np.zeros((C, D), dtype=np.float64)
        priors = np.zeros(C, dtype=np.float64)
        for i, c in enumerate(classes):
            mask = (y == c)
            mu[i] = F[mask].mean(axis=0)
            priors[i] = mask.sum() / N_
        S_w = np.zeros((D, D), dtype=np.float64)
        for i, c in enumerate(classes):
            mask = (y == c)
            Xc = F[mask] - mu[i]
            S_w += Xc.T @ Xc
        S_w /= (N_ - C)
        S_w_reg = S_w + alpha * np.eye(D)
        W = np.linalg.solve(S_w_reg, mu.T).T  # (C, D)
        b = -0.5 * np.sum(W * mu, axis=1) + np.log(priors)
        return W, b

    def predict(F, W, b):
        logits = F @ W.T + b
        return np.argmax(logits, axis=1)

    # Choose regularization alpha via validation
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_acc = -1.0
    for alpha in alphas:
        W, b = fit_lda(F_train, y_train, alpha)
        pred = predict(F_val, W, b)
        acc = np.mean(pred == y_val)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    # Refit on all augmented data with best alpha
    W_final, b_final = fit_lda(F_aug, y_aug, best_alpha)

    # Clip to safe range
    W_final = np.clip(W_final, -1000.0, 1000.0)
    b_final = np.clip(b_final, -1000.0, 1000.0)

    # Update weights
    w["fc2.weight"] = W_final.astype(np.float32)
    w["fc2.bias"] = b_final.astype(np.float32)

    diagnostics = {
        "best_alpha": float(best_alpha),
        "val_accuracy": float(best_acc),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "fc2_weight_min": float(w["fc2.weight"].min()),
        "fc2_weight_max": float(w["fc2.weight"].max()),
        "fc2_bias_min": float(w["fc2.bias"].min()),
        "fc2_bias_max": float(w["fc2.bias"].max()),
    }

    return {"weights": w, "diagnostics": diagnostics}
