import numpy as np

def edit(w, f):
    # Extract training data
    x = f['x'].reshape(f['x'].shape[0], -1).astype(np.float64)  # (2000, 784)
    y = f['y'].astype(np.int64)
    N = x.shape[0]

    # Current fc1 parameters
    W1 = w['fc1.weight'].astype(np.float64).copy()  # (128, 784)
    b1 = w['fc1.bias'].astype(np.float64).copy()    # (128,)

    # Compute pre-activations
    A = x @ W1.T + b1  # (N, 128)

    # Re-center each unit's pre-activation to zero median
    med = np.median(A, axis=0)  # (128,)
    b1_new = -med

    # New hidden features
    H = np.maximum(0.0, x @ W1.T + b1_new)  # (N, 128)

    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # 5-fold cross-validation to select alpha and standardization
    n_folds = 5
    indices = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    best_acc = -1.0
    best_alpha = None
    best_standardize = False

    for standardize in [False, True]:
        if standardize:
            H_use = (H - H.mean(axis=0)) / (H.std(axis=0) + 1e-6)
        else:
            H_use = H.copy()

        for alpha in alphas:
            accs = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

                H_train = H_use[train_idx]
                Y_train = Y[train_idx]
                H_val = H_use[val_idx]
                y_val = y[val_idx]

                # Center using training statistics
                H_mean = H_train.mean(axis=0)
                Y_mean = Y_train.mean(axis=0)
                H_c = H_train - H_mean
                Y_c = Y_train - Y_mean

                # Ridge solve
                A_mat = H_c.T @ H_c + alpha * np.eye(H.shape[1])
                W = np.linalg.solve(A_mat, H_c.T @ Y_c)
                b = Y_mean - H_mean @ W

                logits = H_val @ W + b
                pred = np.argmax(logits, axis=1)
                accs.append(np.mean(pred == y_val))

            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_alpha = alpha
                best_standardize = standardize

    # Refit on all data with best hyperparameters
    if best_standardize:
        H_use = (H - H.mean(axis=0)) / (H.std(axis=0) + 1e-6)
    else:
        H_use = H.copy()

    H_mean = H_use.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    H_c = H_use - H_mean
    Y_c = Y - Y_mean

    A_mat = H_c.T @ H_c + best_alpha * np.eye(H.shape[1])
    W = np.linalg.solve(A_mat, H_c.T @ Y_c)  # (128, 10)
    b = Y_mean - H_mean @ W                 # (10,)

    # Update weights
    w['fc1.weight'] = W1.astype(np.float32)
    w['fc1.bias'] = b1_new.astype(np.float32)
    w['fc2.weight'] = W.T.astype(np.float32)  # (10, 128)
    w['fc2.bias'] = b.astype(np.float32)

    # Clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        'best_alpha': best_alpha,
        'best_standardize': best_standardize,
        'cv_accuracy': float(best_acc),
        'fc1_weight_norm': float(np.linalg.norm(w['fc1.weight'])),
        'fc1_bias_norm': float(np.linalg.norm(w['fc1.bias'])),
        'fc2_weight_norm': float(np.linalg.norm(w['fc2.weight'])),
        'fc2_bias_norm': float(np.linalg.norm(w['fc2.bias'])),
    }

    return {'weights': w, 'diagnostics': diagnostics}
