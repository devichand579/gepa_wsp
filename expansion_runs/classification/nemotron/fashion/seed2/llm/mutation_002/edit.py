def edit(w, f):
    """
    Edit the network's head weights via closed‑form ridge regression
    on the entire permitted feedback pool.
    """
    # ----- feedback pool -----
    X = f["features"]                 # shape (2000, 1568), float32
    y = f["y"]                        # shape (2000,), int64
    K = 10                            # number of classes

    # one‑hot encode labels
    Y = np.eye(K, dtype=np.float32)[y]   # (2000, 10)

    # augment features with a bias column
    D = X.shape[1]                     # 1568
    F_aug = np.hstack([
        X,
        np.ones((X.shape[0], 1), dtype=np.float32)
    ])                                 # (2000, 1569)

    lam = 1e-3                         # ridge regularisation strength

    # normal‑equation matrices
    A = F_aug.T @ F_aug + lam * np.eye(D + 1, dtype=np.float64)   # (1569, 1569)
    B = F_aug.T @ Y                     # (1569, 10)

    # closed‑form solve for augmented weight matrix
    W_aug = np.linalg.solve(A, B)       # (1569, 10)

    # split into head weight and bias
    head_weight_new = W_aug[:D].T.astype(np.float32)   # (10, 1568)
    head_bias_new   = W_aug[D].astype(np.float32)      # (10,)

    # replace head parameters in the current weight dict
    w["head.weight"] = head_weight_new
    w["head.bias"]   = head_bias_new

    # ----- diagnostics on the feedback pool -----
    logits_train = X @ head_weight_new.T + head_bias_new   # (2000, 10)
    pred = np.argmax(logits_train, axis=1)
    train_acc = float(np.mean(pred == y))

    # cross‑entropy (softmax)
    exp_logits = np.exp(logits_train - logits_train.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    correct_probs = probs[np.arange(X.shape[0]), y]
    train_xent = -np.mean(np.log(correct_probs + 1e-12))

    diagnostics = {
        "train_accuracy": train_acc,
        "train_cross_entropy": float(train_xent)
    }

    return {"weights": w, "diagnostics": diagnostics}
