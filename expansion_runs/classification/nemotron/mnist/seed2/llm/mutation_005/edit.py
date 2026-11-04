def edit(w, f):
    """
    Closed‑form ridge‑regression update of fc2 on the full feedback pool.
    Keeps fc1 weights unchanged.
    Returns updated weight dict and a small diagnostics dict.
    """
    # ----- feedback pool -------------------------------------------------
    X = f["features"]                # (2000, 128)  hidden activations before fc2
    y = f["y"]                     # (2000,)      integer labels 0‑9
    N, D = X.shape                 # N=2000, D=128
    C = 10                         # number of classes

    # one‑hot encode the labels
    Y = np.eye(C)[y]               # (2000, 10)

    # ----- ridge regularisation -----------------------------------------
    lam = 0.1                       # small ridge coefficient
    A = X.T @ X + lam * np.eye(D)   # (128, 128)
    B = X.T @ Y                     # (128, 10)

    # solve (XᵀX + λI) w = XᵀY  →  w_raw shape (128,10)
    w_raw = np.linalg.solve(A, B)   # numerically stable, no loop over classes

    # fc2.weight is (10,128); transpose to match convention
    new_fc2_weight = w_raw.T.astype(np.float32)   # (10, 128)

    # ----- optional bias update -----------------------------------------
    # bias = mean( Y - X @ weightᵀ ) over the pool
    residuals = Y - X @ new_fc2_weight.T       # (2000, 10)
    new_fc2_bias = np.mean(residuals, axis=0).astype(np.float32)   # (10,)

    # ----- apply update -------------------------------------------------
    w_new = dict(w)                     # shallow copy
    w_new["fc2.weight"] = new_fc2_weight
    w_new["fc2.bias"] = new_fc2_bias

    # ----- diagnostics on the feedback pool ----------------------------
    logits_train = X @ new_fc2_weight.T + new_fc2_bias[np.newaxis, :]   # (2000,10)
    # numerically stable softmax
    max_logit = logits_train.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_train - max_logit)
    probs_train = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    preds = probs_train.argmax(axis=1)
    accuracy = (preds == y).mean()

    # cross‑entropy
    correct_lp = -np.log(probs_train[np.arange(N), y] + 1e-12)
    ce = correct_lp.mean()

    diagnostics = {
        "fc2_weight_mean": float(new_fc2_weight.mean()),
        "fc2_weight_std": float(new_fc2_weight.std()),
        "fc2_bias_mean": float(new_fc2_bias.mean()),
        "train_accuracy": float(accuracy),
        "train_cross_entropy": float(ce),
        "ridge_lambda": float(lam)
    }

    return {"weights": w_new, "diagnostics": diagnostics}
