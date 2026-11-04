def edit(w, f):
    """
    Edit fc2 weight matrix and bias using a closed‑form ridge regression
    on the entire 2000‑example feedback pool.  fc1 weights are left untouched.
    """
    import numpy as np

    # ----- data from feedback dict -----
    features = f["features"]               # shape (2000, 128)
    y        = f["y"]                      # shape (2000,), int64 labels

    # one‑hot encode labels (2000 x 10)
    Y = np.zeros((2000, 10), dtype=np.float32)
    Y[np.arange(2000), y] = 1.0

    # ----- ridge regression parameters -----
    lam = 0.1                              # small ridge regularisation
    I128 = np.eye(128, dtype=np.float32)

    # Gram matrix (128 x 128)
    gram = features.T @ features

    # closed‑form solution for fc2 weight (128 x 10)
    #   w = (F^T F + λ I)^{-1} F^T Y
    w_tilde = np.linalg.solve(gram + lam * I128, features.T @ Y)  # (128, 10)

    # bias: negative mean of the residuals
    resid = Y - features @ w_tilde        # (2000, 10)
    b_new = -resid.mean(axis=0)            # (10,)

    # ----- update weights in the dict -----
    # store transposed to match PyTorch convention [out, in] = [10, 128]
    w["fc2.weight"] = w_tilde.T.astype(np.float32)   # (10, 128)
    w["fc2.bias"]   = b_new.astype(np.float32)       # (10,)

    # ----- diagnostics -----
    # recompute logits and probabilities for the full pool
    logits = features @ w_tilde.T + b_new          # (2000, 10)
    # numerically stable softmax
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    pred = np.argmax(probs, axis=1)                # (2000,)
    acc  = (pred == y).mean()

    correct_probs = probs[np.arange(2000), y]
    ce = -np.log(correct_probs).mean()

    diag = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "fc2_weight_norm": float(np.linalg.norm(w_tilde)),
        "fc2_bias": b_new.tolist()
    }

    return {"weights": w, "diagnostics": diag}
