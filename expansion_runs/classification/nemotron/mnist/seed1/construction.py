def edit(w, f):
    """
    One‑shot feedback‑assisted construction of fc2 weights.
    - Uses the entire training pool (f["x"], f["y"]) and the current
      fc1 activations (f["features"] = layer.fc1.output).
    - Closed‑form ridge regression:  Y ≈ X W + b
    - No autograd, no iterative optimizer, no gradient estimation.
    """
    import numpy as np  # np is already available globally; we just reference it

    # ----- data preparation -----
    X = f["features"]                              # shape (2000, 128)
    y_idx = f["y"]                                 # shape (2000,), int64 0‑9
    N, D = X.shape                                 # N=2000, D=128
    C = 10                                          # number of classes

    # one‑hot targets Y (N, C)
    Y = np.eye(C, dtype=np.float32)[y_idx]        # (2000, 10)

    # ----- ridge regression (closed‑form) -----
    lam = 1e-3                                      # small ridge coefficient
    A = X.T @ X + lam * np.eye(D, dtype=np.float32)   # (128,128)
    # Solve A * W_raw = X.T @ Y  →  W_raw shape (128, 10)
    # Using np.linalg.solve for numerical stability
    XTY = X.T @ Y                                   # (128, 10)
    W_raw = np.linalg.solve(A, XTY)                # (128, 10)

    # bias: mean residual after weight fit
    b_vec = np.mean(Y - X @ W_raw, axis=0)         # (10,)

    # ----- map to network‑convention weight shape [out,in] -----
    # Network expects fc2.weight of shape (10, 128).  Our raw solve gave (128,10).
    fc2_weight_new = W_raw.T                       # (10, 128)

    # ----- update w (only fc2 parameters) -----
    w["fc2.weight"] = np.asarray(fc2_weight_new, dtype=np.float32)
    w["fc2.bias"]   = np.asarray(b_vec, dtype=np.float32)

    # ----- diagnostics (optional but useful for later reflection) -----
    # Compute logits, probabilities and accuracy on the *full* training pool
    logits_new = X @ W_raw + b_vec                 # (2000, 10)  (same as X @ fc2.weight.T + b)
    # numerically stable softmax
    max_logit = logits_new.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_new - max_logit)
    probs_new = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    acc = np.mean(np.argmax(probs_new, axis=1) == y_idx)
    # cross‑entropy loss (mean over training set)
    # clip probs to avoid log(0)
    eps = 1e-12
    ce = -np.mean(np.log(probs_new[np.arange(N), y_idx] + eps))

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "ridge_lambda": float(lam),
        "fc2_weight_mean": float(np.mean(fc2_weight_new)),
        "fc2_weight_std": float(np.std(fc2_weight_new)),
        "fc2_bias_mean": float(np.mean(b_vec)),
        "fc2_bias_std": float(np.std(b_vec)),
    }

    # Return the *full* weight dict together with a tiny diagnostics block.
    return {"weights": w, "diagnostics": diagnostics}
