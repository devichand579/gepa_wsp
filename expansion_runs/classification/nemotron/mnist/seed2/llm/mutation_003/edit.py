def edit(w, f):
    """
    Edit the network weights using a closed‑form ridge regression on the
    full feedback pool.  Only fc2 weights and bias are changed; fc1 remains
    untouched so that the cached activations stay valid.

    Parameters
    ----------
    w : dict
        Full current parameter dictionary (mutable).  Keys "fc1.weight",
        "fc1.bias", "fc2.weight", "fc2.bias" exist with the shapes defined
        in the network description.
    f : dict
        Read‑only feedback arrays with the exact keys/shapes listed in the
        problem statement.  In particular f["features"] (shape [2000,128])
        and f["y"] (shape [2000]) are used.

    Returns
    -------
    {"weights": w, "diagnostics": diag}
        diag is a small JSON‑compatible statistics dictionary.
    """
    # ------------------------------------------------------------------
    # 1. Pull the permitted feedback arrays
    # ------------------------------------------------------------------
    features = f["features"]            # shape (2000, 128), dtype float32
    y        = f["y"]                   # shape (2000,), int64, values 0..9

    # ------------------------------------------------------------------
    # 2. Convert class labels to one‑hot vectors (N, C)
    # ------------------------------------------------------------------
    N, C = features.shape[0], 10
    Y_onehot = np.eye(C, dtype=np.float32)[y.astype(np.int32)]  # (N,10)

    # ------------------------------------------------------------------
    # 3. Ridge‑regression closed‑form solution for fc2 weight matrix
    # ------------------------------------------------------------------
    lam = 0.1                     # regularisation strength (moderate)

    # Gram matrix of features: (128, 128)
    gram = features.T.astype(np.float32) @ features.astype(np.float32)

    # Right‑hand side: (128, 10)  =  features^T @ Y_onehot
    rhs = features.T.astype(np.float32) @ Y_onehot

    # Solve (gram + λ I) X = rhs  for X of shape (128, 10)
    I128 = np.eye(128, dtype=np.float32)
    X = np.linalg.solve(gram + lam * I128, rhs)   # (128, 10)

    # The fc2 weight matrix has shape (out, in) = (10, 128)
    W2_new = X.T.astype(np.float32)               # (10, 128)

    # ------------------------------------------------------------------
    # 4. Compute the new fc2 bias (mean of residuals)
    # ------------------------------------------------------------------
    residuals = Y_onehot - features @ X           # (N, 10)
    b2_new = residuals.mean(axis=0).astype(np.float32)  # (10,)

    # ------------------------------------------------------------------
    # 5. Edit the weight dictionary in‑place
    # ------------------------------------------------------------------
    w["fc2.weight"] = W2_new
    w["fc2.bias"]   = b2_new

    # ------------------------------------------------------------------
    # 6. Collect diagnostics (all values are plain Python floats)
    # ------------------------------------------------------------------
    diag = {
        "fc2_weight_mean": float(W2_new.mean()),
        "fc2_weight_std":  float(W2_new.std()),
        "fc2_bias_mean":   float(b2_new.mean()),
        "fc2_bias_std":    float(b2_new.std()),
        "ridge_lambda":    lam,
        "train_features_mean": float(features.mean()),
    }

    return {"weights": w, "diagnostics": diag}
