def edit(w, f):
    """
    Edit the network weights using a closed‑form ridge regression on the
    full feedback pool.  Only fc2 parameters are changed; fc1 stays intact.
    Returns the modified weight dict together with a small diagnostics dict.
    """
    # ------- data preparation -------
    # Full feedback pools
    y = f["y"]                              # shape (2000,), int64
    features = f["features"]                # shape (2000, 128)

    n_samples = features.shape[0]
    n_classes = 10

    # One‑hot encode the labels
    Y = np.eye(n_classes, dtype=np.float32)[y]   # (2000, 10)

    H = features.astype(np.float32)              # (2000, 128)

    # ------- ridge regression -------
    lam = 0.1                                     # small ridge coefficient

    # Normal equations: (H^T H + λ I) W^T = H^T Y
    HtH = H.T @ H + lam * np.eye(128, dtype=np.float32)   # (128, 128)
    HtY = H.T @ Y                                      # (128, 10)

    # Solve for W2^T; then transpose to get W2 of shape (10, 128)
    Wt = np.linalg.solve(HtH, HtY)                   # (128, 10)
    W2_new = Wt.T.astype(np.float32)                 # (10, 128)

    # Bias: mean residual after the linear map
    pred_logits = H @ W2_new.T                       # (2000, 10)
    b2_new = np.mean(Y - pred_logits, axis=0).astype(np.float32)   # (10,)

    # ------- apply the updates -------
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new

    # ------- diagnostics -------
    # Accuracy on the reflection batch (128 samples)
    H_batch = f["layer.fc1.output"]                # (128, 128)
    pred_batch = H_batch @ W2_new.T                # (128, 10)
    pred_labels = np.argmax(pred_batch, axis=1)
    batch_y = f["batch.y"].astype(int)
    acc = float(np.mean(pred_labels == batch_y))

    diag = {
        "reflection_batch_accuracy_after": acc,
        "fc2_weight_norm": float(np.linalg.norm(W2_new)),
        "fc2_bias_norm": float(np.linalg.norm(b2_new)),
        "ridge_lambda": float(lam),
    }

    return {"weights": w, "diagnostics": diag}
