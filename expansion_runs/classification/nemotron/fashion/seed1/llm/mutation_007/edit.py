def edit(w, f):
    """
    Closed‑form re‑training of the network's head via ridge regression on
    the entire permitted feedback pool.  Only the head weights (and bias)
    are modified; all convolutional parameters stay unchanged.
    """
    # shallow copy of the full weight dict
    w = {k: v.copy() for k, v in w.items()}

    # --- feedback arrays ---
    features = f["features"]               # shape (2000, 1568), float32
    y        = f["y"]                      # shape (2000,), int64

    N, D = features.shape                  # N=2000, D=1568

    # one‑hot encode the labels -> (N, 10)
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # ridge regularisation (small lambda for numerical stability)
    lam = np.float32(1e-3)

    # normal‑equation matrices
    XtX = features.T @ features + lam * np.eye(D, dtype=np.float32)   # (D, D)
    XtY = features.T @ Y                                            # (D, 10)

    # solve for the un‑transposed weight matrix (D, 10)
    A = np.linalg.solve(XtX, XtY)       # (D, 10)

    # PyTorch linear expects weight of shape (out_features, in_features) = (10, D)
    head_weight = A.T.astype(np.float32)            # (10, 1568)

    # bias: average residual after the linear transformation
    logits_pred = features @ head_weight.T        # (N, 10)
    bias = np.mean(Y - logits_pred, axis=0).astype(np.float32)   # (10,)

    # --- update head parameters ---
    w["head.weight"] = head_weight
    w["head.bias"]   = bias

    # --- diagnostics (small JSON‑compatible dict) ---
    preds = np.argmax(logits_pred, axis=1)
    train_acc = float(np.mean(preds == y))

    diag = {
        "train_accuracy": train_acc,
        "head_weight_mean": float(np.mean(head_weight)),
        "head_weight_std":  float(np.std(head_weight)),
        "head_bias_mean":   float(np.mean(bias)),
    }

    return {"weights": w, "diagnostics": diag}
