def edit(w, f):
    """
    Edit the neural network weights using a closed‑form ridge regression
    on the full feedback pool. Only fc2 parameters are changed; fc1 stays
    untouched. Returns the modified weight dict and a small diagnostics
    dictionary.
    """
    import numpy as np

    # ----- full feedback pool -----
    X = f["features"].astype(np.float32)          # (2000, 128)
    y = f["y"].astype(np.int64)                   # (2000,)
    C = 10                                        # number of classes
    D = X.shape[1]                                # 128

    # one‑hot encode labels
    Y_onehot = np.eye(C, dtype=np.float32)[y]    # (2000, 10)

    # ridge regularisation (small lambda)
    lam = 0.1
    # normal‑equation matrices
    XtX = X.T @ X + lam * np.eye(D, dtype=np.float32)          # (128,128)
    XtX_inv = np.linalg.inv(XtX)

    # closed‑form weight matrix: W = Y^T X (X^T X + λI)^{-1}
    W_raw = Y_onehot.T @ X                                 # (10,128)
    fc2_weight = (W_raw @ XtX_inv).astype(np.float32)       # (10,128)

    # bias: average residual after fitting
    logits_pred = X @ fc2_weight.T                         # (2000,10)
    residual = Y_onehot - logits_pred                      # (2000,10)
    fc2_bias = residual.mean(axis=0).astype(np.float32)    # (10,)

    # ----- edit the weight dict -----
    # fc1 remains unchanged; only fc2 is overwritten
    w["fc2.weight"] = fc2_weight
    w["fc2.bias"] = fc2_bias

    # ----- diagnostics -----
    # evaluate on the paired reflection batch (using current fc1)
    batch_x = f["batch.x"].astype(np.float32)            # (128,1,28,28)
    batch_y = f["batch.y"].astype(np.int64)             # (128,)

    # forward through fc1 (unchanged)
    batch_flat = batch_x.reshape(batch_x.shape[0], -1)   # (128,784)
    h = np.maximum(0.0, batch_flat @ w["fc1.weight"].T + w["fc1.bias"])   # (128,128)

    # forward through new fc2
    batch_logits = h @ fc2_weight.T + fc2_bias           # (128,10)
    batch_pred = np.argmax(batch_logits, axis=1)
    batch_accuracy = float(np.mean(batch_pred == batch_y))

    # weight norms
    w_norm = float(np.linalg.norm(fc2_weight.ravel()))
    b_norm = float(np.linalg.norm(fc2_bias))

    diagnostics = {
        "batch_accuracy": batch_accuracy,
        "fc2_weight_norm": w_norm,
        "fc2_bias_norm": b_norm,
        "ridge_lambda": float(lam)
    }

    return {"weights": w, "diagnostics": diagnostics}
