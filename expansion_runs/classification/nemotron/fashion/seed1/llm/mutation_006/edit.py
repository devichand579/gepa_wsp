# ------------------------------------------------------------
# edit(w, f) – closed‑form head‑only weight update
# ------------------------------------------------------------
def edit(w, f):
    """
    Update the network's head (linear classifier) by ordinary least‑squares
    on the full training set (features vs. one‑hot labels).  All other
    parameters are left untouched.

    Parameters
    ----------
    w : dict
        Current full‑precision parameter dictionary (contains
        ``conv1.weight``, ``conv1.bias``, ``conv2.weight``,
        ``conv2.bias``, ``head.weight``, ``head.bias``).
    f : dict
        Feedback dictionary with the keys listed in the prompt
        (``x``, ``y``, ``features``, ``logits``, ``probs``, etc.).

    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        ``w`` with the updated ``head.weight`` and ``head.bias``;
        ``diagnostics`` contains the mean/std of the new head weights,
        the training accuracy after the edit, and the previous accuracy.
    """
    # ---- copy current weights so we only modify the head ----------
    w = {k: v.copy() for k, v in w.items()}

    # ---- extract the permitted feedback arrays --------------------
    X = f["features"]                 # shape (2000, 1568), float32
    y = f["y"]                        # shape (2000,), int64

    # ---- build one‑hot target matrix -------------------------------
    N, C = X.shape[0], 10
    Y_onehot = np.eye(C, dtype=np.float32)[y]   # (2000, 10)

    # ---- closed‑form least‑squares solution for head weight -------
    # We solve  X @ W^T ≈ Y_onehot  →  W = Y_onehot.T @ pinv(X)
    # np.linalg.pinv gives the Moore‑Penrose pseudo‑inverse.
    W_new = (Y_onehot.T @ np.linalg.pinv(X)).astype(np.float32)
    # W_new shape (10, 1568) – matches head.weight layout

    # ---- set the new head weight ---------------------------------
    w["head.weight"] = W_new

    # ---- bias: simple mean of the one‑hot targets ----------------
    b_new = Y_onehot.mean(axis=0).astype(np.float32)   # shape (10,)
    w["head.bias"] = b_new

    # ---- compute training accuracy with the new head -------------
    logits_new = X @ W_new.T + b_new          # (2000, 10)
    pred = np.argmax(logits_new, axis=1)
    acc = (pred == y).mean()

    # ---- diagnostics ------------------------------------------------
    diag = {
        "head_weight_mean": float(W_new.mean()),
        "head_weight_std":  float(W_new.std()),
        "head_bias_mean":   float(b_new.mean()),
        "train_accuracy":   float(acc),
        "previous_accuracy": 0.099   # from feedback_arrays["full_feedback_metrics"]["accuracy"]
    }

    return {"weights": w, "diagnostics": diag}
