import numpy as np

def edit(w, f):
    """
    Closed‑form ridge‑regression adjustment of the final head weights
    using the entire feedback pool. Returns the modified weight dict
    and a small diagnostics dictionary.
    """
    # ------------------------------------------------------------------
    # 1. Pull feedback arrays (read‑only)
    # ------------------------------------------------------------------
    X = f["features"].astype(np.float32)          # shape (4000, 32)
    y = f["y"].astype(np.int64)                   # shape (4000,)

    # ------------------------------------------------------------------
    # 2. One‑hot encode labels
    # ------------------------------------------------------------------
    N = X.shape[0]
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # ------------------------------------------------------------------
    # 3. Save original head parameters for diagnostics
    # ------------------------------------------------------------------
    Hw_orig = w["head.weight"].astype(np.float32).copy()   # (2, 32)
    Hb_orig = w["head.bias"].astype(np.float32).copy()     # (2,)

    # original logits / accuracy on the feedback pool (using stale features)
    logits_orig = X @ Hw_orig.T + Hb_orig                 # (N, 2)
    pred_orig = np.argmax(logits_orig, axis=1)
    acc_orig = float((pred_orig == y).mean())

    # ------------------------------------------------------------------
    # 4. Ridge‑regression on head weights
    # ------------------------------------------------------------------
    lam = np.float32(1e-3)                # ridge penalty
    I32 = np.eye(32, dtype=np.float32)
    gram = X.T @ X + lam * I32            # (32, 32)
    XtY = X.T @ Y                         # (32, 2)
    W_raw = np.linalg.solve(gram, XtY)   # (32, 2)  -> columns are weights for each class

    # new head weight: transpose to (2, 32) convention
    Hw_new = W_raw.T.astype(np.float32)    # (2, 32)

    # bias: mean residual after applying the new weight
    logits_new = X @ W_raw.T + np.mean(Y - X @ W_raw.T, axis=0, keepdims=True)  # optional
    # simpler: compute bias as mean of (Y - X @ W_raw.T)
    resid = Y - X @ W_raw.T               # (N, 2)
    Hb_new = resid.mean(axis=0).astype(np.float32)   # (2,)

    # ------------------------------------------------------------------
    # 5. Write new head parameters back into w
    # ------------------------------------------------------------------
    w["head.weight"] = Hw_new
    w["head.bias"]   = Hb_new

    # ------------------------------------------------------------------
    # 6. Compute new accuracy on the feedback pool (using the same stale features)
    # ------------------------------------------------------------------
    logits_new2 = X @ Hw_new.T + Hb_new          # (N, 2)
    pred_new = np.argmax(logits_new2, axis=1)
    acc_new = float((pred_new == y).mean())

    # ------------------------------------------------------------------
    # 7. Diagnostics (JSON‑compatible, small)
    # ------------------------------------------------------------------
    diag = {
        "lambda": float(lam),
        "head_weight_norm_before": float(np.linalg.norm(Hw_orig)),
        "head_bias_before": Hb_orig.tolist(),
        "head_weight_norm_after":  float(np.linalg.norm(Hw_new)),
        "head_bias_after":         Hb_new.tolist(),
        "feedback_accuracy_before": acc_orig,
        "feedback_accuracy_after":  acc_new,
        "feedback_n": N,
    }

    # Return either w alone or a dict with diagnostics – we follow the spec
    # that permits returning {"weights": w, "diagnostics": diag}.
    return {"weights": w, "diagnostics": diag}
