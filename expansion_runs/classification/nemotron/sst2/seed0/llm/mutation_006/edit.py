def edit(w, f):
    """
    Edit the neural‑network weights using a single closed‑form ridge‑regression step
    on the provided feedback pool. Only the final `head` layer is modified.
    Returns the modified weight dictionary and a small diagnostics dict.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # 1. Extract feedback arrays (shapes are guaranteed by the prompt)
    # ------------------------------------------------------------------
    X = f["features"]                # (4000, 32) float32
    y = f["y"].astype(np.int64)      # (4000,) int64

    # ------------------------------------------------------------------
    # 2. Pre‑edit statistics (use the already‑supplied logits / probs)
    # ------------------------------------------------------------------
    pre_logits = f["logits"]         # (4000, 2) float32
    pre_acc = float((np.argmax(pre_logits, axis=1) == y).mean())

    # Keep original head parameters for diagnostics
    orig_weight = w["head.weight"].copy()          # (2, 32) float32
    orig_bias   = w["head.bias"].copy()            # (2,) float32
    orig_w_norm = float(np.linalg.norm(orig_weight))
    orig_b_norm = float(np.linalg.norm(orig_bias))

    # ------------------------------------------------------------------
    # 3. Closed‑form ridge regression:  w = (XᵀX + λI)⁻¹ Xᵀ y_onehot
    # ------------------------------------------------------------------
    lam = 1e-3                                 # small ridge coefficient
    XTX = X.T @ X                              # (32, 32)
    XTX_reg = XTX + lam * np.eye(32, dtype=np.float32)

    # One‑hot encode the binary labels
    Y_onehot = np.zeros((4000, 2), dtype=np.float32)
    Y_onehot[np.arange(4000), y] = 1.0

    XT_Y = X.T @ Y_onehot                      # (32, 2)
    W_raw = np.linalg.solve(XTX_reg, XT_Y)    # (32, 2)

    # Reshape to match the original head weight layout (2, 32)
    head_weight = (W_raw.T).astype(np.float32)

    # Bias: average residual per class
    residual = Y_onehot - X @ W_raw            # (4000, 2)
    head_bias = (np.mean(residual, axis=0)).astype(np.float32)   # (2,)

    # ------------------------------------------------------------------
    # 4. Apply the new head parameters
    # ------------------------------------------------------------------
    w["head.weight"] = head_weight
    w["head.bias"]   = head_bias

    # ------------------------------------------------------------------
    # 5. Post‑edit diagnostics
    # ------------------------------------------------------------------
    new_w_norm = float(np.linalg.norm(head_weight))
    new_b_norm = float(np.linalg.norm(head_bias))

    diagnostics = {
        "pre_accuracy": pre_acc,
        "head_weight_norm_before": orig_w_norm,
        "head_weight_norm_after": new_w_norm,
        "head_bias_norm_before": orig_b_norm,
        "head_bias_norm_after": new_b_norm,
        "ridge_lambda": float(lam)
    }

    return {"weights": w, "diagnostics": diagnostics}
