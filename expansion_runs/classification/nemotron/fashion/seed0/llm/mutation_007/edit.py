import numpy as np

def edit(w, f):
    """
    Closed‑form ridge‑regression training of the network head on the
    full permitted feedback pool.  All other modules (conv1, conv2) are left unchanged.
    
    Parameters
    ----------
    w : dict
        Current full parameter dictionary (float32).  Keys include
        "head.weight" shape (10,1568) and "head.bias" shape (10,).
    f : dict
        Feedback dictionaries as described in the prompt.  Must contain
        "features" (N,1568), "y" (N,), and optionally "logits","probs".
    
    Returns
    -------
    {"weights": w, "diagnostics": diag}
        w with updated head weights/bias; diag contains useful statistics.
    """
    # ------------------------------------------------------------------
    # 1. Pull current head parameters and feedback data
    # ------------------------------------------------------------------
    head_weight = w["head.weight"].astype(np.float32)      # (10, 1568)
    head_bias   = w["head.bias"].astype(np.float32)       # (10,)

    features = f["features"].astype(np.float32)           # (N, 1568), N = 2000
    y        = f["y"].astype(np.int64)                    # (N,)

    N, D = features.shape                                 # D = 1568

    # ------------------------------------------------------------------
    # 2. Ridge‑regression (closed‑form) for each output class
    # ------------------------------------------------------------------
    lam = 1e-3                                            # small L2 penalty
    Gram = features.T @ features + lam * np.eye(D, dtype=np.float32)   # (D,D)

    new_weight = np.copy(head_weight)
    new_bias   = np.copy(head_bias)

    for c in range(10):
        # binary target: 1 if sample belongs to class c, else 0
        y_c = (y == c).astype(np.float32)               # (N,)

        # solve (Gram) w_c = features.T @ y_c
        w_c = np.linalg.solve(Gram, features.T @ y_c)   # (D,)

        # bias = mean( y_c - X w_c )
        b_c = float(np.mean(y_c - features @ w_c))

        new_weight[c, :] = w_c
        new_bias[c]      = b_c

    # ------------------------------------------------------------------
    # 3. Write updated head back into the weight dict (shapes unchanged)
    # ------------------------------------------------------------------
    w["head.weight"] = new_weight
    w["head.bias"]   = new_bias

    # ------------------------------------------------------------------
    # 4. Diagnostics: compute new logits, probabilities, accuracy & CE
    # ------------------------------------------------------------------
    logits_new = features @ new_weight.T + new_bias[np.newaxis, :]   # (N,10)

    # numerically stable softmax
    max_logit = logits_new.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_new - max_logit)
    probs_new = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    acc_new = float(np.mean(np.argmax(probs_new, axis=1) == y))
    ce_new  = -float(np.mean(np.log(probs_new[np.arange(N), y])))

    diag = {
        "head_weight_mean":    float(np.mean(new_weight)),
        "head_weight_std":     float(np.std(new_weight)),
        "head_bias_mean":      float(np.mean(new_bias)),
        "feedback_accuracy":   acc_new,
        "feedback_cross_entropy": ce_new,
        "ridge_lambda":        lam
    }

    return {"weights": w, "diagnostics": diag}
