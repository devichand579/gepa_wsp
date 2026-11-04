import numpy as np

def edit(w, f):
    """
    Closed‑form ridge‑regression of fc2 on the full feedback pool.
    fc1 weights are kept fixed; only fc2.weight and fc2.bias are edited.
    Returns the modified weight dictionary and a small diagnostics dict.
    """
    # ------- data from feedback pool -------
    features = f["features"]                 # shape [2000, 128]
    y        = f["y"]                        # shape [2000], int64 0‑9

    # one‑hot labels
    Y = np.eye(10, dtype=np.float32)[y]    # [2000, 10]

    # ridge parameter (small value; can be tuned later)
    lam = 0.1

    N, D = features.shape                  # N=2000, D=128

    # ------- normal equations -------
    A = features.T @ features + lam * np.eye(D, dtype=np.float32)   # [128,128]
    B = features.T @ Y                                                    # [128,10]

    # solve A @ Wt = B  for Wt = (fc2.weight)^T  (shape [128,10])
    Wt = np.linalg.solve(A, B)                                            # [128,10]

    # fc2.weight has shape [10,128]; transpose back
    new_fc2_weight = Wt.T.astype(np.float32)                             # [10,128]

    # ------- bias -------
    # compute logits with current bias then replace bias with mean residual
    logits = features @ new_fc2_weight.T                                 # [2000,10]
    residual = Y - logits                                                 # [2000,10]
    new_fc2_bias = np.mean(residual, axis=0).astype(np.float32)          # [10]

    # ------- update w -------
    w = dict(w)  # shallow copy to avoid mutating the original if desired
    w["fc2.weight"] = new_fc2_weight
    w["fc2.bias"]   = new_fc2_bias

    # ------- diagnostics -------
    # predictions on the feedback pool
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs      = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pred       = np.argmax(probs, axis=1)
    accuracy   = float(np.mean(pred == y))

    diag = {
        "train_accuracy": accuracy,
        "fc2_weight_mean": float(new_fc2_weight.mean()),
        "fc2_weight_std":  float(new_fc2_weight.std()),
        "fc2_bias_mean":   float(new_fc2_bias.mean()),
        "fc2_bias_std":    float(new_fc2_bias.std()),
        "ridge_lambda":    float(lam)
    }

    # return the edited weights and diagnostics
    return {"weights": w, "diagnostics": diag}
