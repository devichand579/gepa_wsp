import numpy as np

def edit(w, f):
    """
    Edit the neural‑network weights using a single closed‑form ridge‑regression step
    on the head (linear) layer.  All other weights are left untouched.

    Parameters
    ----------
    w : dict
        Current full‑precision parameter dictionary (mutable).
    f : dict
        Feedback arrays as described in the prompt.
        Required keys:  f["x"], f["y"], f["features"], f["logits"], f["probs"].

    Returns
    -------
    {"weights": w, "diagnostics": diag}
        * w – the modified weight dictionary (head.weight and head.bias updated).
        * diag – a small JSON‑compatible statistics dict.
    """
    # ------------------------------------------------------------------
    # 1.  Pull the relevant arrays (all are float32 unless noted)
    # ------------------------------------------------------------------
    X          = f["x"]                     # [2000, 1, 28, 28]   – not needed directly
    y          = f["y"]                     # [2000] int64
    features   = f["features"]              # [2000, 1568]  – head input (flattened conv2 output)
    logits     = f["logits"]                # [2000, 10] – current logits
    probs      = f["probs"]                 # [2000, 10] – current probabilities

    # ------------------------------------------------------------------
    # 2.  Current baseline accuracy (for diagnostics)
    # ------------------------------------------------------------------
    baseline_acc = float(np.mean(np.argmax(probs, axis=1) == y))

    # ------------------------------------------------------------------
    # 3.  Build one‑hot target matrix Y  [2000, 10]
    # ------------------------------------------------------------------
    n_samples, n_classes = 2000, 10
    Y = np.eye(n_classes, dtype=np.float32)[y]   # [2000,10]  1 at correct class

    # ------------------------------------------------------------------
    # 4.  Ridge‑regression solution for head.weight  [10, 1568]
    # ------------------------------------------------------------------
    lam = np.float32(1e-3)                # small ridge coefficient
    # features.T @ features  -> 1568 × 1568
    Gram = features.T @ features           # shape (1568, 1568), float32
    # Add ridge term
    A = Gram + lam * np.eye(1568, dtype=np.float32)

    # features.T @ Y  -> 1568 × 10
    B = features.T @ Y                     # shape (1568, 10), float32

    # Solve A * W_raw = B  (W_raw will be 1568 × 10)
    W_raw = np.linalg.solve(A, B)          # numerically stable, float32

    # Transpose to PyTorch linear convention [out, in] = [10, 1568]
    head_weight_new = W_raw.T.astype(np.float32)   # shape (10, 1568)

    # ------------------------------------------------------------------
    # 5.  Compute new head.bias  [10]
    # ------------------------------------------------------------------
    # bias = mean( Y - features @ weight.T , axis=0 )
    logits_new = features @ head_weight_new.T   # [2000, 10]
    bias_new   = (Y - logits_new).mean(axis=0).astype(np.float32)   # [10]

    # ------------------------------------------------------------------
    # 6.  Apply the updates to the weight dict (preserve other entries)
    # ------------------------------------------------------------------
    w["head.weight"] = head_weight_new
    w["head.bias"]   = bias_new

    # ------------------------------------------------------------------
    # 7.  Compute post‑edit probabilities and accuracy (diagnostics)
    # ------------------------------------------------------------------
    # numerically stable softmax
    logits_edit = features @ head_weight_new.T + bias_new   # [2000,10]
    max_logits = logits_edit.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_edit - max_logits)
    probs_edit = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    new_acc = float(np.mean(np.argmax(probs_edit, axis=1) == y))

    # ------------------------------------------------------------------
    # 8.  Return modified weights + diagnostics dict
    # ------------------------------------------------------------------
    diag = {
        "lambda": float(lam),
        "baseline_accuracy": baseline_acc,
        "new_accuracy": new_acc,
        "accuracy_gain": new_acc - baseline_acc,
        "head_weight_mean": float(head_weight_new.mean()),
        "head_bias_mean": float(bias_new.mean()),
    }
    return {"weights": w, "diagnostics": diag}
