# --------------------------------------------------------------
# edit(w, f) – closed‑form head re‑fit using the full feedback pool
# --------------------------------------------------------------
def edit(w, f):
    """
    w : dict containing every current parameter (float32).  Keys
        match the schema supplied in the prompt (e.g. "head.weight",
        "head.bias").
    f : dict of read‑only NumPy arrays with the exact keys/shapes
        listed in the prompt.  In particular:
        - f["features"]  shape (4000, 32)  float32
        - f["y"]         shape (4000,)     int64   (0 = negative, 1 = positive)
    The function returns {"weights": w_updated, "diagnostics": diag}.
    No autograd, gradients, iterative optimisers or back‑propagation
    are used – only NumPy linear‑algebra and closed‑form fitting.
    """
    import numpy as np

    # -----------------------------------------------------------------
    # 1. Pull the permitted feedback arrays
    # -----------------------------------------------------------------
    X = f["features"].astype(np.float32)          # (4000, 32)
    y = f["y"].astype(np.int64)                   # (4000,)

    # -----------------------------------------------------------------
    # 2. Convert labels to one‑hot targets (2 classes)
    # -----------------------------------------------------------------
    Y = np.eye(2, dtype=np.float32)[y]            # (4000, 2)

    # -----------------------------------------------------------------
    # 3. Closed‑form ridge regression:  W_fit (32 × 2)  such that
    #    logits ≈ X @ W_fit.T + b
    # -----------------------------------------------------------------
    lam = np.float32(1e-3)                        # small ridge regulariser
    XtX = X.T @ X                                 # (32, 32)
    XtY = X.T @ Y                                 # (32, 2)
    I32 = np.eye(32, dtype=np.float32)
    W_fit = np.linalg.solve(XtX + lam * I32, XtY)  # (32, 2)

    # -----------------------------------------------------------------
    # 4. Map the fitted coefficients to the network’s head parameters
    # -----------------------------------------------------------------
    # head.weight is (2, 32) → transpose of W_fit
    new_weight = W_fit.T.astype(np.float32)       # (2, 32)

    # bias: average residual after the linear fit
    resid = Y - X @ W_fit                         # (4000, 2)
    new_bias = np.mean(resid, axis=0).astype(np.float32)  # (2,)

    # -----------------------------------------------------------------
    # 5. Apply the edits (other weights stay exactly as they were)
    # -----------------------------------------------------------------
    w["head.weight"] = new_weight
    w["head.bias"]   = new_bias

    # -----------------------------------------------------------------
    # 6. Diagnostics – how does the new head perform on the feedback pool?
    # -----------------------------------------------------------------
    new_logits = X @ W_fit.T + new_bias           # (4000, 2)
    # numerically stable softmax
    max_logit = new_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(new_logits - max_logit)
    new_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    new_accuracy = (np.argmax(new_probs, axis=1) == y).mean()

    diag = {
        "lambda": float(lam),
        "new_head_weight_fro_norm": float(np.linalg.norm(new_weight)),
        "new_head_bias": new_bias.tolist(),
        "feedback_accuracy": float(new_accuracy),
        "old_logits_mean": float(f["logits"].mean()),   # for reference
    }

    return {"weights": w, "diagnostics": diag}
