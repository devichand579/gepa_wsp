def edit(w, f):
    """
    Closed‑form ridge‑regression fit of the final head layer using the full
    feedback pool.  Only NumPy linear algebra is used; no autograd, gradients,
    or iterative optimisation loops.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # 1. Pull the permitted feedback arrays
    # ------------------------------------------------------------------
    X = f["features"]                     # shape (4000, 32), float32
    y = f["y"].astype(np.float32)         # shape (4000,), int labels 0/1

    N, D = X.shape                        # N=4000, D=32

    # ------------------------------------------------------------------
    # 2. One‑hot encode the binary labels → (N, 2)
    # ------------------------------------------------------------------
    Y = np.eye(2, dtype=np.float32)[y.astype(int)]   # (4000, 2)

    # ------------------------------------------------------------------
    # 3. Center data (helps numerical stability; bias will be recovered later)
    # ------------------------------------------------------------------
    X_mean = np.mean(X, axis=0, keepdims=True)       # (1, 32)
    Xc = X - X_mean                                 # (4000, 32)
    Y_mean = np.mean(Y, axis=0, keepdims=True)      # (1, 2)
    Yc = Y - Y_mean                                 # (4000, 2)

    # ------------------------------------------------------------------
    # 4. Ridge‑regression normal equations (lambda = 1e‑3)
    # ------------------------------------------------------------------
    lam = 1e-3
    Gram = Xc.T @ Xc + lam * np.eye(D, dtype=np.float32)   # (32, 32)
    XtY = Xc.T @ Yc                                    # (32, 2)

    # Solve for the centred weight matrix (32, 2) – column j is the weight
    # vector for class j when the features are centred.
    weight_centered = np.linalg.solve(Gram, XtY)       # (32, 2)

    # Map to the network's expected shape (2, 32) where row i belongs to class i.
    new_weight = weight_centered.T.astype(np.float32)   # (2, 32)

    # ------------------------------------------------------------------
    # 5. Bias term: recover from centred solution
    # ------------------------------------------------------------------
    # pred = X @ new_weight.T  gives un‑centred logits; bias closes the gap to Y.
    bias = np.mean(Y - X @ new_weight.T, axis=0).astype(np.float32)   # (2,)

    # ------------------------------------------------------------------
    # 6. Apply the new parameters to the weight dictionary
    # ------------------------------------------------------------------
    # shallow‑copy the dict so we do not mutate the caller's reference
    w_new = {k: v.copy() for k, v in w.items()}
    w_new["head.weight"] = new_weight
    w_new["head.bias"] = bias

    # ------------------------------------------------------------------
    # 7. Diagnostics (accuracy, cross‑entropy, weight norms, etc.)
    # ------------------------------------------------------------------
    logits_new = X @ new_weight.T + bias          # (4000, 2)
    # numerically stable softmax
    exp_logits = np.exp(logits_new - np.max(logits_new, axis=1, keepdims=True))
    probs_new = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    acc = float(np.mean(np.argmax(probs_new, axis=1) == y.astype(int)))
    # cross‑entropy (using the predicted probs for the true class)
    ce = -float(np.mean(np.log(probs_new[np.arange(N), y.astype(int)])))

    diag = {
        "accuracy": acc,
        "cross_entropy": ce,
        "head_weight_norm": float(np.linalg.norm(new_weight)),
        "head_bias": list(bias),
        "ridge_lambda": lam
    }

    # ------------------------------------------------------------------
    # 8. Return the modified weights together with a tiny diagnostics dict
    # ------------------------------------------------------------------
    return {"weights": w_new, "diagnostics": diag}
