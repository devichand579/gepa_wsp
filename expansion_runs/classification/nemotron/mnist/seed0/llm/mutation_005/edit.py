def edit(w, f):
    """
    Edit the network weights using a closed‑form ridge regression on the
    full feedback pool.  Only NumPy linear‑algebra is used; no autograd,
    gradients, or iterative optimisers are involved.
    """
    # ----- data ---------------------------------------------------------
    X = f["features"]                     # (2000, 128) post‑fc1 activations
    y = f["y"]                            # (2000,) integer labels

    # one‑hot encode targets
    Y = np.eye(10, dtype=np.float32)[y]   # (2000, 10)

    # ridge regularisation strength (chosen small enough to keep weights
    # well‑behaved but large enough to avoid singularities)
    lam = np.float32(0.1)

    # ----- ridge regression:  X @ W^T ≈ Y  ------------------------------
    # We solve (X^T X + λ I) W^T = X^T Y  for W^T, then transpose.
    A = X.T @ X + lam * np.eye(128, dtype=np.float32)   # (128, 128)
    B = X.T @ Y                                        # (128, 10)
    # np.linalg.solve returns a matrix Z such that A @ Z = B
    Wt = np.linalg.solve(A, B)                         # (128, 10)
    new_fc2_weight = Wt.T.astype(np.float32)           # (10, 128)

    # ----- bias ---------------------------------------------------------
    # bias = mean( Y - X @ W^T ), averaged over the pool
    logits = X @ new_fc2_weight.T          # (2000, 10)
    new_fc2_bias = np.mean(Y - logits, axis=0).astype(np.float32)   # (10,)

    # ----- update the weight dictionary ---------------------------------
    w["fc2.weight"] = new_fc2_weight
    w["fc2.bias"]   = new_fc2_bias

    # ----- diagnostics --------------------------------------------------
    # compute accuracy on the full feedback pool for reporting
    preds = np.argmax(logits, axis=1)
    full_acc = float(np.mean(preds == y))

    diag = {
        "ridge_lambda": float(lam),
        "fc2_weight_norm": float(np.linalg.norm(new_fc2_weight)),
        "fc2_bias_norm":   float(np.linalg.norm(new_fc2_bias)),
        "full_pool_accuracy": full_acc,
        "weight_mean": float(new_fc2_weight.mean()),
        "bias_mean":   float(new_fc2_bias.mean()),
    }

    return {"weights": w, "diagnostics": diag}
