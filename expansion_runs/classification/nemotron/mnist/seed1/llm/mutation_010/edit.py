def edit(w, f):
    """
    Closed‑form ridge‑regression training of fc2 using the full 2000‑example
    feedback pool.  fc1 weights are left untouched; features are recomputed
    from the original images so they are always consistent with the current
    fc1 parameters.
    """
    # --------------------------------------------------------------
    # 1. Flatten the input images [2000, 1, 28, 28] -> [2000, 784]
    # --------------------------------------------------------------
    x = f["x"]                                 # float32 [2000,1,28,28]
    x_flat = x.reshape(-1, 784).astype(np.float32)

    # --------------------------------------------------------------
    # 2. Compute hidden activations with the *current* fc1 (kept fixed)
    # --------------------------------------------------------------
    fc1_w = w["fc1.weight"].astype(np.float32)    # [128, 784]
    fc1_b = w["fc1.bias"].astype(np.float32)      # [128]
    h = np.relu(x_flat @ fc1_w.T + fc1_b)        # [2000, 128]

    # --------------------------------------------------------------
    # 3. Build one‑hot targets from the provided labels
    # --------------------------------------------------------------
    y = f["y"].astype(np.int64)                   # [2000]
    N = y.shape[0]
    Y = np.eye(10, dtype=np.float32)[y]          # [2000, 10]

    # --------------------------------------------------------------
    # 4. Ridge‑regression solution for fc2 weight‑transpose (W^T)
    #    (h^T h + λ I) W^T = h^T Y
    # --------------------------------------------------------------
    lam = 0.1                                     # small ridge parameter
    Gram = h.T @ h + lam * np.eye(128, dtype=np.float32)   # [128,128]
    rhs  = h.T @ Y                                 # [128,10]
    Wt   = np.linalg.solve(Gram, rhs)             # [128,10]   (W^T)

    # --------------------------------------------------------------
    # 5. Recover fc2 weight matrix [10,128] and compute bias
    # --------------------------------------------------------------
    fc2_new_weight = Wt.T.astype(np.float32)       # [10,128]

    # bias = mean( Y - h @ W^T ), i.e. mean residual per class
    res = Y - h @ Wt                               # [2000,10]
    fc2_new_bias = np.mean(res, axis=0).astype(np.float32)   # [10]

    # --------------------------------------------------------------
    # 6. Update the weight dictionary in‑place
    # --------------------------------------------------------------
    w["fc2.weight"] = fc2_new_weight
    w["fc2.bias"]   = fc2_new_bias

    # --------------------------------------------------------------
    # 7. Return the (possibly) edited weights plus a tiny diagnostics block
    # --------------------------------------------------------------
    w_norm_before = np.linalg.norm(w["fc2.weight"] - fc2_new_weight)  # not really needed
    diag = {
        "fc2_weight_norm_after": float(np.linalg.norm(fc2_new_weight)),
        "fc2_bias_before":   w["fc2.bias"].copy(),  # keep original for reference (already overwritten)
        "fc2_bias_after":    fc2_new_bias.tolist(),
        "training_accuracy_approx": None   # placeholder – could be computed if desired
    }
    return {"weights": w, "diagnostics": diag}
