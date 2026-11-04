import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"].astype(np.float64)          # (2000,1,28,28)
    y = f["y"].astype(np.int64)            # (2000,)
    N = x.shape[0]
    x_flat = x.reshape(N, -1)              # (2000,784)

    n_classes = 10
    D = 128

    def fit_lda(H, y, ridge=1e-2):
        """Closed-form ridge-regularized LDA.
        H: (N, D) hidden features
        y: (N,) labels
        Returns W (10,D), b (10,)
        """
        N, D = H.shape
        means = np.zeros((n_classes, D), dtype=np.float64)
        counts = np.zeros(n_classes, dtype=np.float64)
        for c in range(n_classes):
            idx = (y == c)
            counts[c] = idx.sum()
            if counts[c] > 0:
                means[c] = H[idx].mean(axis=0)

        # Within-class scatter
        Sw = np.zeros((D, D), dtype=np.float64)
        for c in range(n_classes):
            idx = (y == c)
            if counts[c] > 0:
                diff = H[idx] - means[c]
                Sw += diff.T @ diff

        # Adaptive ridge: scale by average diagonal of Sw
        avg_diag = np.trace(Sw) / D
        Sw_reg = Sw + ridge * avg_diag * np.eye(D) + 1e-8 * np.eye(D)

        # Solve Sw_reg @ W.T = means.T
        W_T = np.linalg.solve(Sw_reg, means.T)   # (D,10)
        W = W_T.T                                # (10,D)
        b = -0.5 * np.sum(means * W, axis=1) + np.log(counts / N + 1e-12)
        return W, b

    def accuracy(H, y, W, b):
        logits = H @ W.T + b
        pred = np.argmax(logits, axis=1)
        return (pred == y).mean()

    # --- Candidate A: current features + LDA ---
    H_cur = f["features"].astype(np.float64)  # (2000,128)
    W_A, b_A = fit_lda(H_cur, y, ridge=1e-2)
    acc_A = accuracy(H_cur, y, W_A, b_A)

    # --- Candidate B: PCA features + LDA ---
    mean_x = x_flat.mean(axis=0)
    xc = x_flat - mean_x
    # SVD for PCA
    U, S, Vt = np.linalg.svd(xc, full_matrices=False)
    comps = Vt[:D]                            # (128,784)
    # Hidden = relu((x - mean) @ comps.T)
    hidden_pca = xc @ comps.T                 # (2000,128)
    H_pca = np.maximum(hidden_pca, 0.0)
    W_B, b_B = fit_lda(H_pca, y, ridge=1e-2)
    acc_B = accuracy(H_pca, y, W_B, b_B)

    # Choose better candidate
    if acc_B > acc_A:
        # Set fc1 to PCA components
        w["fc1.weight"] = comps.astype(np.float32)
        w["fc1.bias"] = (-mean_x @ comps.T).astype(np.float32)
        w["fc2.weight"] = W_B.astype(np.float32)
        w["fc2.bias"] = b_B.astype(np.float32)
        chosen = "pca_lda"
        acc_chosen = acc_B
    else:
        # Keep current fc1, only update fc2
        w["fc2.weight"] = W_A.astype(np.float32)
        w["fc2.bias"] = b_A.astype(np.float32)
        chosen = "current_lda"
        acc_chosen = acc_A

    # Clip to safe range
    for k in w:
        if isinstance(w[k], np.ndarray):
            w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "chosen": chosen,
        "acc_current_lda": float(acc_A),
        "acc_pca_lda": float(acc_B),
        "acc_chosen": float(acc_chosen),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
