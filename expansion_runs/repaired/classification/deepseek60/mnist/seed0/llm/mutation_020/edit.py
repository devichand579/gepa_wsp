import numpy as np

def edit(w, f):
    # Read feedback pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # [N, 784]
    y = f["y"].astype(np.int64)                                  # [N]
    N, D = X.shape
    C = 10

    # Center globally
    mu = X.mean(axis=0, keepdims=True)          # [1, D]
    Xc = X - mu

    # Class means
    mus = np.zeros((C, D), dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        counts[c] = idx.sum()
        if counts[c] > 0:
            mus[c] = X[idx].mean(axis=0)

    # Scatter matrices
    Sw = np.zeros((D, D), dtype=np.float64)
    Sb = np.zeros((D, D), dtype=np.float64)
    for c in range(C):
        idx = (y == c)
        if counts[c] == 0:
            continue
        Xc_c = X[idx] - mus[c]
        Sw += Xc_c.T @ Xc_c
        diff = (mus[c] - mu.ravel()).reshape(-1, 1)
        Sb += counts[c] * (diff @ diff.T)

    # Regularize Sw for numerical stability
    Sw_reg = Sw + 1e-3 * np.eye(D)

    # Solve generalized eigenproblem for LDA: Sb v = lambda Sw v
    # Use Cholesky of Sw_reg: Sw_reg = L L^T
    # Then solve standard eigenproblem for L^{-1} Sb L^{-T}
    try:
        L = np.linalg.cholesky(Sw_reg)
        Linv = np.linalg.inv(L)
        M = Linv @ Sb @ Linv.T
        M = (M + M.T) * 0.5
        evals, evecs = np.linalg.eigh(M)
        # Sort descending
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        # Transform back to original space
        V_lda = Linv.T @ evecs
        # Keep only top 9 (rank of Sb <= C-1)
        V_lda = V_lda[:, :min(9, V_lda.shape[1])]
    except np.linalg.LinAlgError:
        V_lda = np.zeros((D, 0), dtype=np.float64)

    # PCA directions on centered data for remaining dimensions
    # Compute covariance of Xc
    Cov = (Xc.T @ Xc) / max(N - 1, 1)
    Cov = (Cov + Cov.T) * 0.5
    evals_pca, evecs_pca = np.linalg.eigh(Cov)
    order_pca = np.argsort(evals_pca)[::-1]
    evecs_pca = evecs_pca[:, order_pca]

    # Build orthogonal basis: start with LDA directions, then add PCA directions
    # that are orthogonal to already selected ones.
    basis = []
    # Normalize LDA directions
    for j in range(V_lda.shape[1]):
        v = V_lda[:, j].copy()
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            v /= nv
            basis.append(v)

    # Add PCA directions orthogonal to current basis
    for j in range(evecs_pca.shape[1]):
        if len(basis) >= 128:
            break
        v = evecs_pca[:, j].copy()
        # Orthogonalize against existing basis
        for b in basis:
            v -= (b @ v) * b
        nv = np.linalg.norm(v)
        if nv > 1e-8:
            v /= nv
            basis.append(v)

    # If still not enough, fill with random orthonormal vectors
    rng = np.random.default_rng(12345)
    while len(basis) < 128:
        v = rng.standard_normal(D)
        for b in basis:
            v -= (b @ v) * b
        nv = np.linalg.norm(v)
        if nv > 1e-8:
            v /= nv
            basis.append(v)

    V = np.stack(basis[:128], axis=0)  # [128, D]

    # Set fc1 weights and bias
    # We want hidden pre-activation = X @ V.T + b1.
    # Center the data: use Xc = X - mu. So set bias = -mu @ V.T.
    fc1_w = V.astype(np.float32)                     # [128, 784]
    fc1_b = (-mu.ravel() @ V.T).astype(np.float32)   # [128]

    # Compute hidden features on full pool
    H_pre = X @ fc1_w.T + fc1_b                      # [N, 128]
    H = np.maximum(H_pre, 0.0)                       # ReLU

    # Fit fc2 via ridge regression to one-hot targets
    # We want H @ W2.T + b2 = Y_onehot
    # Solve for W2, b2 by augmenting H with a column of ones.
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Augment H with bias column
    Ha = np.concatenate([H, np.ones((N, 1), dtype=np.float64)], axis=1)  # [N, 129]

    # Ridge parameter: choose a small value; we can use cross-validation on the pool
    # to select lambda, but to keep it simple and avoid search loops, use a fixed
    # value that is small relative to the scale of Ha^T Ha.
    # Compute a reasonable lambda from the trace.
    HtH = Ha.T @ Ha
    trace = np.trace(HtH)
    lam = 1e-4 * trace / Ha.shape[1] if trace > 0 else 1e-6
    # Ensure lam is not too small
    lam = max(lam, 1e-6)

    # Solve (Ha^T Ha + lam I) W = Ha^T Y
    A = HtH + lam * np.eye(Ha.shape[1])
    B = Ha.T @ Y
    W = np.linalg.solve(A, B)  # [129, C]

    fc2_w = W[:-1, :].T.astype(np.float32)  # [10, 128]
    fc2_b = W[-1, :].astype(np.float32)     # [10]

    # Write back to w
    w["fc1.weight"] = fc1_w
    w["fc1.bias"] = fc1_b
    w["fc2.weight"] = fc2_w
    w["fc2.bias"] = fc2_b

    # Diagnostics
    # Evaluate on the feedback pool
    logits = H @ fc2_w.T + fc2_b
    preds = np.argmax(logits, axis=1)
    acc = float((preds == y).mean())
    # Cross-entropy
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = float(-np.log(probs[np.arange(N), y] + 1e-12).mean())

    diagnostics = {
        "feedback_accuracy": acc,
        "feedback_cross_entropy": ce,
        "fc1_weight_norm": float(np.linalg.norm(fc1_w)),
        "fc1_bias_norm": float(np.linalg.norm(fc1_b)),
        "fc2_weight_norm": float(np.linalg.norm(fc2_w)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_b)),
        "ridge_lambda": float(lam),
        "num_lda_directions": int(V_lda.shape[1]),
        "hidden_mean": float(H.mean()),
        "hidden_std": float(H.std()),
        "hidden_zero_fraction": float((H == 0).mean()),
    }

    return {"weights": w, "diagnostics": diagnostics}
