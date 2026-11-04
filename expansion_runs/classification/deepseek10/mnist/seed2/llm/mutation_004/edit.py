import numpy as np

def edit(w, f):
    # Read full feedback pool
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)  # [2000, 784]
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10

    # Center the input
    mean_x = X.mean(axis=0, keepdims=True)
    Xc = X - mean_x

    # PCA via SVD on centered data
    # Xc = U S Vt, Vt is [min(N,D), D]
    # We need 128 components.
    # Use full_matrices=False for efficiency.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = 128
    V = Vt[:k, :]  # [128, 784]

    # Set fc1 to PCA directions and center bias
    w["fc1.weight"] = V.astype(np.float32)
    w["fc1.bias"] = (-V @ mean_x.T).ravel().astype(np.float32)

    # Compute new hidden features
    Z = X @ V.T + w["fc1.bias"].astype(np.float64)  # [2000, 128]
    H = np.maximum(Z, 0.0)  # ReLU

    # Fit fc2 by regularized LDA on H
    # Class means
    means = np.zeros((C, k), dtype=np.float64)
    for c in range(C):
        means[c] = H[y == c].mean(axis=0)

    # Pooled within-class covariance
    Sw = np.zeros((k, k), dtype=np.float64)
    for c in range(C):
        Hc = H[y == c] - means[c]
        Sw += Hc.T @ Hc

    # Regularization
    ridge = 1e-3
    Sw_reg = Sw + ridge * np.eye(k)

    # Solve for LDA directions: W = Sw^{-1} (means - global_mean)
    global_mean = H.mean(axis=0)
    M = means - global_mean  # [10, 128]
    # Solve Sw_reg @ W.T = M.T  => W.T = solve(Sw_reg, M.T)
    W = np.linalg.solve(Sw_reg, M.T).T  # [10, 128]

    # Bias: w_c = W_c, b_c = - W_c @ global_mean
    b = -W @ global_mean

    w["fc2.weight"] = W.astype(np.float32)
    w["fc2.bias"] = b.astype(np.float32)

    # Diagnostics
    diag = {
        "pca_singular_values_top10": S[:10].tolist(),
        "fc1_weight_norm": float(np.linalg.norm(w["fc1.weight"])),
        "fc1_bias_norm": float(np.linalg.norm(w["fc1.bias"])),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "hidden_mean": float(H.mean()),
        "hidden_std": float(H.std()),
        "hidden_zero_fraction": float((H == 0).mean()),
    }

    return {"weights": w, "diagnostics": diag}
