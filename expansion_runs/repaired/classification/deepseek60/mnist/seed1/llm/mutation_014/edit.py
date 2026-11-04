import numpy as np

def edit(w, f):
    # Extract full feedback pool
    X = f["x"].reshape(2000, -1).astype(np.float64)  # [2000, 784]
    y = f["y"].astype(int)                           # [2000]
    N, D = X.shape
    C = 10

    # Global and class means
    mu = X.mean(axis=0)
    mu_c = np.zeros((C, D))
    for c in range(C):
        mu_c[c] = X[y == c].mean(axis=0)

    # Within-class scatter matrix (regularized)
    Sw = np.zeros((D, D))
    for c in range(C):
        Xc = X[y == c] - mu_c[c]
        Sw += Xc.T @ Xc
    Sw /= N
    lambda1 = 1e-3
    Sw_reg = Sw + lambda1 * np.eye(D)

    # LDA directions: solve Sw_reg * v = (mu_c - mu)
    V = np.zeros((C, D))
    for c in range(C):
        V[c] = np.linalg.solve(Sw_reg, mu_c[c] - mu)

    # Orthonormalize LDA directions (rank <= 9)
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    L = Vt[:9]  # 9 x D

    # Total covariance
    Xc = X - mu
    C_total = Xc.T @ Xc / N

    # Project total covariance onto orthogonal complement of L
    P = np.eye(D) - L.T @ L
    C_proj = P @ C_total @ P

    # Top 119 eigenvectors of projected covariance
    eigvals, eigvecs = np.linalg.eigh(C_proj)
    idx = np.argsort(eigvals)[::-1]
    E = eigvecs[:, idx][:, :119].T  # 119 x D

    # Combine to 128 directions
    W1 = np.vstack([L, E])  # 128 x 784

    # Normalize each feature to zero mean, unit variance on training set
    Z = X @ W1.T
    mean_z = Z.mean(axis=0)
    std_z = Z.std(axis=0) + 1e-8
    W1_scaled = W1 / std_z[:, None]
    bias1 = -mean_z / std_z

    # Write fc1
    w["fc1.weight"] = W1_scaled.astype(np.float32)
    w["fc1.bias"] = bias1.astype(np.float32)

    # Compute hidden features with new fc1
    H = np.maximum(0, X @ W1_scaled.T + bias1)  # [2000, 128]

    # Fit fc2 via LDA on hidden features
    mu_h = H.mean(axis=0)
    mu_h_c = np.zeros((C, 128))
    for c in range(C):
        mu_h_c[c] = H[y == c].mean(axis=0)

    Sw_h = np.zeros((128, 128))
    for c in range(C):
        Hc = H[y == c] - mu_h_c[c]
        Sw_h += Hc.T @ Hc
    Sw_h /= N
    lambda2 = 1e-3
    Sw_h_reg = Sw_h + lambda2 * np.eye(128)

    W2 = np.zeros((C, 128))
    for c in range(C):
        W2[c] = np.linalg.solve(Sw_h_reg, mu_h_c[c] - mu_h)
    b2 = -W2 @ mu_h

    # Write fc2
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc1_bias_mean": float(w["fc1.bias"].mean()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "lambda1": lambda1,
        "lambda2": lambda2,
    }
    return {"weights": w, "diagnostics": diagnostics}
