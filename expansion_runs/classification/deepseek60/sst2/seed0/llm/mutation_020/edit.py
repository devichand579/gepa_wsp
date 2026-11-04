import numpy as np

def edit(w, f):
    # ----- 1. Extract training data -----
    x = f["x"]          # (4000, 64) int64
    y = f["y"]          # (4000,) int64
    N, L = x.shape
    V = 2000

    # ----- 2. Build normalised unigram count features -----
    # counts[i, v] = number of times token v appears in example i
    counts = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        counts[i] = np.bincount(x[i], minlength=V)
    # length = number of non-pad tokens (id != 0)
    lengths = (x != 0).sum(axis=1).astype(np.float32)
    lengths = np.maximum(lengths, 1.0)          # avoid division by zero
    X = counts / lengths[:, None]               # normalised counts

    # ----- 3. Closed-form LDA -----
    # class means
    mu0 = X[y == 0].mean(axis=0)
    mu1 = X[y == 1].mean(axis=0)
    # pooled covariance (with small ridge for numerical stability)
    # We can compute the covariance directly from the data.
    # Since V=2000 and N=4000, we can form the centred data matrix.
    Xc = X - X.mean(axis=0, keepdims=True)
    # covariance = Xc.T @ Xc / (N - 1)
    # To avoid forming a 2000x2000 matrix explicitly if memory is tight,
    # we can solve the linear system (Sigma + lam*I) w = mu1 - mu0.
    # Here we form Sigma explicitly; it is 2000x2000 = 4M floats, fine.
    Sigma = (Xc.T @ Xc) / (N - 1)
    lam = 1e-3                                  # ridge regularisation
    Sigma_reg = Sigma + lam * np.eye(V, dtype=np.float32)
    # solve for w
    w_vec = np.linalg.solve(Sigma_reg, (mu1 - mu0).astype(np.float32))
    # intercept: b = -0.5 * (mu1 + mu0) @ w  (classes are balanced)
    b = -0.5 * np.dot(mu1 + mu0, w_vec)

    # ----- 4. Encode into embed and head -----
    # Scale w_vec so that max(|s|) <= 4
    max_abs = np.max(np.abs(w_vec))
    if max_abs < 1e-8:
        C = 1.0
    else:
        C = max_abs / 4.0
    s = w_vec / C                               # per-token scores, |s| <= 4

    # Prepare embed matrix
    embed = np.zeros((V, 32), dtype=np.float32)
    # For each token v, set dim0 = s[v], dim1 = -s[v],
    # dim2 = a, dim3 = -a, where a = sqrt(16 - s[v]^2)
    # This ensures mean 0 and variance 1 across the 32 dims.
    a = np.sqrt(np.maximum(16.0 - s**2, 0.0))
    embed[:, 0] = s
    embed[:, 1] = -s
    embed[:, 2] = a
    embed[:, 3] = -a
    # dims 4..31 remain zero

    # Update weights
    w["embed.weight"] = embed

    # Head: row0 all zeros, row1 = [C, 0, ..., 0]
    head_w = np.zeros((2, 32), dtype=np.float32)
    head_w[1, 0] = C
    w["head.weight"] = head_w

    head_b = np.zeros(2, dtype=np.float32)
    head_b[1] = b
    w["head.bias"] = head_b

    # All other parameters remain as they are (zero/identity).
    # Ensure finite float32 and |value| <= 1000
    for k in w:
        w[k] = np.clip(w[k].astype(np.float32), -1000.0, 1000.0)

    # ----- 5. Diagnostics -----
    diagnostics = {
        "max_abs_w": float(max_abs),
        "C": float(C),
        "b": float(b),
        "embed_nonzero": int(np.count_nonzero(embed)),
        "head_norm": float(np.linalg.norm(head_w)),
    }
    return {"weights": w, "diagnostics": diagnostics}
