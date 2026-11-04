import numpy as np

def edit(w, f):
    # Extract features and labels from feedback
    X = f["features"].astype(np.float64)  # (N, D)
    y = f["y"].astype(np.int64)           # (N,)
    N, D = X.shape
    C = 10

    # Current head parameters
    W = w["head.weight"].astype(np.float64).copy()
    b = w["head.bias"].astype(np.float64).copy()

    diag = {}
    initial_mis = []
    final_mis = []
    updates = []

    # Sequentially fix misclassified samples with closed-form rank-1 updates
    for it in range(10):
        L = X @ W.T + b
        pred = np.argmax(L, axis=1)
        mis = np.where(pred != y)[0]
        if it == 0:
            initial_mis = mis.tolist()
        if len(mis) == 0:
            break

        i = mis[0]
        c = y[i]
        k = pred[i]
        xi = X[i]
        norm2 = np.dot(xi, xi)
        if norm2 < 1e-12:
            break

        margin = L[i, c] - L[i, k]
        alpha_req = (-margin) / (2.0 * norm2) + 1e-6

        # Compute maximum alpha allowed by all other correct samples
        s = X @ xi  # dot products with the update direction
        alpha_max = np.inf
        for j in range(N):
            if j == i:
                continue
            t_j = y[j]
            s_j = s[j]
            if s_j == 0.0:
                continue
            coeff_t = (1 if t_j == c else 0) - (1 if t_j == k else 0)
            for o in range(C):
                if o == t_j:
                    continue
                coeff_o = (1 if o == c else 0) - (1 if o == k else 0)
                A = s_j * (coeff_t - coeff_o)
                if A < 0.0:
                    M = L[j, t_j] - L[j, o]
                    if M <= 0.0:
                        # j is already misclassified; ignore for this update
                        continue
                    bound = M / (-A)
                    if bound < alpha_max:
                        alpha_max = bound

        if alpha_max <= alpha_req:
            # Cannot fix without flipping a correct sample; try a minimal step
            alpha = alpha_req * 1.01
        else:
            alpha = min(alpha_req * 1.1, alpha_max * 0.9)

        # Apply rank-1 update to head weights
        W[c] += alpha * xi
        W[k] -= alpha * xi
        updates.append({
            "index": int(i),
            "true": int(c),
            "pred": int(k),
            "alpha_req": float(alpha_req),
            "alpha_max": float(alpha_max) if np.isfinite(alpha_max) else None,
            "alpha_used": float(alpha)
        })

    # Final check
    L = X @ W.T + b
    pred = np.argmax(L, axis=1)
    final_mis = np.where(pred != y)[0].tolist()

    # Clip to allowed range
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    # Write back
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    diag["initial_misclassified"] = initial_mis
    diag["final_misclassified"] = final_mis
    diag["updates"] = updates
    diag["final_accuracy"] = float(np.mean(pred == y))
    diag["max_abs_head_weight"] = float(np.max(np.abs(W)))
    diag["max_abs_head_bias"] = float(np.max(np.abs(b)))

    return {"weights": w, "diagnostics": diag}
