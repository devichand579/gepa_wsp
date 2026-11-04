import numpy as np

def edit(w, f):
    # Extract current parameters
    W1 = w["fc1.weight"].astype(np.float64)   # [128, 784]
    b1 = w["fc1.bias"].astype(np.float64)     # [128]
    W2 = w["fc2.weight"].astype(np.float64)   # [10, 128]
    b2 = w["fc2.bias"].astype(np.float64)     # [10]

    # Full feedback data
    H = f["features"].astype(np.float64)      # [2000, 128]  (ReLU outputs)
    y = f["y"].astype(np.int64)               # [2000]
    N = H.shape[0]
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # ---- Step 1: re-optimize temperature on full pool ----
    # Current logits (before any new scaling)
    Z = f["logits"].astype(np.float64)        # [2000, 10]
    # We will scale Z by alpha. Find alpha minimizing CE.
    # CE(alpha) = -mean( sum(Y * log_softmax(alpha * Z)) )
    # Use one Newton step from alpha=1.
    def ce_and_derivs(alpha):
        Zs = alpha * Z
        Zs -= Zs.max(axis=1, keepdims=True)
        expZ = np.exp(Zs)
        P = expZ / expZ.sum(axis=1, keepdims=True)
        logP = Zs - np.log(expZ.sum(axis=1, keepdims=True))
        ce = -np.mean(np.sum(Y * logP, axis=1))
        # derivative w.r.t. alpha
        # dCE/dalpha = -mean( sum(Y * dlogP/dalpha) )
        # dlogP/dalpha = Z - sum(P * Z)
        dlogP = Z - (P * Z).sum(axis=1, keepdims=True)
        dce = -np.mean(np.sum(Y * dlogP, axis=1))
        # second derivative
        # d2logP/dalpha2 = - (sum(P * Z^2) - (sum(P * Z))^2)
        # Actually for softmax: d2logP_i/dalpha2 = - (E[Z^2] - E[Z]^2) for the sample
        # We'll compute numerically for simplicity (still closed-form, no loop)
        # Use a small epsilon to estimate second derivative? No, we can compute analytically.
        # For each sample, let m = sum(P * Z). Then d2logP_i/dalpha2 = - (sum(P * Z^2) - m^2)
        # But we need the Hessian w.r.t. alpha of CE.
        # CE = -mean( sum_i Y_i * (alpha*Z_i - logsumexp(alpha*Z)) )
        # dCE/dalpha = -mean( sum_i Y_i * Z_i - sum_i P_i * Z_i ) = -mean( sum_i (Y_i - P_i) Z_i )
        # d2CE/dalpha2 = mean( sum_i P_i Z_i^2 - (sum_i P_i Z_i)^2 )
        m1 = (P * Z).sum(axis=1)
        m2 = (P * Z * Z).sum(axis=1)
        d2ce = np.mean(m2 - m1 * m1)
        return ce, dce, d2ce

    ce0, dce0, d2ce0 = ce_and_derivs(1.0)
    # Newton step: alpha_new = 1 - dce0 / d2ce0
    if d2ce0 > 1e-12:
        alpha = 1.0 - dce0 / d2ce0
    else:
        alpha = 1.0
    # Clamp alpha to reasonable range
    alpha = float(np.clip(alpha, 0.1, 20.0))

    # Apply temperature scaling to fc2
    W2 *= alpha
    b2 *= alpha

    # ---- Step 2: one regularized Newton step on fc2 using full features ----
    # Compute current logits with new W2, b2
    Z_new = H @ W2.T + b2  # [2000, 10]
    Z_new -= Z_new.max(axis=1, keepdims=True)
    expZ = np.exp(Z_new)
    P = expZ / expZ.sum(axis=1, keepdims=True)

    # Gradient of CE w.r.t. W2 and b2
    # dCE/dZ = (P - Y) / N
    G = (P - Y) / N  # [2000, 10]
    grad_W2 = G.T @ H  # [10, 128]
    grad_b2 = G.sum(axis=0)  # [10]

    # Hessian: we build the full Hessian for parameters [W2 (10x128), b2 (10)]
    # Flatten parameters: first 1280 for W2 (row-major), then 10 for b2.
    # For each sample, Jacobian J_n of logits w.r.t. params is [10, 1290]
    # J_n[:, :1280] = kron(h_n, I_10) ? Actually, logits = W2 @ h + b2.
    # If we flatten W2 row-major, the derivative of logit_c w.r.t. W2_{c', j} is delta_{c,c'} * h_j.
    # So J_n is a 10 x 1290 matrix. We can compute the Hessian as sum_n J_n^T (diag(P_n) - P_n P_n^T) J_n.
    # This is a 1290 x 1290 matrix. We'll compute it efficiently using the fact that
    # the Hessian w.r.t. W2 is (diag(P) - P P^T) ⊗ (h h^T) for each sample.
    # We can compute the full Hessian by summing over samples.
    # To avoid large memory, we can compute the Hessian block-wise.
    # But 1290x1290 is only ~1.66M floats, fine.
    D = 1290
    Hess = np.zeros((D, D), dtype=np.float64)
    for n in range(N):
        h = H[n]  # [128]
        p = P[n]  # [10]
        # Build J_n: [10, 1290]
        J = np.zeros((10, D), dtype=np.float64)
        # W2 part: for each class c, J[c, c*128:(c+1)*128] = h
        for c in range(10):
            J[c, c*128:(c+1)*128] = h
        # b2 part: J[c, 1280+c] = 1
        J[:, 1280:] = np.eye(10)
        # Hessian contribution: J^T (diag(p) - p p^T) J
        S = np.diag(p) - np.outer(p, p)
        Hess += J.T @ S @ J
    Hess /= N

    # Add small ridge for numerical stability
    ridge = 1e-4
    Hess += ridge * np.eye(D)

    # Gradient vector
    grad = np.concatenate([grad_W2.flatten(), grad_b2])

    # Newton step
    try:
        delta = -np.linalg.solve(Hess, grad)
    except np.linalg.LinAlgError:
        delta = -np.linalg.lstsq(Hess, grad, rcond=None)[0]

    # Apply update
    delta_W2 = delta[:1280].reshape(10, 128)
    delta_b2 = delta[1280:]
    W2 += delta_W2
    b2 += delta_b2

    # Clip to safe range
    W2 = np.clip(W2, -1000.0, 1000.0)
    b2 = np.clip(b2, -1000.0, 1000.0)

    # Write back to w (preserve dtype float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # Diagnostics
    diagnostics = {
        "alpha": alpha,
        "ce_before": float(ce0),
        "grad_norm": float(np.linalg.norm(grad)),
        "delta_norm": float(np.linalg.norm(delta)),
        "fc2_weight_norm": float(np.linalg.norm(W2)),
        "fc2_bias_mean": float(np.mean(b2)),
    }
    return {"weights": w, "diagnostics": diagnostics}
