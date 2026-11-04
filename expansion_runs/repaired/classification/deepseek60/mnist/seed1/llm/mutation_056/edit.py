import numpy as np

def edit(w, f):
    # Read current fc2 parameters
    W = w["fc2.weight"].astype(np.float64)   # [10,128]
    b = w["fc2.bias"].astype(np.float64)     # [10]

    # Current logits and labels from the full feedback pool
    Z = f["logits"].astype(np.float64)       # [N,10]
    y = f["y"].astype(np.int64)              # [N]
    N = Z.shape[0]

    # Stable softmax
    Zs = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Zs)
    P = E / E.sum(axis=1, keepdims=True)     # [N,10]

    # One-hot targets
    Y = np.zeros_like(P)
    Y[np.arange(N), y] = 1.0

    # Gradient of CE w.r.t. M and c, where Z' = Z @ M + c
    G = (P - Y) / N                          # [N,10]
    gM = Z.T @ G                             # [10,10]
    gc = G.sum(axis=0)                       # [10]
    g = np.concatenate([gM.ravel(), gc])     # [110]

    # Hessian of CE w.r.t. M and c
    # v_i = [z_i, 1], A_i = diag(p_i) - p_i p_i^T
    V = np.concatenate([Z, np.ones((N, 1))], axis=1)   # [N,11]
    A = np.zeros((N, 10, 10), dtype=np.float64)
    for i in range(N):
        p = P[i]
        A[i] = np.diag(p) - np.outer(p, p)

    # H = sum_i (v_i v_i^T) ⊗ A_i  -> [110,110]
    H = np.einsum('ni,nj,nkl->ijkl', V, V, A).reshape(110, 110)

    # Small ridge for numerical stability
    lam = 1e-6
    H += lam * np.eye(110)

    # One Newton step
    delta = -np.linalg.solve(H, g)
    dM = delta[:100].reshape(10, 10)
    dc = delta[100:]

    # Damped update to avoid overshoot
    alpha = 0.5
    M = np.eye(10) + alpha * dM
    c = alpha * dc

    # Apply to fc2: new logits = (Z @ M + c) = H @ (M.T @ W).T + (M.T @ b + c)
    W_new = M.T @ W
    b_new = M.T @ b + c

    # Write back as float32, preserving shapes
    w["fc2.weight"] = W_new.astype(np.float32)
    w["fc2.bias"] = b_new.astype(np.float32)

    # Diagnostics
    Z_new = Z @ M + c
    Z_new_s = Z_new - Z_new.max(axis=1, keepdims=True)
    E_new = np.exp(Z_new_s)
    P_new = E_new / E_new.sum(axis=1, keepdims=True)
    ce_old = -np.log(P[np.arange(N), y] + 1e-12).mean()
    ce_new = -np.log(P_new[np.arange(N), y] + 1e-12).mean()
    acc_old = (P.argmax(axis=1) == y).mean()
    acc_new = (P_new.argmax(axis=1) == y).mean()

    diagnostics = {
        "ce_old": float(ce_old),
        "ce_new": float(ce_new),
        "acc_old": float(acc_old),
        "acc_new": float(acc_new),
        "alpha": float(alpha),
        "lambda": float(lam),
        "fc2_weight_mean": float(W_new.mean()),
        "fc2_weight_std": float(W_new.std()),
        "fc2_bias_mean": float(b_new.mean()),
        "fc2_bias_std": float(b_new.std()),
    }

    return {"weights": w, "diagnostics": diagnostics}
