import numpy as np

def edit(w, f):
    # Extract current fc2 parameters
    W = w["fc2.weight"].astype(np.float64)  # (10, 128)
    b = w["fc2.bias"].astype(np.float64)    # (10,)
    
    # Full feedback pool features and labels
    H_raw = f["features"].astype(np.float64)  # (2000, 128)
    y = f["y"].astype(np.int64)               # (2000,)
    
    # Center features using the full pool mean
    mu = H_raw.mean(axis=0)                   # (128,)
    H = H_raw - mu                            # (2000, 128)
    
    # Effective bias in centered parameterization
    b_eff = b + W @ mu                        # (10,)
    
    # Current logits and probabilities on the full pool
    logits = H @ W.T + b_eff                  # (2000, 10)
    # Stable softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    # One-hot targets
    Y = np.eye(10)[y]                         # (2000, 10)
    
    # Gradient
    G_W = (probs - Y).T @ H                   # (10, 128)
    G_b = (probs - Y).sum(axis=0)             # (10,)
    
    # Build augmented feature matrix X = [H, 1]
    X = np.concatenate([H, np.ones((H.shape[0], 1))], axis=1)  # (2000, 129)
    
    # Hessian construction
    n_classes = 10
    n_params = 129
    Hess = np.zeros((n_classes * n_params, n_classes * n_params))
    
    # First term: block diagonal
    for k in range(n_classes):
        block = X.T @ (X * probs[:, k][:, None])
        Hess[k*n_params:(k+1)*n_params, k*n_params:(k+1)*n_params] = block
    
    # Second term: - sum_i (p_i p_i^T) ⊗ (x_i x_i^T) = - PX.T @ PX
    PX = (probs[:, :, None] * X[:, None, :]).reshape(H.shape[0], n_classes * n_params)
    Hess -= PX.T @ PX
    
    # Ridge regularization for stability
    ridge = 1e-4
    Hess += ridge * np.eye(n_classes * n_params)
    
    # Gradient vector in the same ordering
    g = np.concatenate([G_W.flatten(), G_b])  # (1290,)
    # Note: The ordering must match the Hessian blocks. 
    # In Hessian, for each class k, the block is 129x129 with W_k (128) then b_k (1).
    # So the gradient vector should be: for k in 0..9: G_W[k] (128), G_b[k] (1).
    # The concatenation above is G_W.flatten() which is row-major: class 0 W, class 1 W, ... then G_b.
    # That does NOT match the block ordering. We need to interleave.
    # Correct ordering: for each class k: G_W[k], G_b[k].
    g_correct = np.zeros(n_classes * n_params)
    for k in range(n_classes):
        g_correct[k*n_params : k*n_params + 128] = G_W[k]
        g_correct[k*n_params + 128] = G_b[k]
    g = g_correct
    
    # Solve Newton system
    try:
        delta = np.linalg.solve(Hess, g)
    except np.linalg.LinAlgError:
        # Fallback: use least squares with small ridge
        delta = np.linalg.lstsq(Hess, g, rcond=None)[0]
    
    # Extract updates
    delta_W = np.zeros((n_classes, 128))
    delta_b = np.zeros(n_classes)
    for k in range(n_classes):
        delta_W[k] = delta[k*n_params : k*n_params + 128]
        delta_b[k] = delta[k*n_params + 128]
    
    # Apply Newton step
    W_new = W - delta_W
    b_eff_new = b_eff - delta_b
    
    # Convert back to original parameterization
    b_new = b_eff_new - W_new @ mu
    
    # Clip to safe range (absolute values <= 1000)
    W_new = np.clip(W_new, -1000.0, 1000.0).astype(np.float32)
    b_new = np.clip(b_new, -1000.0, 1000.0).astype(np.float32)
    
    # Update weights
    w["fc2.weight"] = W_new
    w["fc2.bias"] = b_new
    
    # Diagnostics
    diagnostics = {
        "mu_norm": float(np.linalg.norm(mu)),
        "delta_W_norm": float(np.linalg.norm(delta_W)),
        "delta_b_norm": float(np.linalg.norm(delta_b)),
        "W_new_mean": float(W_new.mean()),
        "W_new_std": float(W_new.std()),
        "b_new_mean": float(b_new.mean()),
        "b_new_std": float(b_new.std()),
        "ridge": ridge,
    }
    
    return {"weights": w, "diagnostics": diagnostics}
