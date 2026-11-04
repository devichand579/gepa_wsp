import numpy as np

def edit(w, f):
    # Extract current head parameters
    W_old = w["head.weight"].astype(np.float64)  # [10, 1568]
    b_old = w["head.bias"].astype(np.float64)    # [10]
    
    # Features and labels from the full feedback pool
    X = f["features"].astype(np.float64)         # [2000, 1568]
    y = f["y"].astype(np.int64)                  # [2000]
    N, D = X.shape
    C = 10
    
    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Current logits and predictions
    L_old = X @ W_old.T + b_old                  # [2000, 10]
    
    # Residual targets: what the current head misses
    R = Y - L_old                                # [2000, 10]
    
    # We want to find delta_W, delta_b that minimize
    # ||X @ delta_W.T + delta_b - R||^2 + lambda * ||delta_W||^2
    # This is a ridge regression on the residual.
    # Add a bias column to X
    X_aug = np.hstack([X, np.ones((N, 1))])      # [2000, 1569]
    
    # Regularization: we only penalize the weights, not the bias.
    # So we build a diagonal matrix for the augmented features.
    lambda_reg = 1.0  # small regularization to avoid overfitting
    
    # Solve for each class independently:
    # delta = (X_aug^T X_aug + lambda * I_aug)^-1 X_aug^T R[:, c]
    # where I_aug has 0 for the bias term.
    I_aug = np.eye(D + 1)
    I_aug[D, D] = 0.0  # no penalty on bias
    
    # Precompute the inverse or solve the system
    # Since D=1568, N=2000, we can use the normal equations.
    # X_aug^T X_aug is [1569, 1569]
    XtX = X_aug.T @ X_aug
    reg_matrix = lambda_reg * I_aug
    A = XtX + reg_matrix
    
    # Solve for all classes at once: A @ Delta = X_aug^T R
    # Delta is [1569, 10]
    XtR = X_aug.T @ R
    Delta = np.linalg.solve(A, XtR)              # [1569, 10]
    
    delta_W = Delta[:D, :].T                     # [10, 1568]
    delta_b = Delta[D, :]                        # [10]
    
    # New head parameters
    W_new = W_old + delta_W
    b_new = b_old + delta_b
    
    # Update weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)
    
    # Diagnostics
    L_new = X @ W_new.T + b_new
    preds_new = np.argmax(L_new, axis=1)
    acc_new = np.mean(preds_new == y)
    
    # Cross-entropy
    def softmax(logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probs_new = softmax(L_new)
    ce_new = -np.mean(np.log(probs_new[np.arange(N), y] + 1e-12))
    
    diagnostics = {
        "train_accuracy": float(acc_new),
        "train_cross_entropy": float(ce_new),
        "lambda_reg": lambda_reg,
        "delta_W_norm": float(np.linalg.norm(delta_W)),
        "delta_b_norm": float(np.linalg.norm(delta_b)),
        "head_weight_norm": float(np.linalg.norm(W_new)),
        "head_bias_norm": float(np.linalg.norm(b_new))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
