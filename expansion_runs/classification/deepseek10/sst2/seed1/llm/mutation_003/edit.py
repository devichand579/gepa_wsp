import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)   # [4000, 32]
    y = f["y"].astype(np.int64)            # [4000]
    
    # Compute class means
    X0 = X[y == 0]
    X1 = X[y == 1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    
    # Compute pooled within-class covariance
    n0 = X0.shape[0]
    n1 = X1.shape[0]
    S0 = (X0 - mu0).T @ (X0 - mu0)
    S1 = (X1 - mu1).T @ (X1 - mu1)
    Sw = (S0 + S1) / (n0 + n1 - 2)
    
    # Regularize slightly for numerical stability
    Sw += 1e-4 * np.eye(Sw.shape[0])
    
    # LDA weight and bias
    w_lda = np.linalg.solve(Sw, mu1 - mu0)   # [32]
    b_lda = -0.5 * w_lda @ (mu0 + mu1)
    
    # The head has shape [2, 32] and bias [2]. We want logits = [0, w_lda @ x + b_lda]
    # So set head.weight[0] = 0, head.weight[1] = w_lda, head.bias[0] = 0, head.bias[1] = b_lda
    head_w = np.zeros((2, 32), dtype=np.float32)
    head_b = np.zeros(2, dtype=np.float32)
    head_w[1] = w_lda.astype(np.float32)
    head_b[1] = np.float32(b_lda)
    
    # Update weights
    w["head.weight"] = head_w
    w["head.bias"] = head_b
    
    # Set final LayerNorm to identity to avoid distortion
    w["norm.weight"] = np.ones(32, dtype=np.float32)
    w["norm.bias"] = np.zeros(32, dtype=np.float32)
    
    # Optional: also set all block weights to zero to make blocks identity?
    # Actually, we don't need to change blocks because the features are already computed
    # with the current blocks. Changing blocks would invalidate the features. We keep them as is.
    
    # Diagnostics
    train_pred = (X @ w_lda + b_lda > 0).astype(np.int64)
    train_acc = np.mean(train_pred == y)
    diagnostics = {
        "lda_train_accuracy": float(train_acc),
        "lda_weight_norm": float(np.linalg.norm(w_lda)),
        "lda_bias": float(b_lda),
        "class_means_diff_norm": float(np.linalg.norm(mu1 - mu0)),
        "pooled_cov_trace": float(np.trace(Sw)),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
