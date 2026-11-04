import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"]  # (2000, 128)
    y = f["y"]         # (2000,)
    
    # Standardize features (zero mean, unit variance)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Z = (X - mean) / std
    
    # Compute class means and global mean
    classes = np.unique(y)
    mu = Z.mean(axis=0)
    mu_c = np.stack([Z[y == c].mean(axis=0) for c in classes])
    
    # Within-class scatter matrix
    Sw = np.zeros((Z.shape[1], Z.shape[1]), dtype=np.float64)
    for c in classes:
        Zc = Z[y == c]
        diff = Zc - Zc.mean(axis=0)
        Sw += diff.T @ diff
    
    # Regularize (small ridge for numerical stability)
    alpha = 1e-3
    Sw_reg = Sw + alpha * np.eye(Z.shape[1])
    
    # LDA weights in standardized space: W_z = (mu_c - mu) @ inv(Sw_reg)
    W_z = np.linalg.solve(Sw_reg, (mu_c - mu).T).T  # (10, 128)
    b_z = -W_z @ mu
    
    # Transform back to original feature space
    W = W_z / std[None, :]
    b = b_z - (W_z * mean / std).sum(axis=1)
    
    # Clip to safe range and cast to float32
    W = np.clip(W, -1000, 1000).astype(np.float32)
    b = np.clip(b, -1000, 1000).astype(np.float32)
    
    # Update only the second-layer weights
    w["fc2.weight"] = W
    w["fc2.bias"] = b
    
    # Diagnostics
    logits = Z @ W_z.T + b_z
    pred = np.argmax(logits, axis=1)
    acc = (pred == y).mean()
    
    # Cross-entropy
    logits_max = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    ce = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
    
    diagnostics = {
        "method": "lda_standardized",
        "alpha": alpha,
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "fc2_weight_mean": float(W.mean()),
        "fc2_weight_std": float(W.std()),
        "fc2_bias_mean": float(b.mean()),
        "fc2_bias_std": float(b.std()),
        "condition_number_Sw_reg": float(np.linalg.cond(Sw_reg))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
