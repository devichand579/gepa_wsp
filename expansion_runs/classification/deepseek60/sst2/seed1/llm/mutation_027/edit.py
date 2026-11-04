import numpy as np

def edit(w, f):
    # Extract features and labels from the full feedback pool
    X = f["features"].astype(np.float64)   # [4000, 32]
    y = f["y"].astype(np.int64)            # [4000]
    
    # Standardize features (zero mean, unit variance) to make LDA well-conditioned
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma
    
    # Class means and counts
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples, n_features = Xs.shape
    means = np.zeros((n_classes, n_features), dtype=np.float64)
    priors = np.zeros(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        mask = (y == c)
        means[i] = Xs[mask].mean(axis=0)
        priors[i] = mask.mean()
    
    # Within-class scatter matrix
    Sw = np.zeros((n_features, n_features), dtype=np.float64)
    for i, c in enumerate(classes):
        mask = (y == c)
        Xc = Xs[mask] - means[i]
        Sw += Xc.T @ Xc
    
    # Regularized within-class scatter (shrinkage toward diagonal)
    shrinkage = 0.15
    Sw_reg = (1.0 - shrinkage) * Sw + shrinkage * np.trace(Sw) / n_features * np.eye(n_features)
    
    # LDA direction(s): for binary case, w = Sw_reg^{-1} @ (mean1 - mean0)
    # Here we use the difference between class 1 and class 0 (positive vs negative)
    diff = means[1] - means[0]
    w_dir = np.linalg.solve(Sw_reg, diff)
    
    # Normalize direction to unit norm for numerical stability
    w_dir = w_dir / (np.linalg.norm(w_dir) + 1e-12)
    
    # Compute bias: threshold at the midpoint of projected class means
    proj0 = means[0] @ w_dir
    proj1 = means[1] @ w_dir
    bias = -0.5 * (proj0 + proj1)
    
    # Convert back to original feature space: w_orig = w_dir / sigma
    w_orig = w_dir / sigma.flatten()
    
    # Set head weights: class 0 (negative) gets -w_orig, class 1 (positive) gets +w_orig
    # head.weight shape [2, 32], head.bias shape [2]
    w["head.weight"][0] = -w_orig.astype(np.float32)
    w["head.weight"][1] =  w_orig.astype(np.float32)
    w["head.bias"][0] = (-bias).astype(np.float32)
    w["head.bias"][1] = ( bias).astype(np.float32)
    
    # Diagnostics
    train_logits = Xs @ w_dir + bias
    train_pred = (train_logits > 0).astype(np.int64)
    train_acc = (train_pred == y).mean()
    
    diagnostics = {
        "shrinkage": shrinkage,
        "train_accuracy": float(train_acc),
        "w_dir_norm": float(np.linalg.norm(w_dir)),
        "bias": float(bias),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
