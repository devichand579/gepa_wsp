import numpy as np

def edit(w, f):
    # Extract features and labels from feedback
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    n, d = X.shape
    num_classes = 10
    
    # Compute class means
    means = np.zeros((num_classes, d), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        mask = (y == c)
        counts[c] = np.sum(mask)
        if counts[c] > 0:
            means[c] = np.mean(X[mask], axis=0)
    
    # Compute within-class scatter matrix S_w
    S_w = np.zeros((d, d), dtype=np.float64)
    for c in range(num_classes):
        mask = (y == c)
        if counts[c] > 0:
            Xc = X[mask] - means[c]
            S_w += Xc.T @ Xc
    
    # Regularize S_w
    # Use a small ridge penalty. We can estimate lambda via GCV-like approach,
    # but a fixed small value works well for LDA with many features.
    # We'll use lambda = 1.0 (scaled by trace/d).
    trace_Sw = np.trace(S_w)
    lambda_reg = 1.0 * (trace_Sw / d) if trace_Sw > 0 else 1.0
    S_w_reg = S_w + lambda_reg * np.eye(d, dtype=np.float64)
    
    # Invert S_w_reg (use solve for stability)
    # Compute W = S_w_reg^{-1} @ means^T
    # We can solve S_w_reg @ W = means^T
    W = np.linalg.solve(S_w_reg, means.T)  # (d, num_classes)
    
    # Compute bias: b_c = -0.5 * means_c^T @ W_c
    bias = -0.5 * np.sum(means * W.T, axis=1)  # (num_classes,)
    
    # Transpose to match head.weight shape (10, 1568)
    new_head_weight = W.T.astype(np.float32)
    new_head_bias = bias.astype(np.float32)
    
    # Clip to ensure finite and within bounds
    new_head_weight = np.clip(new_head_weight, -1000.0, 1000.0)
    new_head_bias = np.clip(new_head_bias, -1000.0, 1000.0)
    
    # Update weights
    w["head.weight"] = new_head_weight
    w["head.bias"] = new_head_bias
    
    # Diagnostics
    diagnostics = {
        "head_weight_norm": float(np.linalg.norm(new_head_weight)),
        "head_bias_norm": float(np.linalg.norm(new_head_bias)),
        "head_weight_max_abs": float(np.max(np.abs(new_head_weight))),
        "head_bias_max_abs": float(np.max(np.abs(new_head_bias))),
        "lambda_reg": float(lambda_reg),
        "trace_Sw": float(trace_Sw),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
