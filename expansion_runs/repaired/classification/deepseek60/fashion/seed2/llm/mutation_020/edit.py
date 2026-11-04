import numpy as np

def edit(w, f):
    # Extract cached features and labels
    X = f["features"]  # (2000, 1568)
    y = f["y"]         # (2000,)
    N, D = X.shape
    
    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Augment features with a bias column
    X_aug = np.hstack([X, np.ones((N, 1), dtype=np.float64)])  # (2000, 1569)
    
    best_acc = -1.0
    best_ce = np.inf
    best_W = None
    best_b = None
    
    # Try several ridge regularization strengths
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        # Solve (X_aug^T X_aug + lam I) W_aug^T = X_aug^T Y
        A = X_aug.T @ X_aug + lam * np.eye(D + 1)
        B = X_aug.T @ Y
        W_aug = np.linalg.solve(A, B).T  # (10, 1569)
        
        W = W_aug[:, :D]
        b = W_aug[:, D]
        
        # Compute logits and probabilities
        logits = X @ W.T + b
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Accuracy
        pred = np.argmax(logits, axis=1)
        acc = np.mean(pred == y)
        
        # Cross-entropy
        probs_clipped = np.clip(probs, 1e-15, 1.0)
        ce = -np.mean(np.log(probs_clipped[np.arange(N), y]))
        
        # Check weight bounds
        max_abs = max(np.max(np.abs(W)), np.max(np.abs(b)))
        if max_abs > 1000.0:
            # Scale down to fit within bounds
            scale = 1000.0 / max_abs
            W = W * scale
            b = b * scale
            # Recompute metrics after scaling
            logits = X @ W.T + b
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            pred = np.argmax(logits, axis=1)
            acc = np.mean(pred == y)
            probs_clipped = np.clip(probs, 1e-15, 1.0)
            ce = -np.mean(np.log(probs_clipped[np.arange(N), y]))
        
        if acc > best_acc or (acc == best_acc and ce < best_ce):
            best_acc = acc
            best_ce = ce
            best_W = W
            best_b = b
    
    # Update the head parameters
    w["head.weight"] = best_W.astype(np.float32)
    w["head.bias"] = best_b.astype(np.float32)
    
    # Final safety clip
    w["head.weight"] = np.clip(w["head.weight"], -1000.0, 1000.0)
    w["head.bias"] = np.clip(w["head.bias"], -1000.0, 1000.0)
    
    return w
