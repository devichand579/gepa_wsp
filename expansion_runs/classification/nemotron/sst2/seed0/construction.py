def edit(w, f):
    # Extract the full feedback pool
    X = f["features"]       # [4000, 32], float32  — final hidden features before head
    y = f["y"]              # [4000], int64 — true labels (0=negative, 1=positive)
    
    # Binary class masks (balanced: 2000 each)
    pos = y == 1
    neg = y == 0
    
    # Class-conditional means
    mu_pos = X[pos].mean(axis=0)   # [32]
    mu_neg = X[neg].mean(axis=0)   # [32]
    
    # Within-class scatter matrix (pooled covariance)
    Sp = ((X[pos] - mu_pos).T @ (X[pos] - mu_pos) +
          (X[neg] - mu_neg).T @ (X[neg] - mu_neg)) / (X.shape[0] - 2)
    # Ridge regularization for numerical stability
    Sp += 1e-4 * np.eye(32, dtype=np.float32)
    
    # LDA projection direction: w = S_p^{-1} (mu_pos - mu_neg)
    w_dir = np.linalg.solve(Sp, (mu_pos - mu_neg).astype(np.float32))  # [32]
    
    # Construct head weights and bias so that logit > 0 ⇢ positive class
    # head.weight shape [2, 32]; row 0 = negative, row 1 = positive
    head_weight = np.array([-w_dir, w_dir], dtype=np.float32)          # [2, 32]
    head_bias   = np.array([-np.dot(w_dir, mu_neg), -np.dot(w_dir, mu_pos)], dtype=np.float32)  # [2]
    
    # Edit the head parameters directly in the weight dictionary
    w["head.weight"] = head_weight
    w["head.bias"]   = head_bias
    
    # Diagnostics: predicted logits and accuracy on the feedback pool
    logits_pool = X @ head_weight.T + head_bias          # [4000, 2]
    preds       = np.argmax(logits_pool, axis=1)
    feedback_acc = float((preds == y).mean())
    
    diag = {
        "head_weight_norm":   float(np.linalg.norm(head_weight)),
        "lda_dir_norm":       float(np.linalg.norm(w_dir)),
        "feedback_pool_acc":  feedback_acc,
        "pos_mean_norm":      float(np.linalg.norm(mu_pos)),
        "neg_mean_norm":      float(np.linalg.norm(mu_neg)),
        "within_cov_trace":   float(np.trace(Sp))
    }
    
    return {"weights": w, "diagnostics": diag}
