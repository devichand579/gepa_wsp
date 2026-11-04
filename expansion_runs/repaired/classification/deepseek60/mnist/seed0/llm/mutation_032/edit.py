import numpy as np

def edit(w, f):
    # Extract current weights
    W1 = w["fc1.weight"].copy()
    b1 = w["fc1.bias"].copy()
    W2 = w["fc2.weight"].copy()
    b2 = w["fc2.bias"].copy()
    
    # Full feedback features and labels
    H = f["features"]          # (2000, 128)
    y = f["y"]                 # (2000,)
    N = H.shape[0]
    
    # Current logits and predictions
    Z = H @ W2.T + b2          # (2000, 10)
    pred = np.argmax(Z, axis=1)
    
    # Identify errors
    err_mask = pred != y
    if np.sum(err_mask) == 0:
        return w  # no errors, nothing to do
    
    # Compute mean feature difference for each (true, pred) pair
    # We'll accumulate sum of features for each error pair
    # and count them.
    # Initialize accumulators for each (true, pred) pair (10x10)
    sum_feat = np.zeros((10, 10, H.shape[1]), dtype=np.float64)
    count_pair = np.zeros((10, 10), dtype=np.int64)
    
    for i in np.where(err_mask)[0]:
        c = y[i]
        d = pred[i]
        sum_feat[c, d] += H[i]
        count_pair[c, d] += 1
    
    # For each pair with errors, compute mean difference and update W2
    # We'll do a line search over alpha to maximize accuracy on full feedback.
    # To avoid overfitting, we can also consider a small ridge penalty on the update.
    # We'll try a few alpha values and pick the best accuracy.
    # The update: W2[c] += alpha * mean_feat[c,d]
    #            W2[d] -= alpha * mean_feat[c,d]
    #            b2[c] += alpha * mean_bias? Actually we can also update bias.
    # We'll just update weights and biases symmetrically.
    
    # Compute mean feature for each pair
    mean_feat = np.zeros((10, 10, H.shape[1]), dtype=np.float64)
    for c in range(10):
        for d in range(10):
            if count_pair[c, d] > 0:
                mean_feat[c, d] = sum_feat[c, d] / count_pair[c, d]
    
    # Also compute mean bias difference? We can just use a constant bias update.
    # For simplicity, we'll update bias by a small constant as well.
    # But let's first try weight-only update.
    
    # Line search over alpha
    alphas = np.concatenate([
        np.linspace(0, 2.0, 21),
        np.linspace(2.0, 10.0, 9)
    ])
    best_acc = -1
    best_alpha = 0
    best_W2 = W2.copy()
    best_b2 = b2.copy()
    
    # Precompute original Z for speed
    Z_base = Z.copy()
    
    for alpha in alphas:
        W2_new = W2.copy()
        b2_new = b2.copy()
        # Apply updates
        for c in range(10):
            for d in range(10):
                if count_pair[c, d] > 0:
                    delta = alpha * mean_feat[c, d]
                    W2_new[c] += delta
                    W2_new[d] -= delta
                    # Bias update: we can add a small constant to c and subtract from d
                    # But we need to be careful. Let's just update weights first.
        Z_new = H @ W2_new.T + b2_new
        acc = np.mean(np.argmax(Z_new, axis=1) == y)
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
            best_W2 = W2_new
            best_b2 = b2_new
    
    # If best_alpha is 0, no improvement
    if best_alpha == 0:
        return w
    
    # Optionally, we can also update bias by a small amount.
    # Let's try a small bias update: for each pair, add alpha_b * 1 to c and -alpha_b to d.
    # But we can just keep the weight update.
    
    # Update weights
    w["fc2.weight"] = best_W2.astype(np.float32)
    w["fc2.bias"] = best_b2.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "best_alpha": float(best_alpha),
        "best_acc": float(best_acc),
        "n_errors_before": int(np.sum(err_mask)),
        "n_errors_after": int(np.sum(np.argmax(H @ best_W2.T + best_b2, axis=1) != y)),
        "fc2_weight_norm": float(np.linalg.norm(best_W2)),
        "fc2_bias_norm": float(np.linalg.norm(best_b2)),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
