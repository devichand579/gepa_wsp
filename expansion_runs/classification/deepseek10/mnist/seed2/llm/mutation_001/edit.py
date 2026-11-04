import numpy as np

def edit(w, f):
    # Full feedback pool: hidden features (post-ReLU) and labels
    H = f["features"].astype(np.float64)      # (2000, 128)
    y = f["y"].astype(np.int64)               # (2000,)
    N = H.shape[0]
    
    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    
    # Add bias column to features
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float64)])  # (2000, 129)
    
    # Closed-form least-squares solution: W = argmin ||H_aug W - Y||^2
    W, residuals, rank, s = np.linalg.lstsq(H_aug, Y, rcond=None)  # (129, 10)
    
    # Extract fc2 weights and bias
    fc2_weight = W[:128, :].T   # (10, 128)
    fc2_bias   = W[128, :]      # (10,)
    
    # Update the network weights
    w["fc2.weight"] = fc2_weight.astype(np.float32)
    w["fc2.bias"]   = fc2_bias.astype(np.float32)
    
    # --- Diagnostics ---
    # New accuracy on the full pool
    new_logits = H @ fc2_weight.T + fc2_bias
    new_preds  = np.argmax(new_logits, axis=1)
    new_acc    = np.mean(new_preds == y)
    
    # Old accuracy on the full pool
    old_logits = f["logits"].astype(np.float64)
    old_preds  = np.argmax(old_logits, axis=1)
    old_acc    = np.mean(old_preds == y)
    
    # Accuracy on the paired batch
    batch_H = f["layer.fc2.input0"].astype(np.float64)  # (128, 128)
    batch_y = f["batch.y"].astype(np.int64)
    batch_logits = batch_H @ fc2_weight.T + fc2_bias
    batch_preds  = np.argmax(batch_logits, axis=1)
    batch_acc    = np.mean(batch_preds == batch_y)
    
    # Condition number of the normal-equation matrix
    cond = np.linalg.cond(H_aug.T @ H_aug)
    
    diagnostics = {
        "old_accuracy": float(old_acc),
        "new_accuracy": float(new_acc),
        "batch_accuracy": float(batch_acc),
        "rank": int(rank),
        "condition_number": float(cond),
        "fc2_weight_norm": float(np.linalg.norm(fc2_weight)),
        "fc2_bias_norm": float(np.linalg.norm(fc2_bias)),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
