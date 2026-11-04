import numpy as np

def edit(w, f):
    # Extract current weights
    W1 = w["fc1.weight"].copy()
    b1 = w["fc1.bias"].copy()
    W2 = w["fc2.weight"].copy()
    b2 = w["fc2.bias"].copy()
    
    # Full feedback pool features and labels
    H = f["features"]          # [2000, 128] post-ReLU hidden features
    y = f["y"]                 # [2000] int64 labels
    N, D = H.shape
    C = 10
    
    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    
    # Current logits and accuracy
    logits_cur = H @ W2.T + b2
    pred_cur = np.argmax(logits_cur, axis=1)
    acc_cur = np.mean(pred_cur == y)
    
    # Ridge regression: solve for W2_new, b2_new
    # Augment features with a bias column
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])  # [N, D+1]
    # Ridge solution: (H_aug^T H_aug + alpha I)^-1 H_aug^T Y
    alpha = 1e-3
    A = H_aug.T @ H_aug + alpha * np.eye(D + 1, dtype=np.float32)
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # [D+1, C]
    W2_new = W_aug[:D, :].T        # [C, D]
    b2_new = W_aug[D, :]           # [C]
    
    # Evaluate new readout
    logits_new = H @ W2_new.T + b2_new
    pred_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(pred_new == y)
    
    # Keep the better one
    if acc_new >= acc_cur:
        w["fc2.weight"] = W2_new.astype(np.float32)
        w["fc2.bias"] = b2_new.astype(np.float32)
        chosen = "ridge"
        acc_final = acc_new
    else:
        chosen = "current"
        acc_final = acc_cur
    
    # Diagnostics
    diagnostics = {
        "chosen": chosen,
        "acc_current": float(acc_cur),
        "acc_ridge": float(acc_new),
        "acc_final": float(acc_final),
        "alpha": alpha,
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
