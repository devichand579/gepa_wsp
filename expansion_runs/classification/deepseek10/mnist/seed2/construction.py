import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (2000, 784)
    Y = f["y"].astype(np.int64)                                 # (2000,)
    N = X.shape[0]
    
    # Center the data
    mean = X.mean(axis=0)
    Xc = X - mean
    
    # PCA via SVD (top 128 components)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = 128
    W1 = Vt[:k, :].astype(np.float32)          # (128, 784)
    b1 = (-mean @ W1.T).astype(np.float32)     # (128,)
    
    # Compute hidden ReLU features
    H = np.maximum(0, X @ W1.T + b1)           # (2000, 128)
    
    # One-hot targets
    Y_onehot = np.eye(10, dtype=np.float32)[Y] # (2000, 10)
    
    # Ridge regression with bias (augment H with a column of ones)
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])  # (2000, 129)
    lambda_reg = 1e-3
    I = np.eye(H_aug.shape[1], dtype=np.float32)
    A = H_aug.T @ H_aug + lambda_reg * I
    B = H_aug.T @ Y_onehot
    W_aug = np.linalg.solve(A, B)              # (129, 10)
    
    W2 = W_aug[:-1, :].T.astype(np.float32)    # (10, 128)
    b2 = W_aug[-1, :].astype(np.float32)       # (10,)
    
    # Update weights
    w["fc1.weight"] = W1
    w["fc1.bias"] = b1
    w["fc2.weight"] = W2
    w["fc2.bias"] = b2
    
    # Diagnostics: training accuracy
    logits = H @ W2.T + b2
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == Y))
    
    diagnostics = {
        "train_accuracy": acc,
        "lambda_reg": lambda_reg,
        "k": k,
        "singular_values_top10": S[:10].tolist(),
        "fc1_weight_norm": float(np.linalg.norm(W1)),
        "fc2_weight_norm": float(np.linalg.norm(W2)),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
