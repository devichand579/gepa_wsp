import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"]                     # [2000, 1, 28, 28]
    y = f["y"]                     # [2000]
    N = X.shape[0]
    X_flat = X.reshape(N, -1)      # [2000, 784]
    
    # Center the data
    mean = X_flat.mean(axis=0, keepdims=True)
    Xc = X_flat - mean
    
    # Compute PCA via SVD on centered data
    # We need top 128 components
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Vt shape: [min(N,784), 784]. We need 128 components.
    # Since N=2000, min is 784, so Vt is [784, 784]. Top 128 rows.
    components = Vt[:128, :]       # [128, 784]
    
    # Project training data onto PCA components
    H_pca = Xc @ components.T      # [2000, 128]
    
    # Set first-layer weights to PCA components
    w["fc1.weight"] = components.astype(np.float32)
    
    # Set first-layer bias to a large positive constant to make ReLU identity
    # We need to ensure H_pca + bias > 0 for all training samples.
    # Find minimum value in H_pca and set bias accordingly.
    min_val = H_pca.min()
    bias_val = -min_val + 10.0     # add margin
    w["fc1.bias"] = np.full(128, bias_val, dtype=np.float32)
    
    # Compute new hidden features for training (with bias added)
    H = H_pca + bias_val           # [2000, 128]
    
    # Fit second-layer weights via ridge regression (closed-form)
    # We want to predict one-hot labels from H.
    # Add bias term to H for the linear model.
    H_aug = np.concatenate([H, np.ones((N, 1))], axis=1)  # [2000, 129]
    
    # One-hot labels
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    
    # Ridge regression: W = (H_aug^T H_aug + lambda I)^{-1} H_aug^T Y
    # Use a small lambda for numerical stability.
    lam = 1e-3
    A = H_aug.T @ H_aug
    A += lam * np.eye(A.shape[0])
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A, B)   # [129, 10]
    
    # Extract fc2 weights and bias
    # W_aug[:128, :] is [128, 10], we need [10, 128] for fc2.weight
    w["fc2.weight"] = W_aug[:128, :].T.astype(np.float32)
    # The bias for the linear model is W_aug[128, :], but we already have
    # the bias_val added to H. The network computes logits = fc2.weight @ h + fc2.bias
    # where h = fc1.weight @ x + fc1.bias.
    # Our linear model is: logits = W_aug[:128].T @ (fc1.weight @ x + fc1.bias) + W_aug[128]
    # = (W_aug[:128].T @ fc1.weight) @ x + W_aug[:128].T @ fc1.bias + W_aug[128]
    # So fc2.bias should be W_aug[128] - W_aug[:128].T @ fc1.bias.
    fc2_bias = W_aug[128, :] - W_aug[:128, :].T @ w["fc1.bias"]
    w["fc2.bias"] = fc2_bias.astype(np.float32)
    
    # Ensure all values are finite and within bounds
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    # Diagnostics
    # Compute training accuracy
    logits = H @ w["fc2.weight"].T + w["fc2.bias"]
    preds = np.argmax(logits, axis=1)
    train_acc = np.mean(preds == y)
    
    diagnostics = {
        "train_accuracy": float(train_acc),
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc1_bias_mean": float(w["fc1.bias"].mean()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
        "bias_val": float(bias_val),
        "lambda": lam
    }
    
    return {"weights": w, "diagnostics": diagnostics}
