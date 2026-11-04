import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float32)  # (N, 784)
    y = f["y"].astype(np.int64)
    N = X.shape[0]
    
    # Center data
    mean = X.mean(axis=0)
    Xc = X - mean
    
    # PCA: compute covariance and top 128 eigenvectors
    C = (Xc.T @ Xc) / N
    eigvals, eigvecs = np.linalg.eigh(C)
    # eigh returns ascending order; take last 128
    V = eigvecs[:, -128:].T.astype(np.float32)  # (128, 784)
    
    # Set fc1: projection onto PCA components, centered
    w["fc1.weight"] = V
    w["fc1.bias"] = (-mean @ V.T).astype(np.float32)
    
    # Compute hidden features for all training samples
    H = np.maximum(0, X @ V.T + w["fc1.bias"])  # (N, 128)
    
    # Add bias column
    H_aug = np.hstack([H, np.ones((N, 1), dtype=np.float32)])
    
    # One-hot targets
    Y = np.zeros((N, 10), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    
    # Ridge regression: solve (H_aug^T H_aug + lambda I) W = H_aug^T Y
    lam = 1e-3
    A = H_aug.T @ H_aug + lam * np.eye(H_aug.shape[1], dtype=np.float32)
    B = H_aug.T @ Y
    W = np.linalg.solve(A, B)  # (129, 10)
    
    # Set fc2
    w["fc2.weight"] = W[:128, :].T.astype(np.float32)  # (10, 128)
    w["fc2.bias"] = W[128, :].astype(np.float32)       # (10,)
    
    # Ensure all weights are finite and within bounds
    for k in w:
        w[k] = np.clip(w[k].astype(np.float32), -1000.0, 1000.0)
    
    diagnostics = {
        "pca_explained_variance_ratio": float(eigvals[-128:].sum() / eigvals.sum()),
        "hidden_mean": float(H.mean()),
        "hidden_std": float(H.std()),
        "train_accuracy": float((np.argmax(H_aug @ W, axis=1) == y).mean()),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
