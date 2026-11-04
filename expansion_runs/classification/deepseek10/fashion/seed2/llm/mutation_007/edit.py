import numpy as np

def edit(w, f):
    # Extract current head parameters
    W_old = w["head.weight"].copy()  # [10, 1568]
    b_old = w["head.bias"].copy()    # [10]
    
    # Get full feedback features and labels
    X = f["features"]                # [2000, 1568]
    y = f["y"]                       # [2000]
    N, D = X.shape
    C = 10
    
    # One-hot encode labels
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    
    # Ridge regression toward current weights:
    # Minimize ||X W^T + b - Y||^2 + lambda * (||W - W_old||^2 + ||b - b_old||^2)
    # Solution: W = (X^T X + lambda I)^-1 (X^T Y + lambda W_old^T)
    #           b = mean(Y - X W^T) + lambda/(N+lambda) * (b_old - mean(Y - X W^T))? 
    # Actually, we can solve for W and b jointly by augmenting X with a column of ones.
    # Let Z = [X, 1] (N x (D+1)), V = [W, b]^T ((D+1) x C)
    # Minimize ||Z V - Y||^2 + lambda * ||V - V_old||^2
    # Solution: V = (Z^T Z + lambda I)^-1 (Z^T Y + lambda V_old)
    
    lambda_reg = 1000.0  # strong regularization
    
    # Build augmented matrix Z
    Z = np.hstack([X, np.ones((N, 1), dtype=np.float32)])  # [2000, 1569]
    
    # Current augmented parameters V_old: [1569, 10]
    V_old = np.vstack([W_old.T, b_old.reshape(1, -1)])  # [1569, 10]
    
    # Compute normal equations
    ZTZ = Z.T @ Z  # [1569, 1569]
    ZTY = Z.T @ Y  # [1569, 10]
    
    # Add regularization
    A = ZTZ + lambda_reg * np.eye(D + 1, dtype=np.float32)
    B = ZTY + lambda_reg * V_old
    
    # Solve for V
    V_new = np.linalg.solve(A, B)  # [1569, 10]
    
    # Extract new weights and biases
    W_new = V_new[:-1, :].T  # [10, 1568]
    b_new = V_new[-1, :]     # [10]
    
    # Update weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)
    
    # Diagnostics
    delta_W = W_new - W_old
    delta_b = b_new - b_old
    diagnostics = {
        "lambda_reg": lambda_reg,
        "delta_W_norm": float(np.linalg.norm(delta_W)),
        "delta_b_norm": float(np.linalg.norm(delta_b)),
        "W_new_norm": float(np.linalg.norm(W_new)),
        "b_new_norm": float(np.linalg.norm(b_new)),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
