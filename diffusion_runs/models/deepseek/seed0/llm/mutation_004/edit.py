import numpy as np

def edit(w, f):
    # Extract final hidden features and targets for all 2000 training examples.
    # features: (2000, 64, 16, 16) -> input to the final 3x3 conv layer
    # y:        (2000, 1, 16, 16) -> target epsilon
    F = f["features"].astype(np.float32)
    y = f["y"].astype(np.float32)
    N, C, H, W = F.shape
    y_flat = y.reshape(-1)

    # Pad features for 3x3 convolution with padding=1.
    F_pad = np.pad(F, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

    # Build the design matrix for the final convolution.
    # For each of the 9 kernel offsets (dr, dc) in {-1,0,1}^2, we extract the
    # corresponding shifted feature map and flatten it to (N*H*W, C).
    # The columns are ordered as (kr, kc) with kr = dr+1, kc = dc+1.
    blocks = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            shifted = F_pad[:, :, 1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
            # (N, C, H, W) -> (N*H*W, C)
            shifted = shifted.transpose(0, 2, 3, 1).reshape(-1, C)
            blocks.append(shifted)
    X = np.concatenate(blocks, axis=1)  # (N*H*W, 9*C)
    # Add a bias column.
    X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)

    # Solve the regularized least-squares problem in closed form.
    # A = X^T X + ridge * I, b = X^T y
    ridge = 1e-3
    A = X.T @ X
    b = X.T @ y_flat
    A += ridge * np.eye(A.shape[0], dtype=np.float32)
    w_sol = np.linalg.solve(A, b)

    # Reshape the solution into the final convolution weight and bias.
    # w_sol layout: 9 blocks of C weights, then 1 bias.
    W_out = np.zeros((1, C, 3, 3), dtype=np.float32)
    idx = 0
    for kr in range(3):
        for kc in range(3):
            W_out[0, :, kr, kc] = w_sol[idx:idx + C]
            idx += C
    b_out = w_sol[-1:].astype(np.float32)

    # Update the network weights.
    w["out.weight"] = W_out
    w["out.bias"] = b_out

    # Small diagnostics dictionary.
    mse_before = float(np.mean((y_flat - f["prediction"].reshape(-1)) ** 2))
    mse_after = float(np.mean((y_flat - X @ w_sol) ** 2))
    diagnostics = {
        "mse_before": mse_before,
        "mse_after_estimate": mse_after,
        "ridge": ridge,
        "out_weight_norm": float(np.linalg.norm(W_out)),
        "out_bias": float(b_out[0]),
    }

    return {"weights": w, "diagnostics": diagnostics}
