import numpy as np
from numpy.lib.stride_tricks import as_strided

def edit(w, f):
    """
    Closed-form least-squares refit of the final 3x3 convolutional output layer.
    The final hidden feature maps (f['features']) are fixed; only out.weight and out.bias
    are updated to minimise MSE against the target epsilon f['y'].
    """
    features = f['features']          # (2000, 64, 16, 16)
    y = f['y']                        # (2000, 1, 16, 16)
    N, C, H, W = features.shape
    K = 3

    # Pad features with one zero-pixel border (conv pad=1)
    features_pad = np.pad(features, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

    # Accumulate normal equations for X^T X and X^T y
    # X has shape (N*H*W, C*K*K + 1)  (last column is bias)
    n_params = C * K * K + 1
    XtX = np.zeros((n_params, n_params), dtype=np.float64)
    Xty = np.zeros((n_params, 1), dtype=np.float64)

    chunk_size = 200
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = features_pad[start:end]          # (chunk, C, H+2, W+2)
        s0, s1, s2, s3 = chunk.strides

        # Extract 3x3 patches for every spatial location
        shape = (end - start, H, W, C, K, K)
        strides = (s0, s2, s3, s1, s2, s3)
        patches = as_strided(chunk, shape=shape, strides=strides)
        patches = patches.reshape(-1, C * K * K)  # (chunk*H*W, C*K*K)

        # Append bias column
        ones = np.ones((patches.shape[0], 1), dtype=np.float64)
        X = np.concatenate([patches.astype(np.float64), ones], axis=1)

        y_chunk = y[start:end].reshape(-1, 1).astype(np.float64)

        XtX += X.T @ X
        Xty += X.T @ y_chunk

    # Small ridge for numerical stability
    ridge = 1e-6
    XtX += ridge * np.eye(n_params)

    # Solve for optimal weights and bias
    w_opt = np.linalg.solve(XtX, Xty)          # (C*K*K + 1, 1)
    W_new = w_opt[:-1].reshape(1, C, K, K).astype(np.float32)
    b_new = w_opt[-1].astype(np.float32)

    # Update the network parameters
    w['out.weight'] = W_new
    w['out.bias'] = b_new

    # Optional diagnostics
    diagnostics = {
        'out_weight_norm': float(np.linalg.norm(W_new)),
        'out_bias': float(b_new),
        'ridge': ridge,
    }
    return {"weights": w, "diagnostics": diagnostics}
