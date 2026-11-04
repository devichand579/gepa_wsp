import numpy as np

def edit(w, f):
    x = f["x"]          # (2000,1,28,28)
    y = f["y"]          # (2000,)
    N = x.shape[0]

    # ---------- Conv1 PCA ----------
    x_pad = np.pad(x, ((0,0),(0,0),(1,1),(1,1)))          # (N,1,30,30)
    win1 = np.lib.stride_tricks.sliding_window_view(x_pad, (3,3), axis=(2,3))  # (N,1,28,28,3,3)
    patches1 = win1.reshape(-1, 9)                         # (N*28*28, 9)
    mean1 = patches1.mean(axis=0)
    patches1_c = patches1 - mean1
    cov1 = patches1_c.T @ patches1_c / patches1.shape[0]
    evals1, evecs1 = np.linalg.eigh(cov1)
    idx1 = np.argsort(evals1)[::-1]
    evecs1 = evecs1[:, idx1]
    conv1_w = evecs1[:, :16].T.reshape(16,1,3,3).astype(np.float32)
    conv1_b = (-conv1_w.reshape(16,9) @ mean1).astype(np.float32)
    w["conv1.weight"] = conv1_w
    w["conv1.bias"] = conv1_b

    # ---------- Conv1 forward ----------
    win1_flat = win1.reshape(N, 28, 28, 9)
    conv1_out = win1_flat @ conv1_w.reshape(16,9).T + conv1_b   # (N,28,28,16)
    conv1_out = conv1_out.transpose(0,3,1,2)                    # (N,16,28,28)
    conv1_out = np.maximum(conv1_out, 0)
    pooled1 = conv1_out.reshape(N,16,14,2,14,2).max(axis=(3,5)) # (N,16,14,14)

    # ---------- Conv2 PCA ----------
    pooled1_pad = np.pad(pooled1, ((0,0),(0,0),(1,1),(1,1)))   # (N,16,16,16)
    win2 = np.lib.stride_tricks.sliding_window_view(pooled1_pad, (3,3), axis=(2,3))  # (N,16,14,14,3,3)
    patches2 = win2.transpose(0,2,3,1,4,5).reshape(-1, 16*3*3) # (N*14*14, 144)
    mean2 = patches2.mean(axis=0)
    patches2_c = patches2 - mean2
    cov2 = patches2_c.T @ patches2_c / patches2.shape[0]
    evals2, evecs2 = np.linalg.eigh(cov2)
    idx2 = np.argsort(evals2)[::-1]
    evecs2 = evecs2[:, idx2]
    conv2_w = evecs2[:, :32].T.reshape(32,16,3,3).astype(np.float32)
    conv2_b = (-conv2_w.reshape(32,144) @ mean2).astype(np.float32)
    w["conv2.weight"] = conv2_w
    w["conv2.bias"] = conv2_b

    # ---------- Conv2 forward ----------
    win2_flat = win2.transpose(0,2,3,1,4,5).reshape(N, 14, 14, 144)
    conv2_out = win2_flat @ conv2_w.reshape(32,144).T + conv2_b  # (N,14,14,32)
    conv2_out = conv2_out.transpose(0,3,1,2)                     # (N,32,14,14)
    conv2_out = np.maximum(conv2_out, 0)
    pooled2 = conv2_out.reshape(N,32,7,2,7,2).max(axis=(3,5))    # (N,32,7,7)
    features = pooled2.reshape(N, -1)                            # (N,1568)

    # ---------- Head ridge regression ----------
    Y = np.eye(10, dtype=np.float32)[y]                          # (N,10)
    X_mean = features.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = features - X_mean
    Yc = Y - Y_mean

    lambda_ = 1.0
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    S_inv = S / (S**2 + lambda_)
    W = Vt.T @ (S_inv[:, None] * (U.T @ Yc))                     # (1568,10)
    b = Y_mean - X_mean @ W                                      # (10,)

    w["head.weight"] = W.T.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # safety clip
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "conv1_weight_norm": float(np.linalg.norm(w["conv1.weight"])),
        "conv2_weight_norm": float(np.linalg.norm(w["conv2.weight"])),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias_max_abs": float(np.max(np.abs(w["head.bias"]))),
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std()),
        "lambda": lambda_,
    }
    return {"weights": w, "diagnostics": diagnostics}
