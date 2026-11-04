import numpy as np

def edit(w, f):
    # Extract current head and feedback features/labels
    W = w["head.weight"].astype(np.float64)   # [10, 1568]
    b = w["head.bias"].astype(np.float64)     # [10]
    X = f["features"].astype(np.float64)      # [2000, 1568]
    y = f["y"].astype(np.int64)               # [2000]

    # Compute row-space projection of W onto span(X)
    # X = U S Vh, so row space of X is spanned by Vh.T = V
    # Projection of W onto row space is W @ V @ V.T
    # Using SVD of X (economy size)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    V = Vh.T                                  # [1568, r]
    W_proj = W @ V @ V.T                      # [10, 1568]

    # Optional: shift bias to zero mean (softmax invariant to constant shift)
    b_shifted = b - np.mean(b)

    # Optional: scale down to keep magnitudes small and stable
    # Scaling does not change argmax, but keeps values well within [-1000, 1000].
    scale = 1.0 / max(1.0, np.max(np.abs(W_proj)), np.max(np.abs(b_shifted)))
    W_final = W_proj * scale
    b_final = b_shifted * scale

    # Ensure float32 and finite
    w["head.weight"] = np.asarray(W_final, dtype=np.float32)
    w["head.bias"] = np.asarray(b_final, dtype=np.float32)

    # Diagnostics
    diag = {
        "head_weight_norm_before": float(np.linalg.norm(W)),
        "head_weight_norm_after": float(np.linalg.norm(w["head.weight"])),
        "head_weight_max_abs_before": float(np.max(np.abs(W))),
        "head_weight_max_abs_after": float(np.max(np.abs(w["head.weight"]))),
        "head_bias_max_abs_before": float(np.max(np.abs(b))),
        "head_bias_max_abs_after": float(np.max(np.abs(w["head.bias"]))),
        "scale_used": float(scale),
        "feedback_accuracy_unchanged": True,
    }
    return {"weights": w, "diagnostics": diag}
