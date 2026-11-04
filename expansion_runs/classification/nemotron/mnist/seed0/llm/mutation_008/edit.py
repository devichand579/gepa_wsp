import numpy as np

def edit(w, f):
    """
    Update fc2.weight and fc2.bias by ridge regression on the full feedback pool.
    Returns a dict with the modified weight dictionary and a small diagnostics block.
    """
    # -------------------------------------------------
    # 1. Gather feedback data (read‑only)
    # -------------------------------------------------
    X = f["features"]                # (2000, 128)  post‑fc1 ReLU
    y = f["y"]                       # (2000,) int64
    # one‑hot encode labels → (2000, 10)
    Y = np.eye(10, dtype=np.float32)[y]

    # -------------------------------------------------
    # 2. Closed‑form ridge regression:  W_b = (Xbᵀ Xb + λI)^{-1} Xbᵀ Y
    # -------------------------------------------------
    # Add a bias column to X
    Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])   # (2000, 129)
    lam = 0.1                                                     # ridge strength
    I = np.eye(Xb.shape[1], dtype=np.float32) * lam

    # Normal‑equation matrices
    A = Xb.T @ Xb + I          # (129, 129)
    b_vec = Xb.T @ Y           # (129, 10)

    # Solve without explicit inverse (more stable)
    W_b = np.linalg.solve(A, b_vec)   # (129, 10)

    # -------------------------------------------------
    # 3. Split into weight matrix and bias vector
    # -------------------------------------------------
    # W_b layout: first 128 rows = weight (out=10, in=128) transposed later,
    # last row = bias.
    new_weight = W_b[:128, :].T          # (10, 128)  – fc2.weight
    new_bias   = W_b[128, :]             # (10,)     – fc2.bias

    # -------------------------------------------------
    # 4. Insert updated parameters back into w
    # -------------------------------------------------
    w["fc2.weight"] = new_weight.astype(np.float32)
    w["fc2.bias"]   = new_bias.astype(np.float32)

    # -------------------------------------------------
    # 5. Diagnostics – evaluate on the supplied reflection batch
    # -------------------------------------------------
    # Re‑compute batch activations using the *unchanged* fc1 and new fc2.
    fc1_w = w["fc1.weight"]          # (128, 784)
    fc1_b = w["fc1.bias"]            # (128,)
    batch_x = f["batch.x"]           # (128, 1, 28, 28)
    batch_x_flat = batch_x.reshape(128, 784)
    batch_features = np.maximum(0.0, batch_x_flat @ fc1_w.T + fc1_b)   # (128, 128)

    logits = batch_features @ w["fc2.weight"].T + w["fc2.bias"]        # (128, 10)
    preds = np.argmax(logits, axis=1)
    batch_acc = float(np.mean(preds == f["batch.y"].astype(np.int64)))

    w_norm = float(np.linalg.norm(new_weight, ord='fro'))
    b_norm = float(np.linalg.norm(new_bias))

    diagnostics = {
        "batch_accuracy": batch_acc,
        "fc2_weight_frobenius_norm": w_norm,
        "fc2_bias_norm": b_norm,
        "ridge_lambda": float(lam)
    }

    # -------------------------------------------------
    # 6. Return
    # -------------------------------------------------
    return {"weights": w, "diagnostics": diagnostics}
