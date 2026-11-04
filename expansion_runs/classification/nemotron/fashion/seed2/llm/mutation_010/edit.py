import numpy as np

def edit(w, f):
    """
    Closed‑form ridge‑regression fit of the head layer using the full
    permitted feedback pool.  All other modules (conv1, conv2) are left
    untouched; only head.weight and head.bias are replaced.
    """
    # 1️⃣ copy current weights so we only modify the head
    w = {k: v.copy() for k, v in w.items()}

    # 2️⃣ pull the feedback arrays
    features = f["features"]          # shape (2000, 1568)
    y        = f["y"]                 # shape (2000,)   int64 labels
    N, D     = features.shape
    C        = 10                     # number of classes

    # 3️⃣ build one‑hot target matrix Y (N × C)
    Y = np.eye(C, dtype=np.float32)[y]   # each row is a one‑hot vector

    # 4️⃣ ridge regularisation (small λ stabilises the solve)
    lam = 1e-3
    XTX = features.T @ features + lam * np.eye(D, dtype=np.float32)   # (D, D)
    XTY = features.T @ Y                  # (D, C)

    # 5️⃣ solve the normal equations  (D, C)  ->  head.weight will be (C, D)
    w_head = np.linalg.solve(XTX, XTY)   # (D, C)

    # 6️⃣ reshape to the network’s expected layout [out, in]
    w["head.weight"] = w_head.T.astype(np.float32)   # (C, D)

    # 7️⃣ compute bias as the mean residual (Y – X·wᵀ) over the pool
    preds = features @ w_head.T          # (N, C) logits before softmax
    bias  = np.mean(Y - preds, axis=0).astype(np.float32)   # (C,)
    w["head.bias"] = bias

    # 8️⃣ diagnostics (tiny JSON‑compatible dict)
    acc = np.mean(np.argmax(preds, axis=1) == y)          # training accuracy
    ce  = -np.mean(np.log(preds[np.arange(N), y] + 1e-12))  # cross‑entropy
    diag = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist()
    }

    return {"weights": w, "diagnostics": diag}
