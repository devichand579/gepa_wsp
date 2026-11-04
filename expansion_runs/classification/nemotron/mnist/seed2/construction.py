import numpy as np

def edit(w, f):
    # --- keep copies of original weights for diagnostics ---
    old_fc1_w = w["fc1.weight"].copy()
    old_fc1_b = w["fc1.bias"].copy()
    old_fc2_w = w["fc2.weight"].copy()
    old_fc2_b = w["fc2.bias"].copy()

    # -------------------------------------------------
    # 1) fc2: ridge regression on the full feedback pool
    # -------------------------------------------------
    N = f["features"].shape[0]                     # 2000
    Y_onehot = np.zeros((N, 10), dtype=np.float32)
    Y_onehot[np.arange(N), f["y"].astype(np.int64)] = 1.0

    features = f["features"].astype(np.float32)    # [2000, 128]
    lam2 = 0.01
    Gram2 = features.T @ features + lam2 * np.eye(128, dtype=np.float32)   # [128,128]
    T2 = features.T @ Y_onehot                                                # [128,10]
    fc2_weight_T = np.linalg.solve(Gram2, T2)                                # [128,10]
    new_fc2_weight = fc2_weight_T.T                                           # [10,128]

    # bias = mean residual after weight fit
    resid2 = Y_onehot - features @ new_fc2_weight.T
    new_fc2_bias = resid2.mean(axis=0)                                        # [10]

    # -------------------------------------------------
    # 2) fc1: ridge regression on the reflection batch
    # -------------------------------------------------
    X = f["layer.fc1.input0"].astype(np.float32)          # [128, 784]
    O = f["layer.fc1.output"].astype(np.float32)          # [128, 128]
    lam1 = 0.01
    Gram1 = X.T @ X + lam1 * np.eye(784, dtype=np.float32)   # [784,784]
    T1 = X.T @ O                                                # [784,128]
    fc1_weight_T = np.linalg.solve(Gram1, T1)               # [784,128]
    new_fc1_weight = fc1_weight_T.T                           # [128,784]

    resid1 = O - X @ fc1_weight_T.T
    new_fc1_bias = resid1.mean(axis=0)                        # [128]

    # -------------------------------------------------
    # 3) assign updated weights back into w (in‑place)
    # -------------------------------------------------
    w["fc1.weight"][:] = new_fc1_weight
    w["fc1.bias"][:] = new_fc1_bias
    w["fc2.weight"][:] = new_fc2_weight
    w["fc2.bias"][:] = new_fc2_bias

    # -------------------------------------------------
    # 4) diagnostics (JSON‑compatible, <24000 chars)
    # -------------------------------------------------
    logits_pool = features @ new_fc2_weight.T + new_fc2_bias
    pred_pool = np.argmax(logits_pool, axis=1)
    pool_acc = float((pred_pool == f["y"].astype(np.int64)).mean())

    diag = {
        "pool_accuracy": pool_acc,
        "fc1_weight_norm_old": float(np.linalg.norm(old_fc1_w.astype(np.float64))),
        "fc1_weight_norm_new": float(np.linalg.norm(new_fc1_weight.astype(np.float64))),
        "fc2_weight_norm_old": float(np.linalg.norm(old_fc2_w.astype(np.float64))),
        "fc2_weight_norm_new": float(np.linalg.norm(new_fc2_weight.astype(np.float64))),
        "fc1_bias_mean_change": float(np.mean(np.abs(new_fc1_bias - old_fc1_b))),
        "fc2_bias_mean_change": float(np.mean(np.abs(new_fc2_bias - old_fc2_b))),
        "lambda1": float(lam1),
        "lambda2": float(lam2),
    }

    return {"weights": w, "diagnostics": diag}
