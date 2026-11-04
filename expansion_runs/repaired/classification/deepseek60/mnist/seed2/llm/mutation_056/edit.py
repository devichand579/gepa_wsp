import numpy as np

def edit(w, f):
    # Read current weights
    fc1_w = w["fc1.weight"].astype(np.float64)
    fc1_b = w["fc1.bias"].astype(np.float64)
    fc2_w = w["fc2.weight"].astype(np.float64)
    fc2_b = w["fc2.bias"].astype(np.float64)

    # Feedback features and labels
    H = f["features"].astype(np.float64)          # [2000, 128]
    y = f["y"].astype(np.int64)                   # [2000]
    N, D = H.shape
    C = 10

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Current readout predictions
    logits_cur = H @ fc2_w.T + fc2_b
    pred_cur = np.argmax(logits_cur, axis=1)
    acc_cur = np.mean(pred_cur == y)
    # Cross-entropy for current readout
    logits_cur_shift = logits_cur - logits_cur.max(axis=1, keepdims=True)
    exp_cur = np.exp(logits_cur_shift)
    probs_cur = exp_cur / exp_cur.sum(axis=1, keepdims=True)
    ce_cur = -np.mean(np.log(np.clip(probs_cur[np.arange(N), y], 1e-12, 1.0)))

    # Ridge regression: solve for W such that H @ W.T + b ~= Y
    # Augment with bias column
    H_aug = np.concatenate([H, np.ones((N, 1), dtype=np.float64)], axis=1)  # [N, D+1]
    # Ridge solution: (H_aug^T H_aug + alpha I) W_aug = H_aug^T Y
    # We do not regularize the bias term (last column)
    alpha = 1e-3
    A = H_aug.T @ H_aug
    reg = alpha * np.eye(D + 1, dtype=np.float64)
    reg[-1, -1] = 0.0
    A_reg = A + reg
    B = H_aug.T @ Y
    W_aug = np.linalg.solve(A_reg, B)  # [D+1, C]

    fc2_w_new = W_aug[:D, :].T          # [C, D]
    fc2_b_new = W_aug[D, :]             # [C]

    # Evaluate new readout on full pool
    logits_new = H @ fc2_w_new.T + fc2_b_new
    pred_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(pred_new == y)
    logits_new_shift = logits_new - logits_new.max(axis=1, keepdims=True)
    exp_new = np.exp(logits_new_shift)
    probs_new = exp_new / exp_new.sum(axis=1, keepdims=True)
    ce_new = -np.mean(np.log(np.clip(probs_new[np.arange(N), y], 1e-12, 1.0)))

    # Choose better readout: higher accuracy, then lower CE
    if (acc_new > acc_cur) or (acc_new == acc_cur and ce_new < ce_cur):
        w["fc2.weight"] = fc2_w_new.astype(np.float32)
        w["fc2.bias"] = fc2_b_new.astype(np.float32)
        chosen = "ridge"
        final_acc = float(acc_new)
        final_ce = float(ce_new)
    else:
        chosen = "current"
        final_acc = float(acc_cur)
        final_ce = float(ce_cur)

    # Ensure finite and bounded
    for k in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]:
        arr = w[k]
        arr = np.nan_to_num(arr, nan=0.0, posinf=1000.0, neginf=-1000.0)
        arr = np.clip(arr, -1000.0, 1000.0)
        w[k] = arr.astype(np.float32)

    diagnostics = {
        "chosen": chosen,
        "acc_current": float(acc_cur),
        "ce_current": float(ce_cur),
        "acc_ridge": float(acc_new),
        "ce_ridge": float(ce_new),
        "final_acc": final_acc,
        "final_ce": final_ce,
        "alpha": alpha,
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
