import numpy as np

def edit(w, f):
    X = np.asarray(f["features"], dtype=np.float64)      # [N,32] pooled features
    y = np.asarray(f["y"], dtype=np.float64)             # [N]
    N, D = X.shape

    # ---- closed-form ridge fit of head on current pooled features ----
    # target: +1 for positive class (1), -1 for negative (0)
    t = 2.0 * y - 1.0
    Xm = X.mean(axis=0, keepdims=True)
    Xc = X - Xm
    # normal equations with intercept: augment with column of ones
    Xa = np.concatenate([Xc, np.ones((N, 1))], axis=1)   # [N, D+1]
    lam = 1e-2
    A = Xa.T @ Xa + lam * np.eye(D + 1)
    A[-1, -1] -= lam                                     # don't regularize bias
    b = Xa.T @ t
    sol = np.linalg.solve(A, b)                          # [D+1]
    w_dir = sol[:D]                                      # direction on centered features
    w_bias = sol[D]

    # Convert to head.weight / head.bias in the network's convention:
    # logits = head.weight @ pooled + head.bias,  argmax over 2 classes.
    # We want logits[1]-logits[0] = 2*(w_dir·X + w_bias)  (so sign matches t)
    # Choose head.weight[1] = w_dir, head.weight[0] = -w_dir,
    #        head.bias[1]   = w_bias, head.bias[0] = -w_bias.
    # Then logits[1]-logits[0] = 2*(w_dir·X + w_bias). Good.
    head_w = np.zeros((2, D), dtype=np.float32)
    head_b = np.zeros((2,), dtype=np.float32)
    head_w[1] = w_dir.astype(np.float32)
    head_w[0] = -w_dir.astype(np.float32)
    head_b[1] = np.float32(w_bias)
    head_b[0] = np.float32(-w_bias)

    # sanity: clip to allowed range
    head_w = np.clip(head_w, -1000.0, 1000.0).astype(np.float32)
    head_b = np.clip(head_b, -1000.0, 1000.0).astype(np.float32)

    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # diagnostics
    logits = X @ head_w.T + head_b
    pred = np.argmax(logits, axis=1)
    acc = float((pred == y).mean())
    diag = {
        "method": "closed_form_ridge_head_on_current_features",
        "lam": lam,
        "train_acc_head": acc,
        "w_dir_norm": float(np.linalg.norm(w_dir)),
        "w_bias": float(w_bias),
    }
    return {"weights": w, "diagnostics": diag}
