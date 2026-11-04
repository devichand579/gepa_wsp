import numpy as np

def edit(w, f):
    # Extract arrays
    features = f["features"]          # [2000, 1568]
    y = f["y"]                        # [2000]
    logits = f["logits"]              # [2000, 10]
    probs = f["probs"]                # [2000, 10]
    N = features.shape[0]
    C = 10

    # Current head weights
    W = w["head.weight"].copy()       # [10, 1568]
    b = w["head.bias"].copy()         # [10]

    # Recompute logits from features to be exact
    logits = features @ W.T + b
    pred = np.argmax(logits, axis=1)
    mis = np.where(pred != y)[0]

    diag = {
        "initial_accuracy": float(np.mean(pred == y)),
        "initial_misclassified": mis.tolist(),
        "initial_cross_entropy": float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12))),
    }

    if len(mis) == 0:
        diag["status"] = "no_misclassified"
        return {"weights": w, "diagnostics": diag}

    # Fix each misclassified sample sequentially (usually only one)
    for idx in mis:
        x_i = features[idx]            # [1568]
        c = int(y[idx])
        k = int(pred[idx])
        L = logits[idx]                # [10]

        # Required alpha to flip this sample
        s = float(np.dot(x_i, x_i))
        if s <= 0:
            continue
        req = []
        for j in range(C):
            if j == c:
                continue
            if j == k:
                req.append((L[k] - L[c]) / (2.0 * s))
            else:
                req.append((L[j] - L[c]) / s)
        alpha_req = max(req) + 1e-6

        # Maximum alpha allowed by all currently correct samples
        s_all = features @ x_i         # [N]
        c_j = y
        L_all = logits

        # d_cj for each sample j
        d_cj = np.zeros(N, dtype=np.float64)
        d_cj[c_j == c] = s_all[c_j == c]
        d_cj[c_j == k] = -s_all[c_j == k]

        # Bounds from competing class c
        denom_c = d_cj - s_all
        mask_c = (c_j != c) & (pred == y) & (denom_c < 0)
        bound_c = np.full(N, np.inf)
        if np.any(mask_c):
            bound_c[mask_c] = (L_all[mask_c, c] - L_all[mask_c, c_j[mask_c]]) / denom_c[mask_c]

        # Bounds from competing class k
        denom_k = d_cj + s_all
        mask_k = (c_j != k) & (pred == y) & (denom_k < 0)
        bound_k = np.full(N, np.inf)
        if np.any(mask_k):
            bound_k[mask_k] = (L_all[mask_k, k] - L_all[mask_k, c_j[mask_k]]) / denom_k[mask_k]

        alpha_max = min(bound_c.min(), bound_k.min())

        # Weight magnitude bound ±1000
        W_c = W[c]
        W_k = W[k]
        # For row c: W_c + alpha * x_i
        pos_c = x_i > 0
        neg_c = x_i < 0
        bound_w_c = np.inf
        if np.any(pos_c):
            bound_w_c = min(bound_w_c, np.min((1000.0 - W_c[pos_c]) / x_i[pos_c]))
        if np.any(neg_c):
            bound_w_c = min(bound_w_c, np.min((-1000.0 - W_c[neg_c]) / x_i[neg_c]))
        # For row k: W_k - alpha * x_i
        bound_w_k = np.inf
        if np.any(pos_c):
            bound_w_k = min(bound_w_k, np.min((W_k[pos_c] + 1000.0) / x_i[pos_c]))
        if np.any(neg_c):
            bound_w_k = min(bound_w_k, np.min((W_k[neg_c] - 1000.0) / x_i[neg_c]))
        alpha_max = min(alpha_max, bound_w_c, bound_w_k)

        if not np.isfinite(alpha_max) or alpha_max <= alpha_req:
            # Fallback: use a small safe step if possible
            alpha_used = max(alpha_req, 1e-6)
        else:
            alpha_used = alpha_req + 0.1 * (alpha_max - alpha_req)
            alpha_used = min(alpha_used, alpha_max * 0.99)

        # Apply rank-1 update
        W[c] += alpha_used * x_i
        W[k] -= alpha_used * x_i

        # Recompute logits and predictions
        logits = features @ W.T + b
        pred = np.argmax(logits, axis=1)

        diag[f"fixed_{idx}"] = {
            "true_class": c,
            "pred_class": k,
            "alpha_req": float(alpha_req),
            "alpha_max": float(alpha_max) if np.isfinite(alpha_max) else None,
            "alpha_used": float(alpha_used),
        }

    # Clip to ±1000 for safety
    W = np.clip(W, -1000.0, 1000.0)

    # Final metrics
    logits = features @ W.T + b
    pred = np.argmax(logits, axis=1)
    probs_new = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_new /= probs_new.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(probs_new[np.arange(N), y] + 1e-12))

    diag["final_accuracy"] = float(np.mean(pred == y))
    diag["final_cross_entropy"] = float(ce)
    diag["final_misclassified"] = np.where(pred != y)[0].tolist()
    diag["max_abs_head_weight"] = float(np.max(np.abs(W)))
    diag["max_abs_head_bias"] = float(np.max(np.abs(b)))

    # Write back
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    return {"weights": w, "diagnostics": diag}
