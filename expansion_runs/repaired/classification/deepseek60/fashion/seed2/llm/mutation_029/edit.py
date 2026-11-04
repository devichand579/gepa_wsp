import numpy as np

def edit(w, f):
    # Extract current head parameters
    W = w["head.weight"]          # shape (10, 1568)
    b = w["head.bias"]            # shape (10,)
    features = f["features"]      # shape (2000, 1568)
    y = f["y"]                    # shape (2000,)

    # Work on a copy so we can safely iterate
    W_new = W.copy()
    b_new = b.copy()

    max_iter = 20
    for _ in range(max_iter):
        # Recompute logits and predictions
        logits = features @ W_new.T + b_new
        preds = np.argmax(logits, axis=1)
        mis = np.where(preds != y)[0]
        if len(mis) == 0:
            break

        i = mis[0]
        t = int(y[i])
        p = int(preds[i])
        x_i = features[i]
        S = float(x_i @ x_i)
        if S <= 0:
            break

        logit_i = logits[i]

        # Minimum alpha to make true class t the argmax for sample i
        alpha_req = 0.0
        for c in range(10):
            if c == t:
                continue
            denom = 2.0 * S if c == p else S
            if denom > 0:
                req = (logit_i[c] - logit_i[t]) / denom
                if req > alpha_req:
                    alpha_req = req
        alpha_req = max(0.0, alpha_req) + 1e-6

        # Maximum alpha allowed by margins of all currently correct samples
        alpha_max = np.inf
        S_j = features @ x_i                      # shape (2000,)
        correct = (np.argmax(logits, axis=1) == y)
        for j in range(len(y)):
            if not correct[j]:
                continue
            yj = int(y[j])
            for c in range(10):
                if c == yj:
                    continue
                m = logits[j, yj] - logits[j, c]
                if m <= 0:
                    continue
                coeff = (1 if yj == t else 0) - (1 if yj == p else 0) \
                        - (1 if c == t else 0) + (1 if c == p else 0)
                prod = S_j[j] * coeff
                if prod < 0:
                    bound = m / (-prod)
                    if bound < alpha_max:
                        alpha_max = bound

        # Maximum alpha allowed by ±1000 weight bounds
        for k in range(len(x_i)):
            xk = x_i[k]
            if xk > 0:
                alpha_max = min(alpha_max, (1000.0 - W_new[t, k]) / xk)
                alpha_max = min(alpha_max, (1000.0 + W_new[p, k]) / xk)
            elif xk < 0:
                alpha_max = min(alpha_max, (-1000.0 - W_new[t, k]) / xk)
                alpha_max = min(alpha_max, (W_new[p, k] - 1000.0) / xk)

        if alpha_max <= 0:
            break

        if alpha_max > alpha_req:
            alpha = alpha_req + 0.5 * (alpha_max - alpha_req)
        else:
            # Cannot fix without breaking others; try a tiny step just below the bound
            alpha = 0.99 * alpha_max
            if alpha <= 0:
                break

        # Apply rank-1 update
        W_new[t] += alpha * x_i
        W_new[p] -= alpha * x_i

    # Write back
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)

    # Diagnostics
    logits_final = features @ w["head.weight"].T + w["head.bias"]
    preds_final = np.argmax(logits_final, axis=1)
    acc = float(np.mean(preds_final == y))
    mis_final = np.where(preds_final != y)[0].tolist()
    diagnostics = {
        "final_accuracy": acc,
        "final_misclassified": mis_final,
        "max_abs_head_weight": float(np.max(np.abs(w["head.weight"]))),
        "max_abs_head_bias": float(np.max(np.abs(w["head.bias"]))),
    }
    return {"weights": w, "diagnostics": diagnostics}
