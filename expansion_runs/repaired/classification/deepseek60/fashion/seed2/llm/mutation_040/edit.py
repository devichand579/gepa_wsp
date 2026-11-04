def edit(w, f):
    import numpy as np

    # Extract full-pool features and labels
    F = f["features"]          # (2000, 1568)
    y = f["y"]                 # (2000,)
    W = w["head.weight"].copy()
    b = w["head.bias"].copy()

    # Current logits and predictions
    L = F @ W.T + b            # (2000, 10)
    pred = np.argmax(L, axis=1)

    wrong = np.where(pred != y)[0]

    diag = {
        "n_wrong_before": int(len(wrong)),
        "max_abs_W_before": float(np.max(np.abs(W))),
        "fixed": []
    }

    # Process each misclassified sample
    for i in wrong:
        t = int(y[i])
        p = int(pred[i])
        if t == p:
            continue

        h = F[i]               # (1568,)
        d = F @ h              # (2000,)
        norm2 = float(h @ h)
        if norm2 == 0.0:
            continue

        N = L.shape[0]
        y_exp = y[:, None]                     # (N,1)
        c_exp = np.arange(10)[None, :]         # (1,10)
        mask_valid = (c_exp != y_exp)          # (N,10)

        # M[j,c] = L[j, y[j]] - L[j, c]
        M = L[np.arange(N), y][:, None] - L    # (N,10)

        # coeff = d[j] * ( (y[j]==t) - (c==t) - (y[j]==p) + (c==p) )
        y_eq_t = (y_exp == t).astype(np.float32)
        c_eq_t = (c_exp == t).astype(np.float32)
        y_eq_p = (y_exp == p).astype(np.float32)
        c_eq_p = (c_exp == p).astype(np.float32)
        coeff = d[:, None] * (y_eq_t - c_eq_t - y_eq_p + c_eq_p)  # (N,10)

        # Lower bound from the misclassified sample i: need margin >= 1.0
        M_target = 1.0
        M_i = M[i]
        coeff_i = coeff[i]
        valid_c = np.arange(10) != t
        alpha_lo = np.max((M_target - M_i[valid_c]) / coeff_i[valid_c])

        # Upper bound from all other samples: need margins > 0
        j_mask = np.ones(N, dtype=bool)
        j_mask[i] = False
        valid_mask = mask_valid & j_mask[:, None]
        upper_mask = valid_mask & (coeff < 0)
        if np.any(upper_mask):
            bounds = M[upper_mask] / (-coeff[upper_mask])
            alpha_hi = np.min(bounds)
        else:
            alpha_hi = np.inf

        if alpha_lo < alpha_hi:
            # Pick a safe step inside the feasible interval
            if np.isfinite(alpha_hi):
                alpha = alpha_lo + 0.5 * (alpha_hi - alpha_lo)
            else:
                alpha = alpha_lo + 1.0
            alpha = min(alpha, 100.0)   # avoid huge weights

            # Apply rank-1 update
            W[t] += alpha * h
            W[p] -= alpha * h

            # Update logits for all samples
            L[:, t] += alpha * d
            L[:, p] -= alpha * d

            # Recompute predictions
            pred = np.argmax(L, axis=1)

            diag["fixed"].append({
                "idx": int(i),
                "true": t,
                "pred": p,
                "alpha": float(alpha),
                "alpha_lo": float(alpha_lo),
                "alpha_hi": float(alpha_hi) if np.isfinite(alpha_hi) else None,
                "norm2": norm2,
                "margin_before": float(M_i[p]),
                "margin_after": float(M_i[p] + alpha * coeff_i[p])
            })
        else:
            diag["fixed"].append({
                "idx": int(i),
                "true": t,
                "pred": p,
                "error": "no feasible alpha",
                "alpha_lo": float(alpha_lo),
                "alpha_hi": float(alpha_hi) if np.isfinite(alpha_hi) else None
            })

    # Enforce ±1000 bound
    W = np.clip(W, -1000.0, 1000.0)
    b = np.clip(b, -1000.0, 1000.0)

    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # Final diagnostics
    L_final = F @ W.T + b
    pred_final = np.argmax(L_final, axis=1)
    acc = float(np.mean(pred_final == y))
    diag["n_wrong_after"] = int(np.sum(pred_final != y))
    diag["acc_after"] = acc
    diag["max_abs_W_after"] = float(np.max(np.abs(W)))

    return {"weights": w, "diagnostics": diag}
