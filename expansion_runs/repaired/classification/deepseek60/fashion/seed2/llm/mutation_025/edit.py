def edit(w, f):
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    logits = f["logits"].astype(np.float64)
    N, C = logits.shape
    pred = np.argmax(logits, axis=1)
    mis = np.where(pred != y)[0]
    if len(mis) == 0:
        return w

    # Pick the misclassified sample with the most negative margin
    margins = logits[mis, y[mis]] - logits[mis, pred[mis]]
    i = mis[np.argmin(margins)]
    c = int(y[i])
    k = int(pred[i])
    x = X[i]
    dots = X @ x  # [N]

    # Alpha required to flip sample i
    M_i = logits[i, c] - logits[i, k]
    alpha_req = -M_i / (2.0 * dots[i] + 1e-12)

    # Compute maximum alpha allowed without flipping any correct sample
    u = np.zeros(C)
    u[c] = 1.0
    u[k] = -1.0
    M = logits[np.arange(N), y][:, None] - logits  # [N, C]
    coeff = u[None, :] - u[y][:, None]              # [N, C]
    denom = dots[:, None] * coeff
    mask = np.arange(C)[None, :] != y[:, None]
    valid = mask & (denom > 0)
    ratios = np.full((N, C), np.inf)
    ratios[valid] = M[valid] / denom[valid]
    alpha_max = np.min(ratios)

    if alpha_max <= alpha_req:
        # Cannot fix without flipping others; return unchanged
        return w

    # Choose alpha slightly above required, but below max
    alpha = alpha_req * 1.05 + 1e-6
    if alpha >= alpha_max:
        alpha = (alpha_req + alpha_max) / 2.0

    # Apply rank-1 update to head weights
    w["head.weight"][c] = (w["head.weight"][c].astype(np.float64) + alpha * x).astype(np.float32)
    w["head.weight"][k] = (w["head.weight"][k].astype(np.float64) - alpha * x).astype(np.float32)

    # Clip to allowed range
    w["head.weight"] = np.clip(w["head.weight"], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "fixed_index": int(i),
        "true_class": c,
        "pred_class": k,
        "alpha_req": float(alpha_req),
        "alpha_max": float(alpha_max),
        "alpha_used": float(alpha),
        "initial_misclassified": mis.tolist(),
    }
    return {"weights": w, "diagnostics": diagnostics}
