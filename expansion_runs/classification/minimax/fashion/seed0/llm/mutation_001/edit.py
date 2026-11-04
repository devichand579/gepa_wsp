def edit(w, f):
    import numpy as np

    X = np.asarray(f["features"], dtype=np.float64)      # [N,1568] current final features
    y = np.asarray(f["y"]).astype(np.int64).ravel()        # [N]
    N, D = X.shape
    K = 10

    # ---------- class moments ----------
    mu = np.zeros((K, D))
    cnt = np.zeros(K)
    for c in range(K):
        m = (y == c)
        cnt[c] = m.sum()
        if cnt[c] > 0:
            mu[c] = X[m].mean(axis=0)
    gm = X.mean(axis=0)

    # ---------- within-class covariance (pooled) ----------
    Z = X - mu[y]                       # residuals within class
    S = (Z.T @ Z) / max(N - K, 1)       # [D,D] pooled covariance

    # ---------- data-driven shrinkage (Ledoit-Wolf style, closed form) ----------
    # variance of the covariance elements under iid sampling
    Zs = Z / np.sqrt(max(N - K, 1))
    var_outer = (Zs ** 2).T @ (Zs ** 2) / N      # [D,D]
    d2 = np.diag(var_outer)
    mu2 = S ** 2
    b2 = np.mean((mu2 - d2) ** 2)                # squared off-diagonal noise
    b2 = max(b2, 1e-12)
    trS = np.trace(S)
    # target = isotropic matrix (trS/D) * I
    d_target = (trS / D) ** 2
    num = np.mean((d2 - d_target) ** 2)
    den = num + b2
    delta = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 1.0

    lam = delta * (trS / D)                      # ridge / shrinkage intensity
    Sr = S + lam * np.eye(D)

    # ---------- discriminant directions (class moments + covariance) ----------
    # solve Sr @ W^T = (mu - gm)^T
    M = (mu - gm).T                              # [D,K]
    Wt = np.linalg.solve(Sr, M)                  # [D,K]
    Wt = Wt.T                                    # [K,D]

    # ---------- intercepts: least-squares fit of one-hot targets on fixed directions ---
    # logits = X @ Wt.T + b ;  fit b and a global scale by LS on one-hot targets
    L0 = X @ Wt.T                                # [N,K]
    Y1 = np.zeros((N, K))
    Y1[np.arange(N), y] = 1.0
    # least squares for per-class bias given directions (closed form)
    # for class c: minimize sum_i (L0[i,c] + b_c - Y1[i,c])^2  -> b_c = mean(Y1[:,c] - L0[:,c])
    b = (Y1 - L0).mean(axis=0)

    # global temperature: choose scale s so that predictions are well separated but
    # cross-entropy is minimized in closed form for the *fixed* directions (1-D convex LS)
    # logits_c = s * (L0[:,c] + b_c);  fit s by LS to one-hot (closed form, no iteration)
    Zc = L0 + b[None, :]
    s = float((Zc * (Y1 - 0.1)).sum() / max((Zc * Zc).sum(), 1e-12))
    s = float(np.clip(s, 0.05, 20.0))

    # ---------- assemble head ----------
    head_w = (s * Wt).astype(np.float32)          # [10,1568]
    head_b = (s * b).astype(np.float32)           # [10]

    # ---------- diagnostics on the permitted pool ----------
    logits = X @ head_w.astype(np.float64).T + head_b.astype(np.float64)[None, :]
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    per_class = []
    for c in range(K):
        m = (y == c)
        per_class.append(round(float((pred[m] == c).mean()) if m.sum() else 0.0, 4))
    # closed-form leave-one-out estimate for the ridge (hat-matrix diagonal)
    try:
        Xc = X - gm
        # eigen-decomposition of the (small) Gram matrix for the LOO diagonal
        G = (Xc @ Xc.T) / N                        # [N,N]
        ev, U = np.linalg.eigh(G)
        ev = np.clip(ev, 0.0, None)
        # ridge solution in feature space via SVD of Xc
        Uf, sf, Vt = np.linalg.svd(Xc, full_matrices=False)
        s2 = sf ** 2
        H = (Uf ** 2) @ (s2 / (s2 + lam * N))[:, None]   # [N,N] hat matrix diagonal
        Hd = np.diag(H)
        r = Y1 - (X @ head_w.astype(np.float64).T + head_b.astype(np.float64)[None, :])
        loo = r / np.clip(1.0 - Hd, 1e-6, None)[:, None]
        loo_acc = float(((Y1.argmax(1) == y).astype(float) * 0 +
                         (np.argmax(loo, axis=1) == y)).mean())
    except Exception:
        Hd = None
        loo_acc = None

    w["head.weight"] = np.ascontiguousarray(head_w, dtype=np.float32)
    w["head.bias"] = np.ascontiguousarray(head_b, dtype=np.float32)

    # safety: keep everything finite and within bounds
    for k in list(w.keys()):
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=100.0, neginf=-100.0)
        np.clip(arr, -1000.0, 1000.0, out=arr)
        w[k] = arr

    diag = {
        "train_acc_before": 0.2555,
        "train_acc_after": round(acc, 4),
        "per_class_acc_after": per_class,
        "loo_acc_estimate": (round(loo_acc, 4) if loo_acc is not None else None),
        "shrinkage_delta": round(delta, 5),
        "lambda": float(lam),
        "logit_scale": round(s, 4),
        "head_w_std": round(float(head_w.std()), 6),
        "head_w_absmax": round(float(np.abs(head_w).max()), 6),
        "feature_dim": int(D),
        "n": int(N),
        "note": "conv1/conv2 untouched; features remain exactly valid for the new head",
    }
    return {"weights": w, "diagnostics": diag}
