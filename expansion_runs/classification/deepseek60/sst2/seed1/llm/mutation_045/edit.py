import numpy as np

def edit(w, f):
    # ---------- read feedback ----------
    X = np.asarray(f["x"])          # [N,64] int64
    Y = np.asarray(f["y"]).astype(np.int64)   # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]  # vocab size (2000)
    D = w["embed.weight"].shape[1]  # 32

    # ---------- count matrix on full pool ----------
    C = np.zeros((N, V), dtype=np.float64)
    for t in range(T):
        ids = X[:, t]
        np.add.at(C, (np.arange(N), ids), 1.0)
    L = C.sum(axis=1, keepdims=True)
    L[L == 0] = 1.0
    P = C / L                        # normalized counts  [N,V]

    # ---------- closed-form ridge with k-fold CV for lambda ----------
    ys = (2.0 * Y - 1.0).astype(np.float64)          # {-1,+1}
    Pa = np.concatenate([P, np.ones((N, 1))], axis=1)  # add bias column
    d = Pa.shape[1]

    # Gram matrix (shared across folds)
    G = Pa.T @ Pa
    b = Pa.T @ ys

    # closed-form k-fold CV over a small lambda grid
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    K = 5
    folds = np.array_split(perm, K)
    lam_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    best_lam, best_acc = lam_grid[0], -1.0
    for lam in lam_grid:
        accs = []
        for k in range(K):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(K) if j != k])
            Gtr = Pa[tr].T @ Pa[tr]
            btr = Pa[tr].T @ ys[tr]
            A = Gtr + lam * np.eye(d)
            try:
                ua = np.linalg.solve(A, btr)
            except np.linalg.LinAlgError:
                ua = np.linalg.lstsq(A, btr, rcond=None)[0]
            pred = np.sign(Pa[te] @ ua)
            pred[pred == 0] = 1.0
            accs.append(np.mean(pred == ys[te]))
        macc = float(np.mean(accs))
        if macc > best_acc:
            best_acc, best_lam = macc, lam

    # final fit on all data
    A = G + best_lam * np.eye(d)
    try:
        ua = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        ua = np.linalg.lstsq(A, b, rcond=None)[0]
    u = ua[:V].astype(np.float64)
    bias = float(ua[V])

    # ---------- realize u in embed + head ----------
    # head[1] = s * e0  =>  logit1 = s*pooled[0] + bias, pooled[0] = (1/L) C (u/s)
    # need |u_j / s| <= sqrt(D-1)  (feasibility of zero-mean/unit-var vector)
    umax = float(np.max(np.abs(u))) if u.size else 0.0
    s = max(umax / np.sqrt(D - 1.0), 1e-6)
    # keep magnitudes sane
    s = min(s, 900.0)
    c = u / s                                   # target first component per token
    c = np.clip(c, -np.sqrt(D - 1.0), np.sqrt(D - 1.0))

    # build zero-mean, unit-variance rows with row[0] = c_j
    # use dims 1 and 2 to absorb mean/variance, rest equal
    v = np.zeros((V, D), dtype=np.float64)
    v[:, 0] = c
    # solve for d_j (dim1) so that sum=0 and sumsq=D
    # dims 2..D-1 share value r_j = (-c_j - d_j)/(D-2)
    m = D - 2
    # (c+d)^2/m + d^2 = D - c^2  ->  solve quadratic in d
    # (c^2 + 2cd + d^2)/m + d^2 = D - c^2
    # d^2 (1 + 1/m) + d (2c/m) + c^2/m + c^2 - D = 0
    Aq = 1.0 + 1.0 / m
    Bq = 2.0 * c / m
    Cq = c * c / m + c * c - D
    disc = Bq * Bq - 4.0 * Aq * Cq
    disc = np.maximum(disc, 0.0)
    d1 = (-Bq + np.sqrt(disc)) / (2.0 * Aq)
    v[:, 1] = d1
    r = (-c - d1) / m
    v[:, 2:] = r[:, None]

    # renormalize rows to exactly zero-mean / unit-variance (numerical safety)
    v = v - v.mean(axis=1, keepdims=True)
    nrm = np.sqrt((v * v).sum(axis=1, keepdims=True))
    nrm[nrm == 0] = 1.0
    v = v / nrm * np.sqrt(D)
    # restore first component (rescale so v[:,0] matches c as closely as possible)
    # (renormalization preserves direction; recompute c_eff)
    c_eff = v[:, 0].copy()

    # embed rows = v (already zero-mean/unit-var => LN(v) = v)
    emb = v.astype(np.float32)
    # head: row0 zeros, row1 = s * e0 ; bias = [0, bias]
    head = np.zeros((2, D), dtype=np.float32)
    head[1, 0] = np.float32(s)
    hbias = np.zeros(2, dtype=np.float32)
    hbias[1] = np.float32(bias)

    # ---------- write back (preserve names/shapes) ----------
    w["embed.weight"] = emb
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    for name in list(w.keys()):
        if name.startswith("block0.") or name.startswith("block1."):
            w[name] = np.zeros_like(w[name])
    # norms stay identity
    for nm in ["norm.weight", "block0.norm1.weight", "block0.norm2.weight",
               "block1.norm1.weight", "block1.norm2.weight"]:
        if nm in w:
            w[nm] = np.ones_like(w[nm])
    for nm in ["norm.bias", "block0.norm1.bias", "block0.norm2.bias",
               "block1.norm1.bias", "block1.norm2.bias"]:
        if nm in w:
            w[nm] = np.zeros_like(w[nm])
    w["head.weight"] = head
    w["head.bias"] = hbias

    # ---------- diagnostics ----------
    # in-sample accuracy of the realized classifier
    score = P @ u + bias
    pred = (score > 0).astype(np.int64)
    train_acc = float(np.mean(pred == Y))

    diag = {
        "method": "closed_form_ridge_linear_classifier_on_normalized_counts",
        "best_lambda": float(best_lam),
        "cv_accuracy": float(best_acc),
        "train_accuracy": train_acc,
        "u_absmax": float(np.max(np.abs(u))),
        "bias": float(bias),
        "scale_s": float(s),
        "mean_len": float(L.mean()),
        "embed_absmax": float(np.max(np.abs(emb))),
    }
    return {"weights": w, "diagnostics": diag}
