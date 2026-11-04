import numpy as np

def edit(w, f):
    rng = np.random.RandomState(1234)

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"]).astype(np.int64).ravel()
    N, D = X.shape
    C = 10

    # ---------------- within-class whitening ----------------
    gm = X.mean(axis=0)
    Sw = np.zeros((D, D))
    for c in range(C):
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        Zc = Xc - Xc.mean(axis=0)
        Sw += Zc.T @ Zc
    Sw /= max(N - C, 1)
    tr = np.trace(Sw) / D
    Sw = 0.7 * Sw + 0.3 * tr * np.eye(D)
    evals, evecs = np.linalg.eigh(Sw)
    evals = np.maximum(evals, 1e-8)
    M = (evecs / np.sqrt(evals)).T          # z = M x  (unit within-class var)

    # ---------------- hidden feature construction ----------------
    Z = X @ M.T
    Zc = Z - Z.mean(axis=0)

    # (a) 32 strongest whitened directions -> +/- pairs (64 units)
    Zs = (Zc ** 2).sum(axis=0)
    top = np.argsort(-Zs)[:32]
    dirs = Zc[:, top] / (np.linalg.norm(Zc[:, top], axis=0, keepdims=True) + 1e-12)

    # (b) 64 random whitened hinge directions
    G = rng.randn(D, 64)
    G /= np.linalg.norm(G, axis=0, keepdims=True)
    proj = Zc @ G
    qs = np.linspace(0.25, 1.60, 64)
    thr = np.quantile(proj, 1.0 - qs, axis=0)   # per-unit thresholds

    W1 = np.zeros((128, D))
    b1 = np.zeros(128)
    W1[:64] = dirs
    W1[64:] = (G * (np.linalg.norm(Zc, axis=1).mean())).T
    b1[64:] = thr

    def feats(Xin, W1_, b1_):
        return np.maximum(Xin @ W1_.T + b1_, 0.0)

    # ---------------- shift augmentation (closed-form data expansion) ----------------
    def shifted(Xin, dy, dx):
        out = np.zeros_like(Xin)
        ys0 = slice(max(0, dy), 28 + min(0, dy))
        ys1 = slice(max(0, -dy), 28 + min(0, -dy))
        xs0 = slice(max(0, dx), 28 + min(0, dx))
        xs1 = slice(max(0, -dx), 28 + min(0, -dx))
        out[:, ys0, xs0] = Xin[:, ys1, xs1]
        return out.reshape(len(Xin), 784)

    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    def build(idx, shift_list):
        Xa = X[idx]
        ya = y[idx]
        Xs, ys = [], []
        for (dy, dx) in shift_list:
            Xs.append(shifted(Xa.reshape(-1, 28, 28), dy, dx))
            ys.append(ya)
        Xall = np.concatenate(Xs, axis=0)
        yall = np.concatenate(ys, axis=0)
        H = feats(Xall @ M.T, W1, b1)
        return H, yall

    # ---------------- closed-form ridge readout ----------------
    def fit_ridge(H, yv, lam):
        mu = H.mean(axis=0)
        sd = H.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Hs = (H - mu) / sd
        T = np.zeros((len(yv), C))
        T[np.arange(len(yv)), yv] = 1.0
        A = Hs.T @ Hs + lam * np.eye(Hs.shape[1])
        B = Hs.T @ T
        W2 = np.linalg.solve(A, B)            # (128, 10)
        b2 = T.mean(axis=0) - mu @ W2 / sd    # fold centering/scaling back
        W2 = W2 / sd[:, None]
        return W2, b2

    def acc_logits(H, yv, W2, b2):
        L = H @ W2 + b2
        return float((L.argmax(axis=1) == yv).mean()), L

    # ---------------- model selection on a held-out split (closed-form fits only) ----------------
    perm = rng.permutation(N)
    va_idx = perm[:500]
    tr_idx = perm[500:]

    lams = [0.3, 1.0, 3.0, 10.0, 30.0]
    shift_opts = [[(0, 0)], [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]]
    best = None
    for sh in shift_opts:
        Htr, ytr = build(tr_idx, sh)
        Hva = feats(X[va_idx] @ M.T, W1, b1)
        for lam in lams:
            W2, b2 = fit_ridge(Htr, ytr, lam)
            a, _ = acc_logits(Hva, y[va_idx], W2, b2)
            if best is None or a > best[0]:
                best = (a, lam, sh)

    _, lam_star, sh_star = best

    # ---------------- final refit on the whole permitted pool ----------------
    Hall, yall = build(np.arange(N), sh_star)
    W2, b2 = fit_ridge(Hall, yall, lam_star)

    # ---------------- write back (float32, bounded) ----------------
    W1_full = (W1 @ M.T)                     # compose whitening into fc1
    W1_full = np.clip(W1_full, -50.0, 50.0)
    b1_full = np.clip(b1, -50.0, 50.0)

    w["fc1.weight"] = W1_full.astype(np.float32)
    w["fc1.bias"] = b1_full.astype(np.float32)
    w["fc2.weight"] = np.clip(W2.T, -1000.0, 1000.0).astype(np.float32)
    w["fc2.bias"] = np.clip(b2, -1000.0, 1000.0).astype(np.float32)

    # ---------------- diagnostics ----------------
    Hf = feats(X @ M.T, W1, b1)
    Lf = Hf @ W2 + b2
    pred = Lf.argmax(axis=1)
    P = np.exp(Lf - Lf.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.clip(P[np.arange(N), y], 1e-12, None)).mean())
    per_class = {}
    for c in range(C):
        m = (y == c)
        per_class[str(c)] = float((pred[m] == c).mean()) if m.any() else 0.0
    Hv = feats(X[va_idx] @ M.T, W1, b1)
    Lv = Hv @ W2 + b2
    val_acc = float((Lv.argmax(axis=1) == y[va_idx]).mean())

    diag = {
        "hypothesis": "whitened-Sw hidden features (64 +/- PCA pairs + 64 random hinges) + closed-form ridge readout on shift-augmented pool",
        "lambda_star": float(lam_star),
        "shift_set": str(sh_star),
        "train_acc_full_pool": float((pred == y).mean()),
        "val_acc_holdout500": val_acc,
        "train_cross_entropy": ce,
        "per_class_acc": per_class,
        "pred_class_hist": [int((pred == c).sum()) for c in range(C)],
        "fc1_w_std": float(w["fc1.weight"].std()),
        "fc1_w_absmax": float(np.abs(w["fc1.weight"]).max()),
        "fc2_w_std": float(w["fc2.weight"].std()),
        "fc2_b": [round(float(v), 4) for v in w["fc2.bias"]],
        "hidden_zero_frac": float((Hf == 0).mean()),
        "hidden_mean": float(Hf.mean()),
        "logit_std": float(Lf.std()),
    }
    return {"weights": w, "diagnostics": diag}
