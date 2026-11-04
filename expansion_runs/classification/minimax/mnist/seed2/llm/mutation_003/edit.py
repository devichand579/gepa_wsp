import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    Y = np.asarray(f["y"], dtype=np.int64)
    N = X.shape[0]

    # ---------- class statistics ----------
    Yh = np.zeros((N, 10))
    Yh[np.arange(N), Y] = 1.0
    mu = X.mean(axis=0)                      # global mean image
    Xc = X - mu
    cls_mean = np.zeros((10, 784))
    for c in range(10):
        m = Y == c
        cls_mean[c] = X[m].mean(axis=0) if m.any() else mu
    D = cls_mean - mu                       # class-mean directions (10 x 784)

    # ---------- regularized within-class covariance ----------
    Sw = np.zeros((784, 784))
    for c in range(10):
        m = Y == c
        if m.any():
            Z = X[m] - cls_mean[c]
            Sw += Z.T @ Z
    Sw /= max(N - 10, 1)
    tr = np.trace(Sw) / 784.0
    Sw += (0.05 * tr + 1e-6) * np.eye(784)   # ridge on covariance

    # ---------- whitened between-class scatter -> LDA subspace ----------
    Sb = (D.T @ D) / 10.0
    evals, evecs = np.linalg.eigh(Sw)
    evals = np.clip(evals, 1e-8, None)
    Wh = evecs / np.sqrt(evals)              # whitening (784 x 784)
    M = Wh @ D.T                            # whitened class means (784 x 10)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    K = 64
    V = Vt[:K].T                            # K x 10  (whitened LDA directions)
    Wdir = (Wh.T @ V).T                      # K x 784 (pixel-space directions)
    # orthonormalize in pixel space (numerical hygiene)
    for _ in range(2):
        Wdir -= (Wdir @ Wdir.T) @ Wdir
    Wdir /= np.maximum(np.linalg.norm(Wdir, axis=1, keepdims=True), 1e-12)

    # ---------- hidden layer: 64 directions x {+,-} ReLU pairs ----------
    # h[:, 2j]   = relu( Wdir[j] . x )
    # h[:, 2j+1] = relu(-Wdir[j] . x )
    # -> the pair spans {linear score, rectification} for each direction
    Z = X @ Wdir.T                          # N x K
    H = np.empty((N, 2 * K))
    H[:, 0::2] = np.maximum(Z, 0.0)
    H[:, 1::2] = np.maximum(-Z, 0.0)
    H = np.hstack([H, np.ones((N, 1))])      # bias column

    # ---------- closed-form ridge readout (GCV over a small lambda path) ----------
    Yt = Yh - 1.0 / 10.0
    G = H.T @ H
    B = H.T @ Yt
    n, p = H.shape
    tr_h = np.trace(G)
    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        R = np.linalg.solve(G + lam * n * np.eye(p), B)
        res = Yt - H @ R
        rss = float(np.sum(res * res))
        dof = float(np.trace(np.linalg.solve(G + lam * n * np.eye(p), G)))
        gcv = rss / max((1.0 - dof / n) ** 2, 1e-6)
        if best is None or gcv < best[0]:
            best = (gcv, lam, R)
    gcv, lam, R = best

    # ---------- write back ----------
    W1 = np.zeros((128, 784))
    B1 = np.zeros(128)
    W1[0::2] = Wdir
    W1[1::2] = -Wdir
    W2 = np.zeros((10, 128))
    W2[:] = R[:10, :128].T
    B2 = R[:10, 128].copy()

    # ---------- diagnostics ----------
    Hraw = np.empty((N, 128))
    Hraw[:, 0::2] = np.maximum(Z, 0.0)
    Hraw[:, 1::2] = np.maximum(-Z, 0.0)
    L = Hraw @ W2.T + B2
    pred = np.argmax(L, axis=1)
    acc = float(np.mean(pred == Y))
    # linear-only reference (ridge on raw pixels)
    Gx = X.T @ X
    Bx = X.T @ Yt
    Rx = np.linalg.solve(Gx + 1e-2 * N * np.eye(784), Bx)
    lx = X @ Rx + (Yt.mean(axis=0) - X.mean(axis=0) @ Rx)
    acc_lin = float(np.mean(np.argmax(lx, axis=1) == Y))

    def _stats(a):
        a = np.asarray(a, dtype=np.float64)
        return {
            "shape": list(a.shape),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
            "finite": bool(np.all(np.isfinite(a))),
        }

    diagnostics = {
        "train_acc_new": acc,
        "train_acc_linear_ref": acc_lin,
        "ridge_lambda": float(lam),
        "gcv": float(gcv),
        "lda_rank_used": int(K),
        "sw_trace_over_d": float(tr),
        "top_lda_eigvals": [float(v) for v in s[:12]],
        "mean_pairwise_lda_corr": float(np.mean([
            np.abs(np.dot(Wdir[i], Wdir[j]))
            for i in range(K) for j in range(i + 1, K)])),
        "fc1_weight": _stats(W1),
        "fc1_bias": _stats(B1),
        "fc2_weight": _stats(W2),
        "fc2_bias": _stats(B2),
        "hidden_sparsity": float(np.mean(Hraw == 0.0)),
        "per_class_acc": [float(np.mean(pred[Y == c] == c)) for c in range(10)],
        "logit_std": float(np.std(L)),
        "note": "fc1 = 64 regularized-LDA directions with +/- ReLU pairs (bias 0); fc2 = closed-form ridge readout on those features with bias; lambda chosen by closed-form GCV.",
    }

    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = B1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = B2.astype(np.float32)
    return {"weights": w, "diagnostics": diagnostics}
