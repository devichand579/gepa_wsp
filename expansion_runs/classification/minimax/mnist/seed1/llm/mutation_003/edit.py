import numpy as np


def edit(w, f):
    rng = np.random.default_rng(0)

    # ---------------- data ----------------
    X = f["x"].reshape(f["x"].shape[0], -1).astype(np.float64)   # (N,784)
    y = f["y"].astype(np.int64)
    N, D = X.shape
    C = 10
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    mu = X.mean(axis=0)
    Xc = X - mu

    # ---------------- PCA basis ----------------
    # economical SVD of centred data
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k_pca = 64
    V = Vt[:k_pca].T                      # (784,64)
    Z = Xc @ V                            # (N,64) PCA scores

    # ---------------- class statistics ----------------
    mu_c = np.zeros((C, D))
    Sw = np.zeros((D, D))
    for c in range(C):
        Xc_cls = X[y == c]
        mu_c[c] = Xc_cls.mean(axis=0)
        Zc = Xc_cls - mu_c[c]
        Sw += Zc.T @ Zc
    Sw /= max(N - C, 1)
    St = Xc.T @ Xc / max(N - 1, 1)

    def _inv_sqrt(A, eps):
        w_, V_ = np.linalg.eigh(A)
        w_ = np.maximum(w_, eps)
        return (V_ * (w_ ** -0.5)) @ V_.T

    Sw_inv_sqrt = _inv_sqrt(Sw, 1e-3)
    St_inv_sqrt = _inv_sqrt(St, 1e-3)

    # ---------------- hidden unit design ----------------
    W_list = []      # (784,) each
    B_list = []      # scalar thresholds (pre-activation offset)
    kinds = []

    # (A) folding PCA units: relu(z - t) and relu(t - z)
    for j in range(k_pca):
        t = float(np.median(Z[:, j]))
        s = float(Z[:, j].std()) + 1e-9
        W_list.append(V[:, j]);  B_list.append(-t);      kinds.append("pca+")
        W_list.append(-V[:, j]); B_list.append(t);       kinds.append("pca-")

    # (B) class-conditional prototype units
    for c in range(C):
        Xc_cls = X[y == c]
        m = mu_c[c]
        # within-class principal directions (deformation templates)
        Zcls = Xc_cls - m
        swc = Zcls.T @ Zcls / max(len(Xc_cls) - 1, 1)
        ew, ev = np.linalg.eigh(swc)
        order = np.argsort(ew)[::-1]
        vdirs = ev[:, order[:3]]
        proj_std = np.array([float((Zcls @ vdirs[:, j]).std()) for j in range(3)])
        # whitened class template
        w_white = Sw_inv_sqrt @ m
        nrm = np.linalg.norm(w_white) + 1e-12
        w_white = w_white / nrm
        # LDA direction
        w_lda = St_inv_sqrt @ (m - mu)
        nrm2 = np.linalg.norm(w_lda) + 1e-12
        w_lda = w_lda / nrm2
        # raw class mean template
        w_raw = m / (np.linalg.norm(m) + 1e-12)

        cands = [w_raw, w_white, w_lda]
        for j in range(3):
            g = 1.0 * proj_std[j] / (np.linalg.norm(vdirs[:, j]) + 1e-12)
            cands.append(m + g * vdirs[:, j])
        for wv in cands:
            wv = wv / (np.linalg.norm(wv) + 1e-12)
            W_list.append(wv); B_list.append(0.0); kinds.append("proto")

    W1 = np.stack(W_list, axis=0)          # (128,784)
    B1 = np.asarray(B_list, dtype=np.float64)

    # per-unit thresholds for prototype units (quantile of projection)
    proj = X @ W1.T                        # (N,128)
    for j in range(W1.shape[0]):
        if kinds[j] == "proto":
            q = float(np.quantile(proj[:, j], 0.55))
            B1[j] = -q
        else:
            B1[j] = B1[j] - 0.0

    # ---------------- forward pass with new fc1 ----------------
    Hpre = X @ W1.T + B1                   # (N,128)
    H = np.maximum(Hpre, 0.0)
    act_frac = (H > 0).mean(axis=0)

    # ---------------- closed-form ridge readout ----------------
    Hc = H - H.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    G = Hc.T @ Hc                          # (128,128)
    B = Hc.T @ Yc                          # (128,10)
    tr = np.trace(G) / max(G.shape[0], 1)

    best = None
    for lam in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]:
        A = G + lam * tr * np.eye(G.shape[0])
        try:
            W2 = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            continue
        # GCV-style score (closed form)
        dof = float(np.trace(np.linalg.solve(A, G)))
        resid = float(np.sum((Yc - Hc @ W2) ** 2))
        denom = max(1.0 - dof / N, 1e-6)
        gcv = resid / (N * denom * denom)
        if best is None or gcv < best[0]:
            best = (gcv, lam, W2)

    gcv, lam, W2 = best
    b2 = Y.mean(axis=0) - H.mean(axis=0) @ W2

    # ---------------- diagnostics ----------------
    pred = (H @ W2 + b2).argmax(axis=1)
    acc = float((pred == y).mean())
    logits = H @ W2 + b2
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.maximum(p[np.arange(N), y], 1e-12)).mean())

    # ---------------- write back ----------------
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = B1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)

    # sanity: finite / bounded
    for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
        arr = w[k]
        assert np.all(np.isfinite(arr)), k
        assert float(np.abs(arr).max()) <= 1000.0, k

    diagnostics = {
        "design": "64 folding-PCA + 64 class-prototype hidden units, ridge readout",
        "n_train": int(N),
        "lambda": float(lam),
        "gcv": float(gcv),
        "train_acc": acc,
        "train_cross_entropy": ce,
        "prev_acc": 0.1295,
        "prev_ce": 2.2962,
        "act_frac_mean": float(act_frac.mean()),
        "act_frac_min": float(act_frac.min()),
        "act_frac_max": float(act_frac.max()),
        "fc1_w_std": float(W1.std()),
        "fc1_w_absmax": float(np.abs(W1).max()),
        "fc2_w_std": float(W2.std()),
        "fc2_w_absmax": float(np.abs(W2).max()),
        "fc2_b": [float(v) for v in b2],
        "per_class_acc": [float((pred[y == c] == c).mean()) for c in range(C)],
        "top1_pca_evr": float((S[:k_pca] ** 2).sum() / (S ** 2).sum()),
    }
    return {"weights": w, "diagnostics": diagnostics}
