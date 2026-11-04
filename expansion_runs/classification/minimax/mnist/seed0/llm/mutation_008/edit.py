import numpy as np
import math

def edit(w, f):
    # ------------------------------------------------------------------ data
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"], dtype=np.int64)
    N, D = X.shape
    K = 10
    Y1 = np.zeros((N, K)); Y1[np.arange(N), y] = 1.0

    W1 = np.asarray(w["fc1.weight"], dtype=np.float64)
    b1 = np.asarray(w["fc1.bias"], dtype=np.float64)
    W2 = np.asarray(w["fc2.weight"], dtype=np.float64)
    b2 = np.asarray(w["fc2.bias"], dtype=np.float64)

    # ------------------------------------------- verify current forward pass
    def fwd(Xa, W1, b1, W2, b2):
        h = np.maximum(0.0, Xa @ W1.T + b1)
        return h, h @ W2.T + b2
    h0, l0 = fwd(X, W1, b1, W2, b2)
    feat_err = float(np.abs(h0 - np.asarray(f["features"], dtype=np.float64)).max())
    logit_err = float(np.abs(l0 - np.asarray(f["logits"], dtype=np.float64)).max())
    cur_acc = float((l0.argmax(1) == y).mean())

    # ------------------------------------------------------------- utilities
    def ridge_fit(A, T, lam):
        """closed-form ridge normal equations (centered) with intercept."""
        n = A.shape[0]
        Am = A.mean(0); Tm = T.mean(0)
        Ac = A - Am; Tc = T - Tm
        G = Ac.T @ Ac
        G.flat[::G.shape[0] + 1] += lam * n
        B = np.linalg.solve(G, Ac.T @ Tc)
        return B, Tm - Am @ B

    def ridge_loo(A, T, lam):
        """closed-form LOO MSE via hat matrix (no training loop)."""
        n = A.shape[0]
        Am = A.mean(0); Tm = T.mean(0)
        Ac = A - Am; Tc = T - Tm
        G = Ac.T @ Ac
        G.flat[::G.shape[0] + 1] += lam * n
        B = np.linalg.solve(G, Ac.T @ Tc)
        R = Tc - Ac @ B
        H = Ac @ np.linalg.solve(G, Ac.T)
        d = 1.0 - np.clip(np.diag(H), -0.999, 0.999)
        loo = float(np.mean((R / d[:, None]) ** 2))
        return loo, B, Tm - Am @ B

    def pick_lam(A, T, lams):
        best = (None, None, None, 1e18)
        for lam in lams:
            loo, B, c = ridge_loo(A, T, lam)
            if loo < best[3]:
                best = (lam, B, c, loo)
        return best

    # ------------------------------------------------- augmentation (shifts)
    def shift_batch(Xa, dx, dy):
        P = np.zeros((Xa.shape[0], 28, 28))
        for r in range(28):
            for c in range(28):
                rr = r - dx; cc = c - dy
                if 0 <= rr < 28 and 0 <= cc < 28:
                    P[:, r, c] = Xa[:, rr, cc]
        return P.reshape(-1, 784)

    Xim = X.reshape(-1, 28, 28)
    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    Xa_list = [X] + [shift_batch(Xim, dx, dy) for dx, dy in shifts[1:]]
    Xa = np.concatenate(Xa_list, 0)
    ya = np.concatenate([y] * len(Xa_list), 0)
    Ya = np.zeros((Xa.shape[0], K)); Ya[np.arange(Xa.shape[0]), ya] = 1.0

    # ------------------------------------------- PCA / whitening of pool
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / N
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    evals = np.maximum(evals, 1e-8)
    # whitened basis (top 200 directions), scaled so that var(w.x)=1
    nW = 200
    Vw = evecs[:, :nW] / np.sqrt(evals[:nW])[None, :]
    Z = Xc @ Vw                      # whitened coordinates (pool)
    Za = (Xa - mu) @ Vw             # whitened coordinates (augmented)

    # class means in whitened space
    cm = np.zeros((K, nW))
    for c in range(K):
        m = (y == c)
        cm[c] = Z[m].mean(0)
    gm = Z.mean(0)
    Sw = np.zeros((nW, nW))
    for c in range(K):
        m = (y == c)
        Zc = Z[m] - cm[c]
        Sw += Zc.T @ Zc
    Sw /= N
    Sw.flat[::nW + 1] += 1e-3 * np.trace(Sw) / nW
    ev2, U2 = np.linalg.eigh(Sw)
    ev2 = np.maximum(ev2, 1e-8)
    Wh = U2 / np.sqrt(ev2)[None, :]        # within-class whitening (nW x nW)
    Ld = (cm - gm) @ Wh                    # LDA directions in whitened space

    # --------------------------------------------------- hidden unit design
    # (a) always-on linear features: whitened PCA directions (bias >> range)
    n_lin = 96
    dirs_lin = Vw[:, :n_lin].T                       # (n_lin, 784)
    # (b) rectified LDA "hinge pairs" (16 directions x 2 sides)
    n_lda = 16
    lda_dirs = Ld[:n_lda]                            # (n_lda, nW)
    # (c) rectified class-template projections (8 classes x 2 templates)
    n_tpl_cls = 8
    tpl_dirs = []
    for c in range(n_tpl_cls):
        m = (y == c)
        Zc = Z[m] - cm[c]
        Uc, sv, Vtc = np.linalg.svd(Zc, full_matrices=False)
        for j in range(2):
            v = Vtc[j]
            tpl_dirs.append(cm[c] + 0.8 * v)
    tpl_dirs = np.asarray(tpl_dirs)                  # (16, nW)

    # assemble directions in whitened space, then map back to pixel space
    dirW = np.concatenate([np.eye(nW)[:, :n_lin], lda_dirs, tpl_dirs], 0)   # (nW, m)
    m_units = dirW.shape[0]
    Wpix = (mu @ dirW.T)                              # bias part: dirW @ mu
    # pixel-space weight rows: dirW[k] in whitened coords -> x-space
    # z = (x-mu) @ Vw  =>  dirW[k].z = (x-mu) @ (Vw @ dirW[k])
    Wpix = (Vw @ dirW).T                              # (m_units, 784)
    b_pix = -(Wpix @ mu)                              # so that Wpix x + b_pix = dirW.z

    # pre-activation ranges (pool + augmented) to set biases
    pre = Xa @ Wpix.T + b_pix[None, :]
    lo = pre.min(0); hi = pre.max(0)
    is_lin = np.zeros(m_units, dtype=bool); is_lin[:n_lin] = True
    is_lda = np.zeros(m_units, dtype=bool); is_lda[n_lin:n_lin + 2 * n_lda] = True
    is_tpl = ~(is_lin | is_lda)

    bnew = np.zeros(m_units)
    # linear units: always on (bias above max pre-activation)
    bnew[is_lin] = hi[is_lin] + 1.0
    # LDA units: hinge pair around class-score threshold (median of pool pre)
    med = np.median(X @ Wpix.T + b_pix[None, :], axis=0)
    bnew[is_lda] = -med[is_lda]
    # template units: rectified at a fraction of their pool max
    pm = (X @ Wpix.T + b_pix[None, :])
    for i in np.where(is_tpl)[0]:
        bnew[i] = -0.5 * pm[:, i].max()

    # ------------------------------------------------- build feature matrices
    def feats(Xin):
        return np.maximum(0.0, Xin @ Wpix.T + bnew[None, :])
    A0 = feats(X)                    # pool features (true, post-ReLU)
    Aa = feats(Xa)                   # augmented features

    # ------------------------------------------------- readout: ridge + LOO
    lams = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    lam_a, B_a, c_a, loo_a = pick_lam(Aa, Ya, lams)
    lam_p, B_p, c_p, loo_p = pick_lam(A0, Y1, lams)

    # blend: augmented fit is the robust choice; pool fit as fallback
    use_aug = (loo_a <= 1.25 * loo_p)
    if use_aug:
        B = B_a; c = c_a; lam = lam_a; loo = loo_a
    else:
        B = B_p; c = c_p; lam = lam_p; loo = loo_p

    # ------------------------------------------------- assemble final weights
    # hidden: h = relu(x @ W1.T + b1); our design: W1 = Wpix, b1 = bnew
    # readout: logits = h @ W2.T + b2  =>  W2 = B.T, b2 = c
    W1_new = Wpix.astype(np.float32)
    b1_new = bnew.astype(np.float32)
    W2_new = B.T.astype(np.float32)
    b2_new = c.astype(np.float32)

    # numerical safety: finite, bounded magnitude
    W1_new = np.clip(np.nan_to_num(W1_new, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
    b1_new = np.clip(np.nan_to_num(b1_new, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)
    W2_new = np.clip(np.nan_to_num(W2_new, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
    b2_new = np.clip(np.nan_to_num(b2_new, nan=0.0, posinf=0.0, neginf=0.0), -100.0, 100.0)

    # ------------------------------------------------------------ diagnostics
    def acc_of(W1_, b1_, W2_, b2_, Xa_, ya_):
        hh = np.maximum(0.0, Xa_ @ W1_.T + b1_)
        ll = hh @ W2_.T + b2_
        return float((ll.argmax(1) == ya_).mean()), ll
    acc_pool, lpool = acc_of(W1_new, b1_new, W2_new, b2_new, X, y)
    acc_aug, _ = acc_of(W1_new, b1_new, W2_new, b2_new, Xa, ya)
    # cross-entropy on pool
    lp = lpool - lpool.max(1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    ce_pool = float(-lp[np.arange(N), y].mean())
    # batch check
    bidx = np.asarray(f["batch.ids"], dtype=np.int64)
    bacc, lb = acc_of(W1_new, b1_new, W2_new, b2_new, X[bidx], np.asarray(f["batch.y"])[np.argsort(bidx)] if False else np.asarray(f["batch.y"], dtype=np.int64))
    # per-class accuracy on pool
    pc = [float((lpool[y == c].argmax(1) == c).mean()) for c in range(K)]
    # active fraction of rectified units
    act = (A0 > 0).mean(0)

    diag = {
        "hypothesis": "shift-augmented ridge readout on 96 always-on whitened-PCA units + 32 rectified LDA/template units",
        "current_forward_feature_maxerr": round(feat_err, 6),
        "current_forward_logit_maxerr": round(logit_err, 6),
        "current_pool_acc": round(cur_acc, 4),
        "used_augmented_fit": bool(use_aug),
        "ridge_lambda": float(lam),
        "loo_mse": round(float(loo), 5),
        "loo_mse_aug": round(float(loo_a), 5),
        "loo_mse_pool": round(float(loo_p), 5),
        "new_pool_acc": round(acc_pool, 4),
        "new_pool_acc_augmented": round(acc_aug, 4),
        "new_pool_ce": round(ce_pool, 4),
        "new_batch_acc": round(bacc, 4),
        "per_class_acc": [round(v, 3) for v in pc],
        "active_frac_linear": round(float(act[:n_lin].mean()), 4),
        "active_frac_lda": round(float(act[n_lin:n_lin + 2 * n_lda].mean()), 4),
        "active_frac_tpl": round(float(act[n_lin + 2 * n_lda:].mean()), 4),
        "fc1_w_absmax": round(float(np.abs(W1_new).max()), 4),
        "fc2_w_absmax": round(float(np.abs(W2_new).max()), 4),
        "fc1_b_min_max": [round(float(b1_new.min()), 4), round(float(b1_new.max()), 4)],
        "n_train": int(N),
        "n_aug": int(Xa.shape[0]),
        "n_units": int(m_units),
    }

    w["fc1.weight"] = W1_new
    w["fc1.bias"] = b1_new
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new
    return {"weights": w, "diagnostics": diag}
