import numpy as np

def edit(w, f):
    X_ids = f["x"]                      # [4000,64] int64
    y     = f["y"].astype(np.int64)     # [4000]
    N, L  = X_ids.shape
    V, D  = w["embed.weight"].shape     # 2000, 32
    eps   = 1e-5

    # ---------- helpers ----------
    def bow_norm(ids):
        # ids: [n,64] int64 -> normalized bag-of-words [n,V]
        n = ids.shape[0]
        mask = (ids != 0).astype(np.float32)          # [n,64]
        cnt  = mask.sum(axis=1, keepdims=True)        # [n,1]
        cnt  = np.maximum(cnt, 1.0)
        B = np.zeros((n, V), dtype=np.float32)
        for i in range(n):
            row = ids[i][mask[i] > 0]
            np.add.at(B[i], row, 1.0)
        B /= cnt
        return B, mask

    def pooled_from_embed(E, ids, mask):
        # E: [V,D]; ids: [n,64]; mask: [n,64] float
        h = E[ids]                                    # [n,64,D]
        mu = h.mean(axis=2, keepdims=True)
        var = h.var(axis=2, keepdims=True)
        z = (h - mu) / np.sqrt(var + eps)             # layernorm (gamma=1,beta=0)
        # masked mean over tokens
        z = z * mask[:, :, None]
        s = z.sum(axis=1)                             # [n,D]
        n_tok = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        return (s / n_tok).astype(np.float32)

    # ---------- 1. fit ridge on bag-of-words to get supervised directions ----------
    B, mask = bow_norm(X_ids)                         # [4000,V]
    # center features
    Bm = B.mean(axis=0, keepdims=True)
    Bc = B - Bm
    # one-hot targets
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # ridge: solve for W_bow [V,2] minimizing ||Bc @ W - Yc||^2 + lam ||W||^2
    lam = 1.0
    # use SVD of Bc for stability: Bc = U S Vt
    U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
    # W = Vt.T @ diag(S/(S^2+lam)) @ U.T @ Yc
    inv = S / (S**2 + lam)
    W_bow = Vt.T @ (inv[:, None] * (U.T @ Yc))        # [V,2]

    # ---------- 2. low-rank factorization W_bow ≈ P @ H^T (P:[V,D], H:[2,D]) ----------
    # SVD of W_bow (V x 2): W_bow = Uw Sw Vtw
    Uw, Sw, Vtw = np.linalg.svd(W_bow, full_matrices=False)   # Uw [V,2], Sw [2], Vtw [2,2]
    r = min(D, 2)
    # scale so that P has small norm (for constant-offset trick) and H large
    scale = 0.05
    P = (Uw[:, :r] * Sw[:r] * scale).astype(np.float32)       # [V,r]
    H = (Vtw[:r, :] / scale).astype(np.float32)               # [r,2]
    # pad to D columns
    P_full = np.zeros((V, D), dtype=np.float32)
    P_full[:, :r] = P
    H_full = np.zeros((2, D), dtype=np.float32)
    H_full[:, :r] = H.T                                       # [2,D]

    # ---------- 3. build new embed with constant offset w0 ----------
    w0 = np.zeros(D, dtype=np.float32)
    # choose w0 with large norm and zero mean over dims, orthogonal to P directions
    rng = np.random.RandomState(0)
    w0[:] = rng.randn(D).astype(np.float32)
    w0 -= w0.mean()
    w0 /= (np.linalg.norm(w0) + 1e-9)
    w0 *= 8.0                                                 # large offset

    E_new = w0[None, :] + P_full                              # [V,D]
    w["embed.weight"] = E_new.astype(np.float32)
    # pos stays zero, blocks stay zero (already zero)

    # ---------- 4. recompute pooled features with new embed ----------
    F_new = pooled_from_embed(w["embed.weight"], X_ids, mask) # [4000,D]

    # ---------- 5. fit head by ridge on F_new ----------
    Fm = F_new.mean(axis=0, keepdims=True)
    Fc = F_new - Fm
    lam2 = 1.0
    U2, S2, Vt2 = np.linalg.svd(Fc, full_matrices=False)
    inv2 = S2 / (S2**2 + lam2)
    W_head = (Vt2.T @ (inv2[:, None] * (U2.T @ Yc))).T        # [2,D]
    b_head = Y.mean(axis=0) - Fm @ W_head.T                   # [2]

    w["head.weight"] = W_head.astype(np.float32)
    w["head.bias"]   = b_head.astype(np.float32)

    # ---------- diagnostics ----------
    logits = F_new @ W_head.T + b_head
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    # CE
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    ce = float(-np.log(np.clip(p[np.arange(N), y], 1e-9, 1.0)).mean())

    diag = {
        "train_accuracy": acc,
        "train_ce": ce,
        "ridge_lambda_bow": lam,
        "ridge_lambda_head": lam2,
        "embed_offset_norm": float(np.linalg.norm(w0)),
        "P_norm_mean": float(np.linalg.norm(P_full, axis=1).mean()),
        "head_norm": float(np.linalg.norm(W_head)),
        "rank_used": int(r),
        "feature_mean_abs": float(np.abs(F_new.mean(axis=0)).mean()),
        "feature_std_mean": float(F_new.std(axis=0).mean()),
    }
    return {"weights": w, "diagnostics": diag}
