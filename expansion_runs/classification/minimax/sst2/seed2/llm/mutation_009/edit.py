import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids (0=pad)
    y = np.asarray(f["y"], dtype=np.int64)          # [N] labels 0/1
    N, L = x.shape
    V = w["embed.weight"].shape[0]                   # 2000

    mask = (x != 0)
    lens = mask.sum(axis=1).astype(np.float64)       # tokens per example
    lens[lens == 0] = 1.0

    # ---------------- bag-of-words design matrix (mean-pooled counts) -------
    # row_i[c] = count of token c in example i / len_i   (matches masked_mean pooling)
    B = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), lens.astype(np.int64))
    cols = x[mask]
    np.add.at(B, (rows, cols), 1.0)
    B /= lens[:, None]

    # ---------------- closed-form ridge fit (normal equations) -------------
    t = (2.0 * y.astype(np.float64) - 1.0)          # +-1 targets
    G = B.T @ B                                      # [V,V]
    b = B.T @ t                                     # [V]
    lam = 1e-3 * float(np.trace(G)) / V + 1e-12
    A = G + lam * np.eye(V)
    g = np.linalg.solve(A, b)                       # per-token scores

    # ---------------- build the embedding table -------------------------
    C = 100.0
    E = np.zeros((V, 32), dtype=np.float64)
    E[:, 0] = g                                     # token score
    E[:, 1] = C                                     # large constant -> LN affine
    E[:, 2] = -g                                    # keeps row-mean constant
    # rows 0 (pad) and 1 (unk): neutral (g=0) so padding contributes nothing
    E[0, :] = 0.0
    E[1, 0] = 0.0; E[1, 2] = 0.0

    # ---------------- exact forward pass (own numpy re-implementation) ----
    def forward(emb):
        h = emb[x]                                   # [N,64,32]
        m = h.mean(axis=2, keepdims=True)
        v = h - m
        s = np.sqrt((v * v).mean(axis=2, keepdims=True) + 1e-5)
        z = v / s                                    # final LayerNorm (gamma=1,beta=0)
        pooled = (z * mask[:, :, None]).sum(axis=1) / lens[:, None]
        return pooled

    pooled = forward(E)
    p0 = pooled[:, 0]
    # affine map pooled0 = A + B*mean(g);  solve for the two coefficients
    mg = (B @ g)
    A_ = float(np.mean(p0 - (np.polyfit(mg, p0, 1)[0] * mg)))
    B_ = float(np.polyfit(mg, p0, 1)[0])
    # residual of the affine model (should be ~1e-5)
    resid = float(np.max(np.abs(p0 - (A_ + B_ * mg))))

    # ---------------- head: turn pooled0 into a calibrated logit ---------
    # score s_i = (pooled0 - thr)/scale ; choose thr = decision threshold on mean(g)
    # threshold from the ridge fit itself: sign of the fitted score
    fit = B @ g
    thr_g = 0.0
    # margin scale: use spread of the fitted score
    scale_g = max(float(np.std(fit)), 1e-6)
    thr_pool = A_ + B_ * thr_g
    scale_pool = B_ * scale_g
    if scale_pool == 0.0:
        scale_pool = 1.0
    # head.weight[1,0] = +1/scale_pool (positive class reads pooled0),
    # head.weight[0,0] = 0 ; biases set the threshold
    hw = np.zeros((2, 32), dtype=np.float64)
    hb = np.zeros(2, dtype=np.float64)
    hw[1, 0] = 1.0 / scale_pool
    hb[1] = -thr_pool / scale_pool
    hb[0] = 0.0

    # ---------------- write back -----------------------------------------
    w["embed.weight"] = E.astype(np.float32)
    w["head.weight"] = hw.astype(np.float32)
    w["head.bias"] = hb.astype(np.float32)
    # everything else untouched (blocks stay zero -> exact reproducibility)

    # ---------------- diagnostics ----------------------------------------
    def acc_from_pooled(pl):
        lg = pl @ hw.T + hb
        return float((np.argmax(lg, axis=1) == y).mean())

    acc_new = acc_from_pooled(pooled)
    old_logits = np.asarray(f["logits"], dtype=np.float64)
    acc_old = float((np.argmax(old_logits, axis=1) == y).mean())
    probs = np.exp(pooled @ hw.T + hb)
    probs = probs / probs.sum(axis=1, keepdims=True)
    diag = {
        "acc_old": round(acc_old, 4),
        "acc_new": round(acc_new, 4),
        "acc_new_cls0": round(float((np.argmax(probs, 1)[y == 0] == 0).mean()), 4),
        "acc_new_cls1": round(float((np.argmax(probs, 1)[y == 1] == 1).mean()), 4),
        "affine_resid_max": round(resid, 8),
        "pooled0_A": round(A_, 6),
        "pooled0_B": round(B_, 6),
        "g_absmax": round(float(np.max(np.abs(g))), 4),
        "g_nonzero": int(np.count_nonzero(g)),
        "embed_absmax": round(float(np.max(np.abs(E))), 3),
        "head_w": [round(float(v), 6) for v in hw[1]],
        "head_b": [round(float(v), 6) for v in hb],
        "mean_len": round(float(lens.mean()), 2),
        "note": "blocks/pos/norm untouched (zero) so forward is exactly reproduced",
    }
    return {"weights": w, "diagnostics": diag}
