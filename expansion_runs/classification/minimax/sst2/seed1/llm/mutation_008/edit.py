import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------ data
    X_ids = np.asarray(f["x"], dtype=np.int64)          # [N,64] token ids
    y = np.asarray(f["y"], dtype=np.int64)              # [N] labels 0/1
    N, L = X_ids.shape

    mask = (X_ids != 0)                                  # non-pad
    ntok = mask.sum(1).astype(np.float64)
    ntok[ntok == 0] = 1.0

    # ------------------------------------------------- length-normalised BoW
    V = int(w["embed.weight"].shape[0])
    rows = np.repeat(np.arange(N), L)
    cols = X_ids.reshape(-1)
    vals = mask.reshape(-1).astype(np.float64)
    keep = vals > 0
    rows, cols = rows[keep], cols[keep]
    C = np.zeros((N, V), dtype=np.float64)
    np.add.at(C, (rows, cols), 1.0)
    Xn = C / ntok[:, None]                              # [N,V] mean-normalised counts

    t = (y.astype(np.float64) * 2.0 - 1.0)               # +-1 targets
    # centre features (intercept handled separately by head.bias)
    Xc = Xn - Xn.mean(0, keepdims=True)
    tc = t - t.mean()

    # ------------------------------------------- closed-form ridge (normal eq.)
    lam = 1e-2
    A = Xc.T @ Xc
    A[np.diag_indices_from(A)] += lam
    s = np.linalg.solve(A, Xc.T @ tc)
    b = float(t.mean() - Xn.mean(0) @ s)                # intercept for head.bias

    # ------------------------------------------------- deployable score check
    score = Xn @ s + b
    pred = (score > 0).astype(np.int64)
    acc = float((pred == y).mean())
    p = np.clip(0.5 * (1.0 + np.tanh(score)), 1e-6, 1 - 1e-6)
    ce = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # --------------------------------------------- construct embed rows (exact)
    # e_t = (kappa*s_t, -kappa*s_t/31 repeated 31x):  zero mean, unit variance
    # => final LayerNorm(e_t) == e_t, so pooled[0] = kappa * mean_t s_t
    kappa = np.sqrt(31.0 / 32.0)
    e = np.zeros((V, 32), dtype=np.float64)
    e[:, 0] = kappa * s
    e[:, 1:] = (-kappa * s / 31.0)[:, None]
    # re-normalise exactly (numerical hygiene) -> zero mean, unit var
    e -= e.mean(1, keepdims=True)
    e /= np.maximum(np.sqrt((e * e).mean(1, keepdims=True)), 1e-12)

    E = w["embed.weight"]
    newE = np.array(E, dtype=np.float64)
    newE[:] = e
    # keep pad row (id 0) exactly zero: it is masked out of pooling anyway
    newE[0, :] = 0.0
    w["embed.weight"] = newE.astype(np.float32)

    # ------------------------------------------------------- pos / blocks / head
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    for blk in ("block0", "block1"):
        for suf in ("norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                    "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                    "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"):
            key = blk + "." + suf
            if key in w:
                w[key] = np.zeros_like(w[key])
    w["norm.weight"] = np.ones_like(w["norm.weight"])
    w["norm.bias"] = np.zeros_like(w["norm.bias"])

    hw = np.zeros((2, 32), dtype=np.float64)
    hw[0, 0] = -1.0
    hw[1, 0] = +1.0
    w["head.weight"] = hw.astype(np.float32)
    hb = np.array([-b, +b], dtype=np.float64)
    w["head.bias"] = hb.astype(np.float32)

    # ------------------------------------------------- exact forward verification
    def forward(ids):
        m = (ids != 0)
        h = w["embed.weight"][ids].astype(np.float64) + w["pos.weight"][None, :, :]
        # blocks are all-zero -> identity
        g = w["norm.weight"].astype(np.float64)
        hb2 = w["norm.bias"].astype(np.float64)
        mu = h.mean(-1, keepdims=True)
        var = h.var(-1, keepdims=True)
        z = (h - mu) / np.sqrt(var + 1e-5) * g + hb2
        cnt = m.sum(1)[:, None]
        pooled = (z * m[:, :, None]).sum(1) / np.maximum(cnt, 1)
        lg = pooled @ w["head.weight"].astype(np.float64).T + w["head.bias"].astype(np.float64)
        return lg, pooled

    lg, pooled = forward(X_ids)
    new_pred = lg.argmax(1)
    new_acc = float((new_pred == y).mean())
    ex = np.exp(lg - lg.max(1, keepdims=True))
    pr = ex / ex.sum(1, keepdims=True)
    new_ce = float(-np.mean(np.log(np.clip(pr[np.arange(N), y], 1e-12, None))))

    # --------------------------------------------------------------- diagnostics
    old_logits = np.asarray(f["logits"], dtype=np.float64)
    old_pred = old_logits.argmax(1)
    old_acc = float((old_pred == y).mean())
    exo = np.exp(old_logits - old_logits.max(1, keepdims=True))
    pro = exo / exo.sum(1, keepdims=True)
    old_ce = float(-np.mean(np.log(np.clip(pro[np.arange(N), y], 1e-12, None))))

    # token-level diagnostic: how many tokens carry non-trivial sentiment weight
    nz = int(np.sum(np.abs(s) > 1e-6))
    top = np.argsort(-np.abs(s))[:15]
    diag = {
        "pool_acc_before": round(old_acc, 4),
        "pool_acc_after": round(new_acc, 4),
        "pool_ce_before": round(old_ce, 4),
        "pool_ce_after": round(new_ce, 4),
        "ridge_fit_acc": round(acc, 4),
        "ridge_fit_ce": round(ce, 4),
        "acc_delta": round(new_acc - old_acc, 4),
        "class_acc_after": [round(float((new_pred[y == c] == c).mean()), 4) for c in (0, 1)],
        "class_acc_before": [round(float((old_pred[y == c] == c).mean()), 4) for c in (0, 1)],
        "n_tokens_nonzero_weight": nz,
        "lambda": lam,
        "intercept_b": round(b, 4),
        "top_tokens_by_abs_weight": [[int(t_), round(float(s[t_]), 4)] for t_ in top],
        "embed_row0_absmax": float(np.abs(w["embed.weight"][0]).max()),
        "embed_absmax": float(np.abs(w["embed.weight"]).max()),
        "pooled0_std_after": round(float(pooled[:, 0].std()), 4),
        "pooled_other_absmax": round(float(np.abs(pooled[:, 1:]).max()), 6),
    }
    return {"weights": w, "diagnostics": diag}
