import numpy as np
import math

def edit(w, f):
    X = np.asarray(f["x"])          # [N,64] int64
    y = np.asarray(f["y"]).astype(np.float64)  # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # ---------- 1. build length-normalised bag-of-words features ----------
    mask = (X != 0)                 # ignore PAD
    L = mask.sum(axis=1).astype(np.float64)
    L[L == 0] = 1.0
    # sparse-ish accumulation into dense [N, V]
    F = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    sel = mask.reshape(-1)
    np.add.at(F, (rows[sel], cols[sel]), 1.0)
    F /= L[:, None]                 # mean pooling

    # ---------- 2. closed-form ridge fit of token scores ----------
    # target: +1 / -1
    t = 2.0 * y - 1.0
    # centre features (helps conditioning); keep intercept separately
    Fm = F.mean(axis=0, keepdims=True)
    Fc = F - Fm
    tm = t.mean()
    tc = t - tm
    lam = 1e-2 * N                  # mild ridge
    A = Fc.T @ Fc + lam * np.eye(V)
    b = Fc.T @ tc
    try:
        wv = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        wv = np.linalg.lstsq(A, b, rcond=None)[0]
    bias = tm - float(Fm @ wv)      # intercept so that wv·phi + bias ~ t

    # ---------- 3. realise w(t) exactly via a 2-D circle ----------
    # choose two orthonormal zero-mean directions p,q in R^D
    p = np.zeros(D, dtype=np.float64)
    p[0] = 1.0
    p[1] = -1.0
    p /= np.linalg.norm(p)
    q = np.zeros(D, dtype=np.float64)
    q[:] = 1.0
    q[0] -= D
    q[1] -= D
    q /= np.linalg.norm(q)
    # verify orthogonality / zero-mean
    # p,q are orthonormal and zero-mean by construction

    # scale so that |alpha| <= sqrt(D)
    amax = float(np.max(np.abs(wv)))
    if amax < 1e-12:
        amax = 1.0
    c = amax / math.sqrt(D) * 1.0000001   # head magnitude
    alpha = wv / c                        # alpha_t = w(t)/c
    # clip to keep beta real
    alpha = np.clip(alpha, -math.sqrt(D) * 0.999, math.sqrt(D) * 0.999)
    beta = np.sqrt(np.maximum(D - alpha ** 2, 0.0))

    # embed[t] = alpha_t * p + beta_t * q  (zero-mean, unit variance -> LN = itself)
    E = alpha[:, None] * p[None, :] + beta[:, None] * q[None, :]
    # PAD row -> zero (excluded anyway)
    E[0, :] = 0.0

    # head: logit1 - logit0 = c * (p @ pooled) = mean_t w(t)
    head = np.zeros((2, D), dtype=np.float64)
    head[0, :] = -0.5 * c * p
    head[1, :] =  0.5 * c * p
    hb = np.zeros(2, dtype=np.float64)
    hb[0] = -0.5 * bias
    hb[1] =  0.5 * bias

    # ---------- 4. write back ----------
    w["embed.weight"] = E.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    for blk in ("block0", "block1"):
        w[f"{blk}.qkv.weight"] = np.zeros_like(w[f"{blk}.qkv.weight"], dtype=np.float32)
        w[f"{blk}.qkv.bias"]   = np.zeros_like(w[f"{blk}.qkv.bias"],   dtype=np.float32)
        w[f"{blk}.proj.weight"] = np.zeros_like(w[f"{blk}.proj.weight"], dtype=np.float32)
        w[f"{blk}.proj.bias"]   = np.zeros_like(w[f"{blk}.proj.bias"],   dtype=np.float32)
        w[f"{blk}.ff1.weight"] = np.zeros_like(w[f"{blk}.ff1.weight"], dtype=np.float32)
        w[f"{blk}.ff1.bias"]   = np.zeros_like(w[f"{blk}.ff1.bias"],   dtype=np.float32)
        w[f"{blk}.ff2.weight"] = np.zeros_like(w[f"{blk}.ff2.weight"], dtype=np.float32)
        w[f"{blk}.ff2.bias"]   = np.zeros_like(w[f"{blk}.ff2.bias"],   dtype=np.float32)
        w[f"{blk}.norm1.weight"] = np.ones_like(w[f"{blk}.norm1.weight"], dtype=np.float32)
        w[f"{blk}.norm1.bias"]   = np.zeros_like(w[f"{blk}.norm1.bias"],   dtype=np.float32)
        w[f"{blk}.norm2.weight"] = np.ones_like(w[f"{blk}.norm2.weight"], dtype=np.float32)
        w[f"{blk}.norm2.bias"]   = np.zeros_like(w[f"{blk}.norm2.bias"],   dtype=np.float32)

    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"]   = np.zeros_like(w["norm.bias"],   dtype=np.float32)
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"]   = hb.astype(np.float32)

    # ---------- diagnostics ----------
    # quick check of realised scores on the feedback pool
    pooled = F @ (E @ (head[1] - head[0]))  # mean_t (head1-head0)·LN(embed[t])
    logit_diff = pooled + (hb[1] - hb[0])
    pred = (logit_diff > 0).astype(np.int64)
    acc = float((pred == y.astype(np.int64)).mean())

    diag = {
        "train_acc": round(acc, 5),
        "w_absmax": round(amax, 5),
        "w_std": round(float(np.std(wv)), 5),
        "bias": round(float(bias), 5),
        "head_norm": round(float(np.linalg.norm(head[1] - head[0])), 5),
        "n_nonzero_embed": int(np.count_nonzero(E)),
        "ridge_lam": float(lam),
    }
    return {"weights": w, "diagnostics": diag}
