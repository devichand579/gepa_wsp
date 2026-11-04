import numpy as np
import math

def edit(w, f):
    x = np.asarray(f["x"]); y = np.asarray(f["y"]).astype(np.int64)
    N, T = x.shape
    V = w["embed.weight"].shape[0]          # 2000
    D = w["embed.weight"].shape[1]          # 32
    eps = 1e-5

    # ---------- 1. length-normalized BoW counts ----------
    mask = (x != 0)
    L = mask.sum(axis=1).astype(np.float64)
    L[L == 0] = 1.0
    P = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = x.reshape(-1)
    keep = cols != 0
    np.add.at(P, (rows[keep], cols[keep]), 1.0)
    P /= L[:, None]                          # P[i,v] = count/L

    # ---------- 2. closed-form ridge BoW (target +1 / -1) ----------
    t = np.where(y == 1, 1.0, -1.0)
    # center targets, add explicit bias
    t = t - t.mean()
    G = P.T @ P                              # [V,V]
    # GCV over a small closed-form grid of lambdas (no iterative training)
    best = None
    PtP_diag = np.diag(G).copy()
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        A = G + lam * np.eye(V)
        try:
            s = np.linalg.solve(A, P.T @ t)
        except np.linalg.LinAlgError:
            continue
        # GCV: ||P s - t||^2 / (N - tr(H))^2, H = P (G+lam I)^-1 P^T
        resid = P @ s - t
        # tr(H) = sum_v (P^T P)_vv * A^-1_vv  -> use diag of A^-1
        Ainv_diag = np.diag(np.linalg.inv(A))
        trH = float(np.sum(PtP_diag * Ainv_diag))
        denom = max(N - trH, 1e-6)
        gcv = float(resid @ resid) / (denom * denom)
        if best is None or gcv < best[0]:
            best = (gcv, lam, s)
    lam, s = best[1], best[2]
    b_prime = float(t.mean())                # bias in centered target space

    # ---------- 3. realize exactly ----------
    c = 0.5 * (np.abs(s).max() / math.sqrt(31.0)) + 1e-3
    c = max(c, 1e-3)
    a = s / (2.0 * c)                        # desired u_v[0], |a| <= sqrt(31)

    U = np.zeros((V, D), dtype=np.float64)
    U[:, 0] = a
    rem = D - 1                              # 31
    m = -a / rem                             # mean of remaining coords
    # remaining coords: m + r, with sum(r)=0 and sum(r^2)=32 - a^2 - 31 m^2
    target_sq = D - a * a - rem * m * m
    target_sq = np.maximum(target_sq, 0.0)
    # distribute r across the 31 coords: r_k = alpha * cos(theta_k) pattern
    # simple feasible construction: r = sqrt(target_sq/rem) * z, z zero-mean unit-var
    z = np.zeros((V, rem), dtype=np.float64)
    # deterministic zero-mean, unit-variance pattern over 31 coords
    idx = np.arange(rem)
    pat = np.cos(2.0 * np.pi * idx / rem)
    pat = pat - pat.mean()
    pat = pat / np.sqrt((pat * pat).mean())
    z = np.sqrt(target_sq / rem)[:, None] * pat[None, :]
    U[:, 1:] = m[:, None] + z
    # exact renormalization to zero-mean / unit-variance
    U = U - U.mean(axis=1, keepdims=True)
    U = U / np.sqrt((U * U).mean(axis=1, keepdims=True) + 1e-12)
    # re-inject the desired first coordinate (small correction)
    U[:, 0] = a
    U = U - U.mean(axis=1, keepdims=True)
    U = U / np.sqrt((U * U).mean(axis=1, keepdims=True) + 1e-12)

    embed = U.astype(np.float32)
    head = np.zeros((2, D), dtype=np.float32)
    head[0, 0] = -c
    head[1, 0] = c
    bias = np.array([-b_prime / 2.0, b_prime / 2.0], dtype=np.float32)

    w["embed.weight"] = embed
    w["head.weight"] = head
    w["head.bias"] = bias
    # keep blocks/pos zero (already zero) and norm gains at 1
    for k in ["pos.weight"]:
        if k in w:
            w[k] = np.zeros_like(w[k])
    for k in ["block0.qkv.weight","block0.qkv.bias","block0.proj.weight","block0.proj.bias",
              "block0.ff1.weight","block0.ff1.bias","block0.ff2.weight","block0.ff2.bias",
              "block1.qkv.weight","block1.qkv.bias","block1.proj.weight","block1.proj.bias",
              "block1.ff1.weight","block1.ff1.bias","block1.ff2.weight","block1.ff2.bias"]:
        if k in w:
            w[k] = np.zeros_like(w[k])
    for k in ["block0.norm1.weight","block0.norm2.weight","block1.norm1.weight",
              "block1.norm2.weight","norm.weight"]:
        if k in w:
            w[k] = np.ones_like(w[k])
    for k in ["block0.norm1.bias","block0.norm2.bias","block1.norm1.bias",
              "block1.norm2.bias","norm.bias"]:
        if k in w:
            w[k] = np.zeros_like(w[k])

    # ---------- 4. self-verify with a NumPy forward pass ----------
    def ln(h):
        mu = h.mean(axis=-1, keepdims=True)
        var = ((h - mu) ** 2).mean(axis=-1, keepdims=True)
        return (h - mu) / np.sqrt(var + eps)

    h = w["embed.weight"][x]                 # [N,T,D], pos=0
    h = ln(h)                                # block0.norm1
    # qkv/proj/ff are zero -> block output = h
    h = ln(h)                                # block1.norm1
    h = ln(h)                                # final norm
    msk = (x != 0).astype(np.float64)[..., None]
    pooled = (h * msk).sum(axis=1) / np.maximum(msk.sum(axis=1), 1.0)
    logits = pooled @ w["head.weight"].T + w["head.bias"]
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())

    diag = {
        "lambda": float(lam),
        "bias_prime": float(b_prime),
        "c": float(c),
        "s_absmax": float(np.abs(s).max()),
        "embed_mean_absmax": float(np.abs(embed.mean(axis=1)).max()),
        "embed_var_mean": float((embed.var(axis=1)).mean()),
        "self_check_accuracy": acc,
        "n_samples": int(N),
        "n_tokens": int(V),
    }
    return {"weights": w, "diagnostics": diag}
