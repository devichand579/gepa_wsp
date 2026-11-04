import numpy as np

def edit(w, f):
    # ---- 0. pull data -------------------------------------------------
    X = f["x"].astype(np.int64)            # (N,64) token ids
    y = f["y"].astype(np.int64)            # (N,)
    N, L = X.shape
    V = w["embed.weight"].shape[0]         # 2000
    D = w["embed.weight"].shape[1]         # 32

    # ---- 1. build normalised bag-of-words counts ----------------------
    mask = (X != 0).astype(np.float32)     # (N,64)
    lens = mask.sum(axis=1)                # (N,)
    lens = np.maximum(lens, 1.0)
    # one-hot style accumulation of counts per vocab id
    counts = np.zeros((N, V), dtype=np.float32)
    rows = np.repeat(np.arange(N), L)
    cols = X.reshape(-1)
    np.add.at(counts, (rows, cols), 1.0)
    # normalise by length (mean pooling) and drop pad column
    counts = counts / lens[:, None]
    C = counts[:, 1:]                      # (N, V-1) drop <pad>

    # ---- 2. closed-form ridge fit of a token score --------------------
    # target: +1 for positive, -1 for negative
    t = np.where(y == 1, 1.0, -1.0).astype(np.float32)
    # standardise columns to help conditioning
    mu = C.mean(axis=0, keepdims=True)
    sd = C.std(axis=0, keepdims=True) + 1e-6
    Cs = (C - mu) / sd
    lam = 1.0
    A = Cs.T @ Cs + lam * np.eye(V - 1, dtype=np.float32)
    b = Cs.T @ t
    try:
        s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(A, b, rcond=None)[0]
    # map back to raw-count space and add intercept
    s_raw = (s / sd.reshape(-1)).astype(np.float32)
    intercept = float(-(mu.reshape(-1) @ s_raw) + t.mean())

    # ---- 3. build embed so that LN(embed[token]) encodes the score ----
    # embed[token] = [score, c, 0, ...]  (score in a bounded range)
    sc = np.zeros(V, dtype=np.float32)
    sc[1:] = s_raw
    # scale scores into a safe range
    scale = 1.0 / (np.max(np.abs(sc)) + 1e-6)
    sc = sc * scale
    c = 1.0
    E = np.zeros((V, D), dtype=np.float32)
    E[:, 0] = sc
    E[:, 1] = c
    # keep <pad> row zero (it is masked out anyway)
    E[0, :] = 0.0
    w["embed.weight"] = E.astype(np.float32)

    # ---- 4. compute exact pooled features for this embed --------------
    # h = embed[ids]; final norm is identity-ish LayerNorm
    H = E[X]                               # (N,64,32)
    # LayerNorm over last dim
    m = H.mean(axis=-1, keepdims=True)
    v = H.var(axis=-1, keepdims=True)
    Hn = (H - m) / np.sqrt(v + 1e-5)
    # masked mean over tokens
    Hn = Hn * mask[..., None]
    pooled = Hn.sum(axis=1) / lens[:, None]   # (N,32)

    # ---- 5. closed-form ridge fit of the head on pooled features ------
    # logit_diff = pooled @ wd + bd ; use ridge on +/-1 target
    P = pooled.astype(np.float32)
    Pm = P.mean(axis=0, keepdims=True)
    Ps = P.std(axis=0, keepdims=True) + 1e-6
    Pz = (P - Pm) / Ps
    lam2 = 1e-3
    A2 = Pz.T @ Pz + lam2 * np.eye(D, dtype=np.float32)
    b2 = Pz.T @ t
    try:
        wd = np.linalg.solve(A2, b2)
    except np.linalg.LinAlgError:
        wd = np.linalg.lstsq(A2, b2, rcond=None)[0]
    wd_raw = (wd / Ps.reshape(-1)).astype(np.float32)
    bd = float(-(Pm.reshape(-1) @ wd_raw) + t.mean())

    # ---- 6. write head: logit1 - logit0 = wd_raw·pooled + bd ----------
    head = np.zeros((2, D), dtype=np.float32)
    head[0, :] = -0.5 * wd_raw
    head[1, :] = 0.5 * wd_raw
    bias = np.array([-0.5 * bd, 0.5 * bd], dtype=np.float32)
    w["head.weight"] = head
    w["head.bias"] = bias

    # ---- 7. keep blocks identity, pos zero, norms identity -----------
    for k in ["block0.norm1.weight", "block0.norm1.bias",
              "block0.norm2.weight", "block0.norm2.bias",
              "block1.norm1.weight", "block1.norm1.bias",
              "block1.norm2.weight", "block1.norm2.bias",
              "norm.weight", "norm.bias"]:
        if k in w:
            if k.endswith("weight"):
                w[k] = np.ones_like(w[k], dtype=np.float32)
            else:
                w[k] = np.zeros_like(w[k], dtype=np.float32)
    for k in ["block0.qkv.weight", "block0.qkv.bias",
              "block0.proj.weight", "block0.proj.bias",
              "block0.ff1.weight", "block0.ff1.bias",
              "block0.ff2.weight", "block0.ff2.bias",
              "block1.qkv.weight", "block1.qkv.bias",
              "block1.proj.weight", "block1.proj.bias",
              "block1.ff1.weight", "block1.ff1.bias",
              "block1.ff2.weight", "block1.ff2.bias",
              "pos.weight"]:
        if k in w:
            w[k] = np.zeros_like(w[k], dtype=np.float32)

    # ---- 8. diagnostics ----------------------------------------------
    diag = {
        "score_scale": float(scale),
        "score_absmax": float(np.max(np.abs(sc))),
        "head_norm": float(np.linalg.norm(wd_raw)),
        "head_bias": float(bd),
        "pooled_std_mean": float(np.mean(Ps)),
        "pooled_absmax": float(np.max(np.abs(pooled))),
        "embed_absmax": float(np.max(np.abs(E))),
    }
    return {"weights": w, "diagnostics": diag}
