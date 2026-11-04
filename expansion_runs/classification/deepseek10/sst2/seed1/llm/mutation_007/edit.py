import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    Y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    mask = (X != 0)                      # [N,64]
    L = mask.sum(1).astype(np.float64)   # [N]
    L[L == 0] = 1.0

    # ---- closed-form bag-of-words token weights (ridge on counts) ----
    # build sparse count matrix implicitly via np.add.at on a dense [V] accumulator
    # We use a one-vs-rest style LDA/ridge: solve (C^T C + lam I) w = C^T y
    # C is [N, V] bag-of-words counts (excluding pad).
    # Accumulate C^T C and C^T y without materialising C.
    lam = 2.0
    CtC = np.zeros((V, V), dtype=np.float64)
    Cty = np.zeros(V, dtype=np.float64)
    # group rows by identical bag-of-words is expensive; do it per example
    # Use a hash of the sorted token multiset to batch identical bags.
    # Simpler: accumulate directly (4000 * 64^2 = 16M ops, fine).
    for i in range(N):
        ids = X[i][mask[i]]
        if ids.size == 0:
            continue
        # counts
        cnt = np.bincount(ids, minlength=V).astype(np.float64)
        CtC += np.outer(cnt, cnt)
        Cty += cnt * (1.0 if Y[i] == 1 else -1.0)
    A = CtC + lam * np.eye(V)
    try:
        w_tok = np.linalg.solve(A, Cty)
    except np.linalg.LinAlgError:
        w_tok = np.linalg.lstsq(A, Cty, rcond=None)[0]
    # normalise so magnitudes are sane
    s = np.std(w_tok)
    if s > 0:
        w_tok = w_tok / s
    w_tok[0] = 0.0  # pad never contributes

    # ---- design embed so LayerNorm is ~linear in w_tok ----
    # embed(token) = M * carrier + w_tok(token) * signal_dir
    # carrier: alternating +/- pattern with large magnitude -> dominates variance
    # signal_dir: e_0 so that after LayerNorm the linear correction lives in dim 0
    M = 8.0
    carrier = np.empty(D, dtype=np.float64)
    carrier[0::2] = 1.0
    carrier[1::2] = -1.0
    signal_dir = np.zeros(D, dtype=np.float64)
    signal_dir[0] = 1.0

    E = np.zeros((V, D), dtype=np.float64)
    E[:, :] = M * carrier[None, :]
    E[:, 0] += w_tok            # add signal on dim 0
    # keep pad row at zero so masked-mean ignores it anyway
    E[0, :] = 0.0

    # ---- closed-form head fit on the *new* pooled features ----
    # Compute the pooled features the new embed will produce.
    # LayerNorm per token (eps=1e-5, gamma=1,beta=0), then masked mean.
    eps = 1e-5
    # gather token embeddings for the whole pool (memory: 4000*64*32*8 = 65MB ok)
    emb = E[X]                              # [N,64,32]
    mu = emb.mean(axis=2, keepdims=True)
    var = emb.var(axis=2, keepdims=True)
    ln = (emb - mu) / np.sqrt(var + eps)    # [N,64,32]
    ln = ln * mask[:, :, None]
    pooled = ln.sum(axis=1) / L[:, None]    # [N,32]

    # ridge fit of head on pooled features -> target +1/-1
    tgt = np.where(Y == 1, 1.0, -1.0)
    P = pooled.astype(np.float64)
    Ph = np.concatenate([P, np.ones((N, 1))], axis=1)   # [N,33]
    lam_h = 1e-3
    G = Ph.T @ Ph + lam_h * np.eye(33)
    rhs = Ph.T @ tgt
    try:
        sol = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(G, rhs, rcond=None)[0]
    head_w = sol[:32]
    head_b = sol[32]

    # scale so logits are well-conditioned (softmax friendly)
    scale = 4.0 / (np.abs(head_w).max() + 1e-9)
    head_w = head_w * scale
    head_b = head_b * scale

    # ---- write weights (keep all tensors finite, |.|<=1000) ----
    w["embed.weight"] = np.clip(E, -1000, 1000).astype(np.float32)

    # zero out transformer blocks (identity), keep norms identity
    for name in ["block0.qkv.weight", "block0.qkv.bias",
                 "block0.proj.weight", "block0.proj.bias",
                 "block0.ff1.weight", "block0.ff1.bias",
                 "block0.ff2.weight", "block0.ff2.bias",
                 "block1.qkv.weight", "block1.qkv.bias",
                 "block1.proj.weight", "block1.proj.bias",
                 "block1.ff1.weight", "block1.ff1.bias",
                 "block1.ff2.weight", "block1.ff2.bias"]:
        w[name] = np.zeros_like(w[name], dtype=np.float32)

    for name in ["block0.norm1.weight", "block0.norm1.bias",
                 "block0.norm2.weight", "block0.norm2.bias",
                 "block1.norm1.weight", "block1.norm1.bias",
                 "block1.norm2.weight", "block1.norm2.bias",
                 "norm.weight", "norm.bias"]:
        if name.endswith("weight"):
            w[name] = np.ones_like(w[name], dtype=np.float32)
        else:
            w[name] = np.zeros_like(w[name], dtype=np.float32)

    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # head: row0 = -head_w/2, row1 = +head_w/2  (so argmax matches sign)
    hw = head_w.astype(np.float64)
    hb = head_b
    head = np.zeros((2, 32), dtype=np.float64)
    head[0] = -hw * 0.5
    head[1] = hw * 0.5
    bias = np.array([-hb * 0.5, hb * 0.5], dtype=np.float64)
    w["head.weight"] = np.clip(head, -1000, 1000).astype(np.float32)
    w["head.bias"] = np.clip(bias, -1000, 1000).astype(np.float32)

    # ---- diagnostics ----
    logits = P @ head_w + head_b
    pred = (logits > 0).astype(np.int64)
    acc = float((pred == Y).mean())
    diag = {
        "train_accuracy": acc,
        "w_tok_norm": float(np.linalg.norm(w_tok)),
        "head_w_norm": float(np.linalg.norm(head_w)),
        "head_bias": float(head_b),
        "mean_len": float(L.mean()),
        "pooled_mean_class0": pooled[Y == 0].mean(0).round(4).tolist(),
        "pooled_mean_class1": pooled[Y == 1].mean(0).round(4).tolist(),
    }
    return {"weights": w, "diagnostics": diag}
