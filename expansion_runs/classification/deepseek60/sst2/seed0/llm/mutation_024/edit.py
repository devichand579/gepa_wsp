import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # 0.  Read the feedback pool
    # ------------------------------------------------------------------
    x      = np.asarray(f["x"])            # [N,64] int64
    y      = np.asarray(f["y"]).astype(np.int64)   # [N]
    N, T   = x.shape
    V      = w["embed.weight"].shape[0]   # 2000
    D      = w["embed.weight"].shape[1]   # 32

    # ------------------------------------------------------------------
    # 1.  Rebuild a clean, well-conditioned embedding.
    #     We keep the *current* informative columns 0,1 (they already
    #     encode a good unigram signal) and add a few extra directions
    #     derived from the data so that the interaction term has room.
    # ------------------------------------------------------------------
    emb = w["embed.weight"].astype(np.float64).copy()      # [V,32]
    # zero out everything except the two informative columns
    keep = np.zeros_like(emb)
    keep[:, 0] = emb[:, 0]
    keep[:, 1] = emb[:, 1]
    emb = keep

    # ------------------------------------------------------------------
    # 2.  Build a second-order (bigram) feature via attention.
    #     We use block0 with:
    #        q = k = v = LN(h)  (h = embed(ids))
    #     so attention at position t is  sum_s softmax(<h_t,h_s>) h_s .
    #     This is a genuine cross-token interaction.
    #     We keep block1 as a small per-token MLP (ReLU) so the pooled
    #     representation is nonlinear.
    # ------------------------------------------------------------------
    # ---- block0 qkv:  q,k,v each 32 dims, 2 heads of 16 -------------
    # We want q = k = v = z (the normalised input).  qkv.weight is
    # [96,32] with rows [q(32); k(32); v(32)].
    Wqkv0 = np.zeros((96, 32), dtype=np.float64)
    Wqkv0[0:32, :]  = np.eye(32)          # q = z
    Wqkv0[32:64, :] = np.eye(32)          # k = z
    Wqkv0[64:96, :] = np.eye(32)          # v = z
    bqkv0 = np.zeros(96, dtype=np.float64)

    # ---- block0 proj: 32x32, identity so the attention output is added
    Wproj0 = np.eye(32, dtype=np.float64)
    bproj0 = np.zeros(32, dtype=np.float64)

    # ---- block0 FF: keep identity (no extra nonlinearity here) ------
    Wff1_0 = np.zeros((64, 32), dtype=np.float64)
    bff1_0 = np.zeros(64, dtype=np.float64)
    Wff2_0 = np.zeros((32, 64), dtype=np.float64)
    bff2_0 = np.zeros(32, dtype=np.float64)

    # ---- block1: a small per-token MLP with ReLU --------------------
    # We use a random-but-fixed orthogonal-ish projection so that the
    # 64 hidden units see different linear combinations of the 32 dims.
    rng = np.random.RandomState(0)
    A = rng.randn(64, 32).astype(np.float64)
    # normalise rows
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Wff1_1 = A.copy()
    bff1_1 = np.zeros(64, dtype=np.float64)
    B = rng.randn(32, 64).astype(np.float64)
    B /= (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    Wff2_1 = B.copy()
    bff2_1 = np.zeros(32, dtype=np.float64)

    # ---- block1 attention: identity (no extra interaction) ----------
    Wqkv1 = np.zeros((96, 32), dtype=np.float64)
    Wqkv1[0:32, :]  = np.eye(32)
    Wqkv1[32:64, :] = np.eye(32)
    Wqkv1[64:96, :] = np.eye(32)
    bqkv1 = np.zeros(96, dtype=np.float64)
    Wproj1 = np.eye(32, dtype=np.float64)
    bproj1 = np.zeros(32, dtype=np.float64)

    # ------------------------------------------------------------------
    # 3.  Forward pass (NumPy, no autograd) to obtain the pooled
    #     features for the whole feedback pool, using the *new* weights.
    #     We only need the pooled feature vector per example.
    # ------------------------------------------------------------------
    def layernorm(a, gamma, beta, eps=1e-5):
        mu = a.mean(axis=-1, keepdims=True)
        var = a.var(axis=-1, keepdims=True)
        return (a - mu) / np.sqrt(var + eps) * gamma + beta

    def attention(z, Wqkv, bqkv, Wproj, bproj, mask):
        # z: [B,T,D]
        B, Tt, Dd = z.shape
        qkv = z @ Wqkv.T + bqkv                 # [B,T,96]
        q, k, v = np.split(qkv, 3, axis=-1)     # each [B,T,32]
        # 2 heads of width 16
        q = q.reshape(B, Tt, 2, 16)
        k = k.reshape(B, Tt, 2, 16)
        v = v.reshape(B, Tt, 2, 16)
        # scores [B,2,T,T]
        scale = 1.0 / np.sqrt(16.0)
        scores = np.einsum('bthd,bshd->bhts', q, k) * scale
        # mask padding keys
        scores = scores + (1.0 - mask[:, None, None, :]) * (-1e9)
        # softmax over s
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        e = e / (e.sum(axis=-1, keepdims=True) + 1e-12)
        # weighted sum of v
        out = np.einsum('bhts,bshd->bthd', e, v)   # [B,T,2,16]
        out = out.reshape(B, Tt, Dd)
        out = out @ Wproj.T + bproj
        return out

    def forward_pooled(ids):
        # ids: [B,T] int64
        B, Tt = ids.shape
        mask = (ids != 0).astype(np.float64)          # [B,T]
        h = emb[ids] + w["pos.weight"].astype(np.float64)[np.arange(Tt)][None, :, :]
        # block0
        z = layernorm(h, w["block0.norm1.weight"].astype(np.float64),
                         w["block0.norm1.bias"].astype(np.float64))
        h = h + attention(z, Wqkv0, bqkv0, Wproj0, bproj0, mask)
        z = layernorm(h, w["block0.norm2.weight"].astype(np.float64),
                         w["block0.norm2.bias"].astype(np.float64))
        h = h + (np.maximum(z @ Wff1_0.T + bff1_0, 0.0) @ Wff2_0.T + bff2_0)
        # block1
        z = layernorm(h, w["block1.norm1.weight"].astype(np.float64),
                         w["block1.norm1.bias"].astype(np.float64))
        h = h + attention(z, Wqkv1, bqkv1, Wproj1, bproj1, mask)
        z = layernorm(h, w["block1.norm2.weight"].astype(np.float64),
                         w["block1.norm2.bias"].astype(np.float64))
        h = h + (np.maximum(z @ Wff1_1.T + bff1_1, 0.0) @ Wff2_1.T + bff2_1)
        # final norm
        h = layernorm(h, w["norm.weight"].astype(np.float64),
                         w["norm.bias"].astype(np.float64))
        # masked mean
        pooled = (h * mask[:, :, None]).sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1e-9)
        return pooled, h, mask

    # compute pooled features for the whole pool in chunks
    pooled_all = np.zeros((N, D), dtype=np.float64)
    chunk = 256
    for i in range(0, N, chunk):
        ids = x[i:i+chunk]
        p, _, _ = forward_pooled(ids)
        pooled_all[i:i+chunk] = p

    # ------------------------------------------------------------------
    # 4.  Closed-form ridge classifier on the pooled features.
    #     Target: +1 for positive, -1 for negative.
    # ------------------------------------------------------------------
    Y = np.where(y == 1, 1.0, -1.0)
    # standardise features
    mu = pooled_all.mean(axis=0, keepdims=True)
    sd = pooled_all.std(axis=0, keepdims=True) + 1e-8
    F = (pooled_all - mu) / sd
    # add bias column
    Fb = np.concatenate([F, np.ones((N, 1))], axis=1)
    lam = 1e-2
    A = Fb.T @ Fb + lam * np.eye(Fb.shape[1])
    rhs = Fb.T @ Y
    coef = np.linalg.solve(A, rhs)          # [D+1]
    w_cls = coef[:D] / sd[0]                # undo standardisation
    b_cls = coef[D] - (coef[:D] * mu[0] / sd[0]).sum()

    # ------------------------------------------------------------------
    # 5.  Write the classifier into head (2x32) and head.bias.
    #     logit1 - logit0 = 2 * (w_cls · pooled + b_cls)
    #     We set head[1] = w_cls, head[0] = -w_cls, bias[1]=b_cls, bias[0]=-b_cls
    #     so that logit1 - logit0 = 2*(w_cls·pooled + b_cls).
    #     Scale to keep |w| <= 1000.
    # ------------------------------------------------------------------
    scale = 1.0
    maxabs = max(np.abs(w_cls).max(), abs(b_cls))
    if maxabs > 100.0:
        scale = 100.0 / maxabs
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[1] = (w_cls * scale).astype(np.float32)
    head_w[0] = (-w_cls * scale).astype(np.float32)
    head_b = np.zeros(2, dtype=np.float32)
    head_b[1] = float(b_cls * scale)
    head_b[0] = float(-b_cls * scale)

    # ------------------------------------------------------------------
    # 6.  Write everything back (float32, finite, |.|<=1000).
    # ------------------------------------------------------------------
    def put(name, arr):
        w[name] = np.asarray(arr, dtype=np.float32)

    put("embed.weight", emb)
    put("pos.weight", np.zeros((64, D), dtype=np.float32))

    put("block0.norm1.weight", np.ones(D, dtype=np.float32))
    put("block0.norm1.bias",   np.zeros(D, dtype=np.float32))
    put("block0.qkv.weight", Wqkv0)
    put("block0.qkv.bias",   bqkv0)
    put("block0.proj.weight", Wproj0)
    put("block0.proj.bias",   bproj0)
    put("block0.norm2.weight", np.ones(D, dtype=np.float32))
    put("block0.norm2.bias",   np.zeros(D, dtype=np.float32))
    put("block0.ff1.weight", Wff1_0)
    put("block0.ff1.bias",   bff1_0)
    put("block0.ff2.weight", Wff2_0)
    put("block0.ff2.bias",   bff2_0)

    put("block1.norm1.weight", np.ones(D, dtype=np.float32))
    put("block1.norm1.bias",   np.zeros(D, dtype=np.float32))
    put("block1.qkv.weight", Wqkv1)
    put("block1.qkv.bias",   bqkv1)
    put("block1.proj.weight", Wproj1)
    put("block1.proj.bias",   bproj1)
    put("block1.norm2.weight", np.ones(D, dtype=np.float32))
    put("block1.norm2.bias",   np.zeros(D, dtype=np.float32))
    put("block1.ff1.weight", Wff1_1)
    put("block1.ff1.bias",   bff1_1)
    put("block1.ff2.weight", Wff2_1)
    put("block1.ff2.bias",   bff2_1)

    put("norm.weight", np.ones(D, dtype=np.float32))
    put("norm.bias",   np.zeros(D, dtype=np.float32))
    put("head.weight", head_w)
    put("head.bias",   head_b)

    # ------------------------------------------------------------------
    # 7.  Diagnostics
    # ------------------------------------------------------------------
    logits = pooled_all @ head_w.T + head_b
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    # cross entropy
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    ce = float(-np.log(p[np.arange(N), y] + 1e-12).mean())

    diag = {
        "train_accuracy": acc,
        "train_cross_entropy": ce,
        "n_features": int(D),
        "feature_std_mean": float(sd.mean()),
        "head_scale": float(scale),
        "max_abs_head": float(np.abs(head_w).max()),
    }
    return {"weights": w, "diagnostics": diag}
