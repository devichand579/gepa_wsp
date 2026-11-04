import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    Y = np.asarray(f["y"], dtype=np.int64).astype(np.float64)  # [N]
    N, L = X.shape
    V, D = w["embed.weight"].shape
    H = 64  # ff hidden
    rng = np.random.RandomState(1234)

    # ---------- helpers ----------
    def ln(x, g, b, eps=1e-5):
        m = x.mean(-1, keepdims=True)
        v = x.var(-1, keepdims=True)
        return (x - m) / np.sqrt(v + eps) * g + b

    def softmax_mask(s, mask):
        # s: [B,heads,L,L], mask: [B,L] bool (True=valid key)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        e = e * mask[:, None, None, :]
        e = e / (e.sum(-1, keepdims=True) + 1e-9)
        return e

    def forward_pooled(embed, pos, b0, b1, norm_g, norm_b):
        ids = X
        h = embed[ids] + pos[None, :, :]                     # [N,L,D]
        mask = (ids != 0)
        for blk in (b0, b1):
            z = ln(h, blk["n1g"], blk["n1b"])
            qkv = z @ blk["qkv_w"].T + blk["qkv_b"]          # [N,L,3D]
            q, k, v = np.split(qkv, 3, axis=-1)
            # 2 heads of width D/2
            hh = D // 2
            q = q.reshape(N, L, 2, hh).transpose(0, 2, 1, 3)
            k = k.reshape(N, L, 2, hh).transpose(0, 2, 1, 3)
            v = v.reshape(N, L, 2, hh).transpose(0, 2, 1, 3)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hh)    # [N,2,L,L]
            a = softmax_mask(s, mask)
            att = a @ v                                      # [N,2,L,hh]
            att = att.transpose(0, 2, 1, 3).reshape(N, L, D)
            h = h + att @ blk["proj_w"].T + blk["proj_b"]
            z2 = ln(h, blk["n2g"], blk["n2b"])
            f1 = np.maximum(z2 @ blk["ff1_w"].T + blk["ff1_b"], 0.0)
            h = h + f1 @ blk["ff2_w"].T + blk["ff2_b"]
        h = ln(h, norm_g, norm_b)
        h = h * mask[:, :, None]
        pooled = h.sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
        return pooled

    def ridge_head(P, y, lam):
        # P: [N,D], y: [N] in {0,1} -> fit 2-class linear (one-vs-rest style)
        Y2 = np.stack([1 - y, y], axis=1)                    # [N,2]
        P1 = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)
        A = P1.T @ P1 + lam * np.eye(P1.shape[1])
        W = np.linalg.solve(A, P1.T @ Y2)                    # [D+1,2]
        return W[:-1].T, W[-1]

    def cv_acc(P, y, lam, folds=5):
        idx = np.arange(len(y))
        rng.shuffle(idx)
        accs = []
        for k in range(folds):
            te = idx[k::folds]
            tr = np.setdiff1d(idx, te)
            Wh, bh = ridge_head(P[tr], y[tr], lam)
            pred = np.argmax(P[te] @ Wh.T + bh, axis=1)
            accs.append((pred == y[te]).mean())
        return float(np.mean(accs))

    # ---------- base embedding (keep current rank-2 supervised embedding) ----------
    embed0 = w["embed.weight"].astype(np.float64).copy()
    pos0 = np.zeros((L, D), dtype=np.float64)

    def zero_blk():
        return dict(
            n1g=np.ones(D), n1b=np.zeros(D),
            qkv_w=np.zeros((3 * D, D)), qkv_b=np.zeros(3 * D),
            proj_w=np.zeros((D, D)), proj_b=np.zeros(D),
            n2g=np.ones(D), n2b=np.zeros(D),
            ff1_w=np.zeros((H, D)), ff1_b=np.zeros(H),
            ff2_w=np.zeros((D, H)), ff2_b=np.zeros(D),
        )

    # ---------- candidate A: current model (zero blocks) ----------
    b0a, b1a = zero_blk(), zero_blk()
    Pa = forward_pooled(embed0, pos0, b0a, b1a, np.ones(D), np.zeros(D))
    best = ("A", Pa, 1.0, cv_acc(Pa, Y, 1.0), embed0, pos0, b0a, b1a)

    # ---------- candidate B: bigram via "attend to previous token" + ReLU FFN ----------
    # Build position embeddings P[i] = e_{i} (first L dims of D) so that a linear
    # q/k can make q_i . k_j peak at j = i-1.
    posB = np.zeros((L, D), dtype=np.float64)
    for i in range(L):
        posB[i, i % D] = 1.0
    # scale tokens down so position dominates the q.k score
    embedB = embed0 * 0.15

    for seed in (0, 1, 2):
        r = np.random.RandomState(100 + seed)
        b0 = zero_blk()
        # q = z (identity on the position part), k = shifted position
        # We want q_i . k_j large for j = i-1: set q uses P[i], k uses P[j+1].
        # Implement via qkv: q = Wq z, k = Wk z with Wq = S, Wk = S_shift.
        Wq = np.zeros((D, D)); Wk = np.zeros((D, D)); Wv = np.zeros((D, D))
        for d in range(D):
            Wq[d, d] = 1.0                      # q_i has component e_i
            Wk[(d + 1) % D, d] = 1.0            # k_j has component e_{j+1}
            Wv[d, d] = 1.0                      # v_j = z_j
        b0["qkv_w"][0:D, :] = Wq
        b0["qkv_w"][D:2 * D, :] = Wk
        b0["qkv_w"][2 * D:3 * D, :] = Wv
        b0["proj_w"] = np.eye(D) * 1.0
        # random ReLU FFN on the (token + prev-token) pair
        sc = 1.0 / np.sqrt(D)
        b0["ff1_w"] = r.randn(H, D) * sc * 2.0
        b0["ff1_b"] = r.randn(H) * 0.1
        b0["ff2_w"] = r.randn(D, H) * sc * 2.0
        b1 = zero_blk()
        Pb = forward_pooled(embedB, posB, b0, b1, np.ones(D), np.zeros(D))
        for lam in (0.3, 1.0, 3.0, 10.0):
            a = cv_acc(Pb, Y, lam)
            if a > best[3]:
                best = ("B", Pb, lam, a, embedB, posB, b0, b1)

    # ---------- candidate C: bigram + keep original embedding scale ----------
    for seed in (0, 1):
        r = np.random.RandomState(200 + seed)
        posC = np.zeros((L, D), dtype=np.float64)
        for i in range(L):
            posC[i, i % D] = 1.0
        embedC = embed0.copy()
        b0 = zero_blk()
        Wq = np.zeros((D, D)); Wk = np.zeros((D, D)); Wv = np.zeros((D, D))
        for d in range(D):
            Wq[d, d] = 1.0
            Wk[(d + 1) % D, d] = 1.0
            Wv[d, d] = 1.0
        b0["qkv_w"][0:D, :] = Wq
        b0["qkv_w"][D:2 * D, :] = Wk
        b0["qkv_w"][2 * D:3 * D, :] = Wv
        b0["proj_w"] = np.eye(D) * 1.0
        sc = 1.0 / np.sqrt(D)
        b0["ff1_w"] = r.randn(H, D) * sc * 2.0
        b0["ff1_b"] = r.randn(H) * 0.1
        b0["ff2_w"] = r.randn(D, H) * sc * 2.0
        b1 = zero_blk()
        Pc = forward_pooled(embedC, posC, b0, b1, np.ones(D), np.zeros(D))
        for lam in (0.3, 1.0, 3.0, 10.0):
            a = cv_acc(Pc, Y, lam)
            if a > best[3]:
                best = ("C", Pc, lam, a, embedC, posC, b0, b1)

    # ---------- refit chosen head on full pool ----------
    tag, P, lam, cva, emb, pos, b0, b1 = best
    Wh, bh = ridge_head(P, Y, lam)

    # ---------- write back ----------
    w["embed.weight"] = emb.astype(np.float32)
    w["pos.weight"] = pos.astype(np.float32)
    for name, blk in (("block0", b0), ("block1", b1)):
        w[name + ".norm1.weight"] = blk["n1g"].astype(np.float32)
        w[name + ".norm1.bias"] = blk["n1b"].astype(np.float32)
        w[name + ".qkv.weight"] = blk["qkv_w"].astype(np.float32)
        w[name + ".qkv.bias"] = blk["qkv_b"].astype(np.float32)
        w[name + ".proj.weight"] = blk["proj_w"].astype(np.float32)
        w[name + ".proj.bias"] = blk["proj_b"].astype(np.float32)
        w[name + ".norm2.weight"] = blk["n2g"].astype(np.float32)
        w[name + ".norm2.bias"] = blk["n2b"].astype(np.float32)
        w[name + ".ff1.weight"] = blk["ff1_w"].astype(np.float32)
        w[name + ".ff1.bias"] = blk["ff1_b"].astype(np.float32)
        w[name + ".ff2.weight"] = blk["ff2_w"].astype(np.float32)
        w[name + ".ff2.bias"] = blk["ff2_b"].astype(np.float32)
    w["norm.weight"] = np.ones(D, dtype=np.float32)
    w["norm.bias"] = np.zeros(D, dtype=np.float32)
    w["head.weight"] = Wh.astype(np.float32)
    w["head.bias"] = bh.astype(np.float32)

    # clip for safety
    for k in list(w.keys()):
        w[k] = np.clip(np.nan_to_num(w[k].astype(np.float32), nan=0.0,
                                     posinf=0.0, neginf=0.0), -1000.0, 1000.0)

    train_pred = np.argmax(P @ Wh.T + bh, axis=1)
    diag = {
        "chosen": tag,
        "cv_acc": round(cva, 4),
        "lambda": lam,
        "train_acc": round(float((train_pred == Y).mean()), 4),
        "pooled_std": round(float(P.std()), 4),
        "head_norm": round(float(np.linalg.norm(Wh)), 4),
    }
    return {"weights": w, "diagnostics": diag}
