import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]

    mask = (X != 0).astype(np.float32)   # [N,64]
    cnt = mask.sum(1)                    # [N]
    cnt = np.maximum(cnt, 1.0)

    # normalized unigram count features, full vocabulary
    F = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i][mask[i] > 0]
        for t in row:
            F[i, t] += 1.0
    F /= cnt[:, None]

    # target: +1 / -1
    t = np.where(y == 1, 1.0, -1.0)

    # closed-form ridge on log-odds target with intercept
    ones = np.ones((N, 1), dtype=np.float64)
    A = np.concatenate([F, ones], axis=1)   # [N, V+1]

    best = None
    for alpha in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        # K-fold CV
        k = 5
        idx = np.arange(N)
        rng = np.random.RandomState(0)
        rng.shuffle(idx)
        folds = np.array_split(idx, k)
        accs = []
        for j in range(k):
            va = folds[j]
            tr = np.concatenate([folds[m] for m in range(k) if m != j])
            At = A[tr]; tt = t[tr]
            Av = A[va]; tv = t[va]
            G = At.T @ At + alpha * np.eye(V + 1)
            G[-1, -1] -= alpha  # don't regularize intercept
            beta = np.linalg.solve(G, At.T @ tt)
            pred = np.where(Av @ beta >= 0, 1, -1)
            accs.append(np.mean(pred == tv))
        cv = float(np.mean(accs))
        if best is None or cv > best[0]:
            best = (cv, alpha)

    cv_acc, alpha = best

    G = A.T @ A + alpha * np.eye(V + 1)
    G[-1, -1] -= alpha
    beta = np.linalg.solve(G, A.T @ t)   # [V+1]

    s = beta[:V].astype(np.float64)      # per-token score
    b0 = float(beta[V])

    # scale scores so they fit comfortably
    scale = 4.0
    s = s * scale
    b0 = b0 * scale

    # Build embed rows: zero-mean, unit-variance, with dim0 = s(v)
    # We need a vector e with mean 0, var 1, e[0] = s.
    # Choose e[1] = -s (so e[0]+e[1] = 0), and remaining 30 dims = 0.
    # Then mean = 0, var = (s^2 + s^2)/32 = s^2/16. To make var 1, scale by 4/|s|.
    # But that changes e[0]. Instead: use a construction with a free parameter.
    #
    # Simpler: set e[0] = a, e[1] = -a, e[2..31] = c (constant).
    # mean = (a - a + 30c)/32 = 30c/32 = 0.9375c. Set c = 0 for zero mean.
    # Then var = (a^2 + a^2)/32 = a^2/16. For var=1, a = 4.
    # But we want e[0] = s(v), arbitrary. So we need more freedom.
    #
    # Use e[0] = s, and distribute the rest to achieve mean 0 and var 1.
    # Let e[1..31] = q (constant). Then mean = (s + 31q)/32 = 0 -> q = -s/31.
    # var = (s^2 + 31*(s/31)^2)/32 = (s^2 + s^2/31)/32 = s^2*(1 + 1/31)/32 = s^2 * (32/31)/32 = s^2/31.
    # For var = 1, need s^2 = 31, i.e. |s| = sqrt(31). Not arbitrary.
    #
    # General: e[0] = s, e[1..31] = q. mean = (s + 31q)/32 = 0 -> q = -s/31.
    # var = (s^2 + 31 q^2)/32 = (s^2 + 31 s^2/961)/32 = s^2(1 + 1/31)/32 = s^2/31.
    # So var = s^2/31. To get var=1 we need |s| = sqrt(31). Fixed.
    #
    # To allow arbitrary s with var=1, use two free dims:
    # e[0] = s, e[1] = u, e[2..31] = q.
    # mean = (s + u + 30q)/32 = 0.
    # var = (s^2 + u^2 + 30q^2)/32 = 1.
    # Two equations, three unknowns (u, q, and we can pick q). Solve for u, q given s.
    # From mean: u = -s - 30q.
    # var: s^2 + (s+30q)^2 + 30q^2 = 32
    #   s^2 + s^2 + 60sq + 900q^2 + 30q^2 = 32
    #   2s^2 + 60sq + 930q^2 = 32
    #   930 q^2 + 60 s q + (2s^2 - 32) = 0
    #   q = [-60s ± sqrt(3600 s^2 - 4*930*(2s^2-32))] / (2*930)
    #     = [-60s ± sqrt(3600 s^2 - 3720 s^2 + 119040)] / 1860
    #     = [-60s ± sqrt(119040 - 120 s^2)] / 1860
    # Need 119040 - 120 s^2 >= 0 -> s^2 <= 992 -> |s| <= 31.5.
    # With scale=4 and s_absmax ~5.2*4 = 20.8 < 31.5. OK.

    embed = np.zeros((V, D), dtype=np.float32)
    s_absmax = np.max(np.abs(s))
    # clip to be safe
    s = np.clip(s, -30.0, 30.0)

    for v in range(V):
        sv = s[v]
        disc = 119040.0 - 120.0 * sv * sv
        disc = max(disc, 0.0)
        sq = np.sqrt(disc)
        q1 = (-60.0 * sv + sq) / 1860.0
        q2 = (-60.0 * sv - sq) / 1860.0
        # pick the one that gives smaller |u|
        u1 = -sv - 30.0 * q1
        u2 = -sv - 30.0 * q2
        if abs(u1) <= abs(u2):
            q = q1; u = u1
        else:
            q = q2; u = u2
        embed[v, 0] = sv
        embed[v, 1] = u
        embed[v, 2:] = q

    w["embed.weight"] = embed.astype(np.float32)

    # head: read dim0 only. logit1 - logit0 = d * pooled[0]
    # We want logit1 - logit0 = s_mean + b0, where s_mean = mean_t s(v_t).
    # pooled[0] = mean_t LN(embed[v_t])[0] = mean_t s(v_t) (since LN is identity on our rows).
    # So set head[1,0] - head[0,0] = 1, and head[1,1] - head[0,1] = 0.
    # Also head[:,1] should be 0 to not interfere.
    head = np.zeros((2, D), dtype=np.float32)
    head[0, 0] = -0.5
    head[1, 0] = 0.5
    w["head.weight"] = head
    w["head.bias"] = np.array([-b0 / 2.0, b0 / 2.0], dtype=np.float32)

    # zero out blocks (they are already zero, but ensure)
    for name in ["block0.qkv.weight", "block0.qkv.bias", "block0.proj.weight", "block0.proj.bias",
                 "block0.ff1.weight", "block0.ff1.bias", "block0.ff2.weight", "block0.ff2.bias",
                 "block1.qkv.weight", "block1.qkv.bias", "block1.proj.weight", "block1.proj.bias",
                 "block1.ff1.weight", "block1.ff1.bias", "block1.ff2.weight", "block1.ff2.bias"]:
        if name in w:
            w[name] = np.zeros_like(w[name])

    # norms identity
    for name in ["block0.norm1.weight", "block0.norm2.weight",
                 "block1.norm1.weight", "block1.norm2.weight", "norm.weight"]:
        if name in w:
            w[name] = np.ones_like(w[name])
    for name in ["block0.norm1.bias", "block0.norm2.bias",
                 "block1.norm1.bias", "block1.norm2.bias", "norm.bias"]:
        if name in w:
            w[name] = np.zeros_like(w[name])

    # pos zero
    if "pos.weight" in w:
        w["pos.weight"] = np.zeros_like(w["pos.weight"])

    diag = {
        "cv_acc": cv_acc,
        "alpha": alpha,
        "s_absmax": float(s_absmax),
        "s_std": float(np.std(s)),
        "b0": float(b0),
        "scale": scale,
        "embed_nonzero": int(np.count_nonzero(embed)),
    }
    return {"weights": w, "diagnostics": diag}
