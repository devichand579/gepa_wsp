import numpy as np

def edit(w, f):
    # ---- read feedback ----
    X = np.asarray(f["x"], dtype=np.int64)          # [N,64]
    Y = np.asarray(f["y"], dtype=np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]                  # 2000
    D = w["embed.weight"].shape[1]                  # 32

    # ---- build bag-of-words design matrix (counts, pad=0 excluded) ----
    # Use a compact float32 count matrix.
    C = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        row = X[i]
        nz = row[row != 0]
        if nz.size:
            np.add.at(C[i], nz, 1.0)

    # ---- ridge-regularized least-squares classifier on +/-1 labels ----
    t = np.where(Y == 1, 1.0, -1.0).astype(np.float64)
    # center features (intercept handled separately)
    mu = C.mean(axis=0, dtype=np.float64)
    Cc = C.astype(np.float64) - mu
    # closed-form ridge: w = (X^T X + lam I)^-1 X^T t
    # Use dual form since N < V: w = X^T (X X^T + lam I)^-1 t
    lam = 0.05
    K = Cc @ Cc.T                                   # [N,N]
    K[np.diag_indices_from(K)] += lam
    alpha = np.linalg.solve(K, t)                   # [N]
    wv = Cc.T @ alpha                               # [V]
    # intercept: bias chosen so that mean margin is zero on centered data
    # (centering already removed the mean, so bias = -wv·mu)
    b = -float(wv @ mu)

    # ---- calibrate scale to a reasonable temperature (closed form, no loops) ----
    # raw logits on training data
    z = C.astype(np.float64) @ wv + b               # [N]
    # choose scale s so that std of margins is ~1.5 (well-conditioned softmax)
    sd = float(z.std()) + 1e-12
    s = 1.5 / sd
    wv = wv * s
    b = b * s

    # ---- realize wv in the network ----
    # We want: head.weight[1] . LN(e_v) = wv[v]  (and head.weight[0] = -that)
    # LN(e_v) = (e_v - mean(e_v)) / sqrt(var(e_v) + eps)
    # Choose e_v in a 2-D subspace orthogonal to all-ones:
    #   e_v = r_v * u + q_v * g   with u = ones/sqrt(D), g orthogonal to u, |g|=1
    # Then mean(e_v) = (r_v * sqrt(D))/D = r_v/sqrt(D) along u, and the u-component
    # after centering is zero. The centered vector is q_v * g (plus r_v*(u - u)=0).
    # So LN(e_v) = q_v * g / sqrt(q_v^2/D * D ... ) -> let's compute carefully.
    #
    # Let u = ones/sqrt(D). e_v = a_v * u + q_v * g, g ⟂ u, |g|=1.
    # mean over dims of e_v = a_v * (1/sqrt(D)) * ... actually mean(e_v) = a_v/sqrt(D) * 1? 
    # mean of u = (1/D) * sum(ones)/sqrt(D) = (1/D)*(D)/sqrt(D) = 1/sqrt(D).
    # mean of g = 0. So mean(e_v) = a_v/sqrt(D).
    # centered = e_v - mean(e_v)*ones = a_v*u + q_v*g - (a_v/sqrt(D))*ones
    #          = a_v*u + q_v*g - a_v*sqrt(D)*u = q_v*g  (since ones = sqrt(D)*u)
    # So centered = q_v * g.  var = mean(centered^2) = q_v^2 * mean(g^2) = q_v^2 * (1/D)*D... 
    # mean(g^2) over dims = |g|^2/D = 1/D. So var = q_v^2 / D.
    # LN(e_v) = q_v * g / sqrt(q_v^2/D + eps) ≈ sqrt(D) * g  (independent of q_v!)
    # So LN(e_v) is the SAME direction g for all v. That's degenerate.
    #
    # Fix: use TWO orthogonal directions g1, g2 and set
    #   e_v = a_v*u + p_v*g1 + q_v*g2
    # centered = p_v*g1 + q_v*g2, var = (p_v^2+q_v^2)/D
    # LN(e_v) = (p_v*g1 + q_v*g2) / sqrt((p_v^2+q_v^2)/D + eps)
    #         ≈ sqrt(D) * (p_v*g1+q_v*g2)/sqrt(p_v^2+q_v^2)
    # So LN(e_v) = sqrt(D) * unit vector in the (g1,g2) plane at angle θ_v.
    # We can realize any direction in that plane. Set head.weight[1] = c * g1
    # then head.weight[1]·LN(e_v) = c*sqrt(D)*cos(θ_v). So wv[v] = c*sqrt(D)*cos(θ_v).
    # Given wv[v], we need |wv[v]| <= c*sqrt(D). Choose c = max|wv|/sqrt(D) * 1.0001.
    # Then cos(θ_v) = wv[v]/(c*sqrt(D)), and we can add a second head dim to
    # disambiguate sign via sin. Actually head.weight[1] can have both g1 and g2
    # components: head.weight[1] = c*(g1*cos(φ) + g2*sin(φ)). Then
    # head.weight[1]·LN(e_v) = c*sqrt(D)*(cos(φ)cos(θ_v)+sin(φ)sin(θ_v))
    #                        = c*sqrt(D)*cos(θ_v - φ).
    # So set φ=0 and cos(θ_v) = wv[v]/(c*sqrt(D)). Good, one direction suffices.
    #
    # But we also want head.weight[0] = -head.weight[1] for symmetry, and bias.
    # Actually we need logits = head @ pooled + head.bias. pooled = mean_t LN(e_{x_t}).
    # logits[1] = head.weight[1]·pooled + head.bias[1]
    #           = c*sqrt(D) * mean_t cos(θ_{x_t}) + head.bias[1]
    #           = c*sqrt(D)/... wait we set wv[v] = c*sqrt(D)*cos(θ_v), so
    #           logits[1] = mean_t wv[x_t] + head.bias[1].
    # But our target is logits = mean_t wv[x_t] + b (with the calibrated wv, b).
    # Hmm, but the model uses mean over non-pad tokens. Our C matrix counts non-pad
    # tokens and we divided... let's re-derive: pooled = (1/T) sum_{t: x_t!=0} LN(e_{x_t})
    # where T = number of non-pad tokens. And C[i,v] = count of token v in example i.
    # So pooled_i = (1/T_i) sum_v C[i,v] LN(e_v). And logits[1]_i = head.weight[1]·pooled_i + bias
    # = (1/T_i) sum_v C[i,v] wv[v] + bias.
    # Our ridge fit used C (counts), not normalized. So we need to account for 1/T_i.
    # To keep it simple, we instead fit on the NORMALIZED counts C[i,v]/T_i.
    # Let's redo: define Cn[i,v] = C[i,v]/T_i. Then pooled_i = sum_v Cn[i,v] LN(e_v).
    # So we should fit wv on Cn. Let me use Cn.

    # ---- recompute with normalized counts ----
    T = C.sum(axis=1, keepdims=True)
    T[T == 0] = 1.0
    Cn = C / T                                       # [N,V] normalized counts
    mu = Cn.mean(axis=0, dtype=np.float64)
    Cc = Cn.astype(np.float64) - mu
    K = Cc @ Cc.T
    K[np.diag_indices_from(K)] += lam
    alpha = np.linalg.solve(K, t)
    wv = Cc.T @ alpha
    b = -float(wv @ mu)

    # calibrate scale
    z = Cn.astype(np.float64) @ wv + b
    sd = float(z.std()) + 1e-12
    s = 1.5 / sd
    wv = wv * s
    b = b * s

    # ---- realize in network ----
    eps = 1e-5
    # orthogonal basis in R^D: u = ones/sqrt(D); g1, g2 orthogonal to u and each other
    u = np.ones(D, dtype=np.float64) / np.sqrt(D)
    # g1: e.g. first half positive, second half negative, then orthogonalize to u
    g1 = np.zeros(D, dtype=np.float64)
    g1[: D // 2] = 1.0
    g1[D // 2 :] = -1.0
    g1 -= (g1 @ u) * u
    g1 /= np.linalg.norm(g1)
    # g2: another direction, orthogonalize to u and g1
    g2 = np.zeros(D, dtype=np.float64)
    g2[::2] = 1.0
    g2[1::2] = -1.0
    g2 -= (g2 @ u) * u
    g2 -= (g2 @ g1) * g1
    g2 /= np.linalg.norm(g2)

    # We want head.weight[1]·LN(e_v) = wv[v].
    # LN(e_v) = (p_v*g1 + q_v*g2)/sqrt((p_v^2+q_v^2)/D + eps)
    # Set p_v = cos(θ_v), q_v = sin(θ_v) (so p^2+q^2=1), then
    # LN(e_v) = (cosθ g1 + sinθ g2)/sqrt(1/D + eps) = sqrt(D/(1+eps*D)) * (cosθ g1 + sinθ g2)
    # Let k = sqrt(D/(1+eps*D)).
    # head.weight[1] = A * g1 (say). Then head.weight[1]·LN(e_v) = A*k*cosθ_v.
    # We want this = wv[v]. So cosθ_v = wv[v]/(A*k). Need |wv[v]| <= A*k.
    # Choose A = max|wv| / k * 1.0001 (so |cosθ|<=1).
    k = np.sqrt(D / (1.0 + eps * D))
    maxw = float(np.max(np.abs(wv))) + 1e-12
    A = maxw / k * 1.0001
    cosv = wv / (A * k)
    cosv = np.clip(cosv, -1.0, 1.0)
    sinv = np.sqrt(np.maximum(0.0, 1.0 - cosv ** 2))
    # choose sign of sinv to match... we only need cos for head.weight[1] = A*g1.
    # But to use both dims we could set head.weight[1] = A*(g1*cos(φ)+g2*sin(φ)) with φ=0.
    # So head.weight[1] = A * g1. Then head.weight[1]·LN(e_v) = A*k*cosθ_v = wv[v]. Good.

    # Build embed rows: e_v = a_v*u + p_v*g1 + q_v*g2, with p_v=cosθ_v, q_v=sinθ_v.
    # a_v is free (along u, doesn't affect LN). Set a_v = 0 for simplicity.
    # Then e_v = cosθ_v * g1 + sinθ_v * g2.
    # Check: mean(e_v) = 0 (since g1,g2 ⟂ ones). var = (cos^2+sin^2)/D = 1/D.
    # LN(e_v) = (cosθ g1 + sinθ g2)/sqrt(1/D + eps) = k*(cosθ g1 + sinθ g2). ✓
    theta = np.arctan2(sinv, cosv)
    E = np.outer(np.cos(theta), g1) + np.outer(np.sin(theta), g2)   # [V,D]

    # ---- write weights ----
    w["embed.weight"] = E.astype(np.float32)
    # pos stays zero (already zero)
    # head: row0 = -A*g1, row1 = A*g1 ; bias[0]=-b, bias[1]=b
    hw = np.zeros((2, D), dtype=np.float64)
    hw[0] = -A * g1
    hw[1] = A * g1
    w["head.weight"] = hw.astype(np.float32)
    hb = np.array([-b, b], dtype=np.float64)
    w["head.bias"] = hb.astype(np.float32)

    # ---- diagnostics ----
    # verify realization on a few tokens
    def ln(x):
        m = x.mean(axis=-1, keepdims=True)
        v = x.var(axis=-1, keepdims=True)
        return (x - m) / np.sqrt(v + eps)
    E32 = w["embed.weight"].astype(np.float64)
    LN = ln(E32)                                     # [V,D]
    realized = LN @ w["head.weight"][1].astype(np.float64) + w["head.bias"][1]
    err = float(np.max(np.abs(realized - (wv + b))))
    # training accuracy on full pool
    zt = Cn.astype(np.float64) @ wv + b
    pred = (zt > 0).astype(np.int64)
    train_acc = float((pred == Y).mean())

    diag = {
        "n_samples": int(N),
        "vocab": int(V),
        "ridge_lambda": float(lam),
        "scale": float(s),
        "A": float(A),
        "bias": float(b),
        "realization_max_err": err,
        "train_accuracy": train_acc,
        "w_norm": float(np.linalg.norm(wv)),
        "w_max": float(np.max(np.abs(wv))),
        "nonzero_token_weights": int(np.sum(np.abs(wv) > 1e-9)),
    }
    return {"weights": w, "diagnostics": diag}
