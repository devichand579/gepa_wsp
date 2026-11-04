import numpy as np
import math

def edit(w, f):
    # ---- read feedback ----
    X = np.asarray(f["x"])          # [N,64] int64
    y = np.asarray(f["y"]).astype(np.int64)   # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # ---- build bag-of-words (excluding PAD id 0) ----
    BOW = np.zeros((N, V), dtype=np.float64)
    mask = (X != 0)
    for i in range(N):
        ids = X[i][mask[i]]
        if ids.size:
            np.add.at(BOW[i], ids, 1.0)

    # ---- closed-form ridge classifier on BOW ----
    # targets: +1 for class 1, -1 for class 0
    t = np.where(y == 1, 1.0, -1.0)
    # centre features to allow an explicit bias
    mu = BOW.mean(axis=0, keepdims=True)
    Bc = BOW - mu
    lam = 1.0
    # solve (Bc^T Bc + lam I) a = Bc^T t
    G = Bc.T @ Bc
    G[np.diag_indices_from(G)] += lam
    rhs = Bc.T @ t
    a = np.linalg.solve(G, rhs)          # [V]
    b = -float(mu @ a) + float(t.mean()) # bias so that f = Bc@a + b

    # w_opt is the direction a (up to scale); keep it as the classifier weight
    w_opt = a.astype(np.float64)
    b_opt = b

    # ---- scale w_opt so each entry is safely inside the LayerNorm feasibility box ----
    max_abs = np.max(np.abs(w_opt)) + 1e-12
    scale = 4.0 / max_abs          # keep |w_opt_scaled| <= 4  (well below sqrt(31))
    ws = w_opt * scale             # [V]

    # ---- build embed.weight: column 0 = ws, other columns chosen so rows are zero-mean/unit-var ----
    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = ws
    # For each row i we need: sum_j E[i,j] = 0  and  sum_j E[i,j]^2 = D
    # With column 0 fixed, the remaining D-1 entries must satisfy
    #   sum = -ws[i],  sum of squares = D - ws[i]^2.
    # Use a simple symmetric construction: fill the remaining entries with
    # a constant c_i plus a small alternating perturbation, then rescale.
    rng = np.random.RandomState(0)
    rem = D - 1
    for i in range(V):
        target_sum = -ws[i]
        target_sq = D - ws[i] * ws[i]
        if target_sq <= 0:
            target_sq = 1e-6
        # start with uniform entries
        c = target_sum / rem
        # base vector of length rem with sum target_sum
        v = np.full(rem, c, dtype=np.float64)
        # add a zero-mean perturbation to control the sum of squares
        pert = rng.randn(rem)
        pert -= pert.mean()
        # current sum of squares
        cur_sq = float(v @ v)
        # scale perturbation so that total sum of squares equals target_sq
        # (v + alpha*pert) has sum = target_sum (since pert sums to 0)
        # and sum of squares = cur_sq + 2*alpha*(v@pert) + alpha^2*(pert@pert)
        A2 = float(pert @ pert)
        B2 = 2.0 * float(v @ pert)
        C2 = cur_sq - target_sq
        if abs(A2) < 1e-12:
            alpha = 0.0
        else:
            disc = B2 * B2 - 4.0 * A2 * C2
            if disc < 0:
                disc = 0.0
            alpha = (-B2 + math.sqrt(disc)) / (2.0 * A2)
        v = v + alpha * pert
        E[i, 1:] = v

    # ---- head: pick out column 0 of E with a large gain, plus bias ----
    # logit_1 - logit_0 = (1/n) * (E @ (v1-v0)) @ BOW + (bias1-bias0)
    # We want this to equal w_opt @ BOW + b_opt.
    # E[:,0] = ws = scale * w_opt, so E @ u with u = [g,0,...,0] gives g*ws.
    # Choose g = 1/scale so that g*ws = w_opt.  Then (1/n)*(w_opt@BOW) + bias.
    # The 1/n factor is positive and does not change the sign, so the decision
    # boundary is w_opt@BOW + n*bias > 0.  We therefore set bias so that the
    # effective affine classifier matches the fitted one: use bias = b_opt / n_avg
    # is not exact per-example; instead absorb the bias by adding a constant to
    # the weight direction (constant * ones adds constant * n to w_opt@BOW).
    # Add constant c0 to w_opt so that c0 * n approximates b_opt.
    n_avg = float(mask.sum(axis=1).mean())
    if n_avg < 1e-6:
        n_avg = 1.0
    c0 = b_opt / n_avg
    w_final = w_opt + c0            # [V]
    # re-scale to keep feasibility
    max_abs2 = np.max(np.abs(w_final)) + 1e-12
    scale2 = 4.0 / max_abs2
    ws2 = w_final * scale2

    # rebuild E with the new column 0
    E2 = np.zeros((V, D), dtype=np.float64)
    E2[:, 0] = ws2
    for i in range(V):
        target_sum = -ws2[i]
        target_sq = D - ws2[i] * ws2[i]
        if target_sq <= 0:
            target_sq = 1e-6
        c = target_sum / rem
        v = np.full(rem, c, dtype=np.float64)
        pert = rng.randn(rem)
        pert -= pert.mean()
        cur_sq = float(v @ v)
        A2 = float(pert @ pert)
        B2 = 2.0 * float(v @ pert)
        C2 = cur_sq - target_sq
        if abs(A2) < 1e-12:
            alpha = 0.0
        else:
            disc = B2 * B2 - 4.0 * A2 * C2
            if disc < 0:
                disc = 0.0
            alpha = (-B2 + math.sqrt(disc)) / (2.0 * A2)
        v = v + alpha * pert
        E2[i, 1:] = v

    # head gain: g = 1/scale2 so that g * ws2 = w_final
    g = 1.0 / scale2
    head_w = np.zeros((2, D), dtype=np.float64)
    head_w[1, 0] = g
    head_w[0, 0] = -g
    # bias: we already absorbed the affine offset into w_final via c0.
    # The remaining (1/n) factor is positive, so no extra bias is needed.
    head_b = np.zeros(2, dtype=np.float64)

    # ---- write back into w (float32, finite, |.|<=1000) ----
    w["embed.weight"] = E2.astype(np.float32)
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # zero out all other parameters (they already are, but be explicit)
    for k in list(w.keys()):
        if k in ("embed.weight", "head.weight", "head.bias"):
            continue
        arr = np.asarray(w[k])
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            w[k] = np.zeros_like(arr, dtype=np.float32)
        # leave integer/bool arrays untouched

    # ---- diagnostics ----
    diag = {
        "method": "closed-form ridge on BOW + LayerNorm-identity embedding",
        "bow_shape": [int(N), int(V)],
        "lambda": float(lam),
        "w_opt_norm": float(np.linalg.norm(w_opt)),
        "w_final_norm": float(np.linalg.norm(w_final)),
        "scale": float(scale2),
        "head_gain": float(g),
        "n_avg": float(n_avg),
        "embed_row_mean_maxabs": float(np.max(np.abs(E2.mean(axis=1)))),
        "embed_row_var_maxdev": float(np.max(np.abs(E2.var(axis=1) - 1.0))),
    }
    return {"weights": w, "diagnostics": diag}
