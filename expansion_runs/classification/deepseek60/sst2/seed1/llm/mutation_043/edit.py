import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]       # vocab size
    D = w["embed.weight"].shape[1]       # 32

    # ---- Build count matrix C [N, V] and length L [N] ----
    C = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        nz = row[row != 0]
        if nz.size:
            np.add.at(C[i], nz, 1.0)
    L = C.sum(axis=1)                    # document lengths (non-pad tokens)
    L = np.maximum(L, 1.0)

    # ---- Targets: +1 for positive, -1 for negative ----
    t = np.where(y == 1, 1.0, -1.0)

    # ---- Design matrix: counts normalized by length, plus a length feature ----
    # We want decision:  (1/L) * C @ w  + b  > 0
    # Equivalently:      C @ w + b*L > 0
    # So fit on [C, L] with targets t.
    Cn = C / L[:, None]                  # normalized counts (what pooled[0] will be)
    A = np.concatenate([Cn, np.ones((N, 1))], axis=1)   # [N, V+1]
    # Ridge closed form:  theta = (A^T A + lam I)^-1 A^T t
    # Use a modest ridge for stability.
    lam = 1.0
    AtA = A.T @ A
    AtA[np.diag_indices_from(AtA)] += lam
    Aty = A.T @ t
    try:
        theta = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, t, rcond=None)[0]

    w_counts = theta[:V]                 # per-token weights
    b_len = theta[V]                     # length coefficient
    # We will implement:  pooled[0] = (1/L) sum_j c_j * w_j
    # head: logit1 - logit0 = pooled[0] + b_len  (b_len acts as bias)
    # So set head[0] = 0 vector, head[1] = e_0, and bias difference = b_len.

    # ---- Scale weights so that LN construction is feasible ----
    # We need v_j[0] = w_j, and v_j zero-mean unit-std over D=32 dims.
    # Feasibility: sum(v_j)=0, sum(v_j^2)=D. With v_j[0]=w_j:
    #   sum_{k>=1} v_j[k] = -w_j
    #   sum_{k>=1} v_j[k]^2 = D - w_j^2
    # Need D - w_j^2 >= 0  => |w_j| <= sqrt(D).
    # Also need the remaining components to be able to sum to -w_j with that
    # sum of squares: by Cauchy-Schwarz, (sum)^2 <= 31 * (sum of squares),
    # i.e. w_j^2 <= 31*(D - w_j^2)  =>  w_j^2 <= 31*D/32.
    # So clamp |w_j| <= sqrt(31*D/32) ~ sqrt(31) ~ 5.568.
    max_w = np.sqrt(31.0 * D / 32.0) * 0.98
    # Also rescale to keep things reasonable.
    scale = np.max(np.abs(w_counts)) / max_w if np.max(np.abs(w_counts)) > 0 else 1.0
    if scale > 1.0:
        w_counts = w_counts / scale
        b_len = b_len / scale

    # ---- Build embed.weight ----
    # For each token j: choose v_j in R^D with v_j[0] = w_j, mean 0, std 1.
    # Simple construction: put the remaining mass symmetrically.
    # Let r = -w_j / (D-1)  (so sum of remaining = -w_j)
    # Then remaining components all equal r. Sum of squares = (D-1)*r^2.
    # We need (D-1)*r^2 + w_j^2 = D  =>  w_j^2/(D-1) + w_j^2 = D
    #   => w_j^2 * (1 + 1/(D-1)) = D  => w_j^2 * D/(D-1) = D
    #   => w_j^2 = D-1.  That only works for one specific w_j.
    # So use a two-parameter construction: set v_j[1] = a, v_j[2] = b,
    # and remaining equal to c, such that sum = 0 and sum sq = D.
    # Simpler: use v_j[1] = s, v_j[2] = -s, and rest = 0, then adjust.
    # Let's do: v_j[0] = w_j. Let the rest be: one component = p, one = q,
    # and the other D-3 components = r each.
    # Constraints: p + q + (D-3) r = -w_j
    #              p^2 + q^2 + (D-3) r^2 = D - w_j^2
    # Choose r = -w_j / (D-1) (mean-zero spread), then
    #   p + q = -w_j - (D-3) r = -w_j + (D-3) w_j/(D-1) = -w_j * (1 - (D-3)/(D-1))
    #         = -w_j * (2/(D-1))
    #   p^2 + q^2 = D - w_j^2 - (D-3) r^2
    #             = D - w_j^2 - (D-3) w_j^2/(D-1)^2
    # Need p^2+q^2 >= (p+q)^2/2.  If not, fall back to uniform.
    # To keep it simple and robust, use a direct approach:
    #   v_j = w_j * e_0 + u_j, where u_j is chosen to fix mean and norm.
    # We'll build u_j by starting from a fixed zero-mean unit-norm vector
    # orthogonal to e_0, then scale/shift. But exact feasibility is easier
    # with a numerical projection.

    # Robust numerical construction:
    # Start with a base zero-mean, unit-norm vector in R^D orthogonal to e_0.
    base = np.zeros(D, dtype=np.float64)
    # Use a simple pattern: alternate +/- to get zero mean, orthogonal to e_0.
    # e.g. base[1] = 1/sqrt(2), base[2] = -1/sqrt(2), rest 0 -> mean 0, norm 1.
    base[1] = 1.0 / np.sqrt(2.0)
    base[2] = -1.0 / np.sqrt(2.0)
    # Now for each token, we want v_j = w_j * e_0 + alpha_j * base + beta_j * 1_vec?
    # But 1_vec is not orthogonal to e_0 and has nonzero mean.
    # Let's instead directly solve for v_j given w_j using a small least-squares
    # over a 2-dim family: v_j = w_j e_0 + a_j * base + b_j * ones.
    # Constraints: mean(v_j)=0, ||v_j||^2 = D.
    # mean(v_j) = (w_j + a_j*mean(base) + b_j*D)/D = (w_j + 0 + b_j*D)/D = 0
    #   => b_j = -w_j / D.
    # ||v_j||^2 = w_j^2 + a_j^2 + b_j^2 * D  (since base orthogonal to e_0 and ones)
    #   (base has mean 0, so orthogonal to ones; base orthogonal to e_0)
    #   => a_j^2 = D - w_j^2 - b_j^2 * D = D - w_j^2 - w_j^2/D = D - w_j^2 (1 + 1/D)
    # Need a_j^2 >= 0  => w_j^2 <= D / (1 + 1/D) = D^2/(D+1).
    # For D=32: D^2/(D+1) = 1024/33 ~ 31.03, sqrt ~ 5.57.
    # Good, similar bound.
    max_w2 = D * D / (D + 1.0)
    max_w = np.sqrt(max_w2) * 0.98
    scale = np.max(np.abs(w_counts)) / max_w if np.max(np.abs(w_counts)) > 0 else 1.0
    if scale > 1.0:
        w_counts = w_counts / scale
        b_len = b_len / scale

    ones = np.ones(D, dtype=np.float64)
    base = np.zeros(D, dtype=np.float64)
    base[1] = 1.0 / np.sqrt(2.0)
    base[2] = -1.0 / np.sqrt(2.0)
    # base is zero-mean, unit-norm, orthogonal to e_0 and ones.

    Vmat = np.zeros((V, D), dtype=np.float64)
    for j in range(V):
        wj = w_counts[j]
        bj = -wj / D
        aj2 = D - wj * wj * (1.0 + 1.0 / D)
        aj = np.sqrt(max(aj2, 0.0))
        Vmat[j] = wj * np.eye(D)[0] + aj * base + bj * ones
    # Verify: mean 0, ||.||^2 = D  (so std 1)
    # Now embed.weight must satisfy LN(embed_j) = Vmat[j] with weight=1, bias=0.
    # LN(x) = (x - mean(x)) / sqrt(var(x) + eps).  With eps=1e-5.
    # We want LN(x) = v.  Choose x = m + s * v with s chosen so that
    # var(x) + eps = s^2 * var(v) = s^2 * 1  => s = sqrt(var(x)+eps).
    # But var(x) = s^2 * var(v) = s^2. So s^2 = s^2 + eps -> only eps=0.
    # With eps>0, LN(x) = (x - mean)/sqrt(var+eps) = s*v / sqrt(s^2 + eps).
    # We want this = v, so s / sqrt(s^2+eps) = 1 -> impossible for eps>0.
    # So exact equality is impossible with eps>0.  But eps=1e-5 is tiny;
    # we can make s large so that s/sqrt(s^2+eps) ~ 1 - eps/(2 s^2).
    # Choose s large (e.g. s=100) to make the error negligible.
    # Also add a mean offset m (doesn't affect LN).
    s = 100.0
    m = 0.0
    embed_new = m + s * Vmat   # [V, D]
    # LN(embed_new) = s*Vmat / sqrt(s^2*1 + eps) = Vmat * s/sqrt(s^2+eps)
    # ~ Vmat * (1 - eps/(2 s^2)) ~ Vmat for s=100, eps=1e-5.
    # Good enough.

    # Clamp to be safe
    embed_new = np.clip(embed_new, -1000.0, 1000.0).astype(np.float32)
    w["embed.weight"] = embed_new

    # ---- Set pos to zero (already zero, but ensure) ----
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # ---- Keep blocks as identity: zero all block weights/biases ----
    for k in list(w.keys()):
        if k.startswith("block0.") or k.startswith("block1."):
            w[k] = np.zeros_like(w[k], dtype=np.float32)

    # ---- Final norm: weight=1, bias=0 ----
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # ---- Head: logit1 - logit0 = pooled[0] + b_len ----
    # pooled[0] = (1/L) sum_j c_j * w_j  (since LN(embed_j)[0] = w_j)
    # We want decision: pooled[0] + b_len > 0.
    # Set head[0] = 0, head[1] = e_0, bias[0] = 0, bias[1] = b_len.
    head_w = np.zeros_like(w["head.weight"], dtype=np.float32)
    head_w[1, 0] = 1.0
    w["head.weight"] = head_w
    head_b = np.zeros_like(w["head.bias"], dtype=np.float32)
    head_b[0] = 0.0
    head_b[1] = float(b_len)
    w["head.bias"] = head_b

    # ---- Diagnostics ----
    # Compute training accuracy of the resulting linear classifier on the pool.
    scores = Cn @ w_counts + b_len
    preds = (scores > 0).astype(np.int64)
    train_acc = float(np.mean(preds == y))
    diag = {
        "method": "closed_form_ridge_linear_classifier_on_counts_via_LN_embed",
        "train_accuracy": train_acc,
        "n_tokens": int(V),
        "w_counts_absmax": float(np.max(np.abs(w_counts))),
        "b_len": float(b_len),
        "scale_used": float(scale),
        "embed_absmax": float(np.max(np.abs(embed_new))),
        "embed_mean": float(np.mean(embed_new)),
    }
    return {"weights": w, "diagnostics": diag}
