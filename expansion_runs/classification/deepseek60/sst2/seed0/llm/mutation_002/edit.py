import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # ---- build token-count features (mean over non-pad tokens) ----
    mask = (X != 0)
    counts = np.zeros((N, V), dtype=np.float64)
    for v in range(V):
        counts[:, v] = ((X == v) & mask).sum(axis=1)
    denom = np.maximum(mask.sum(axis=1), 1).astype(np.float64)
    F = counts / denom[:, None]          # [N,V] mean token frequency

    # ---- closed-form ridge regression on counts ----
    # solve for w_v and intercept b:  minimize ||F w + b - t||^2 + lam||w||^2
    # target t = +1 for class1, -1 for class0 (logit difference)
    t = np.where(y == 1, 1.0, -1.0)
    Fc = F - F.mean(axis=0, keepdims=True)
    tc = t - t.mean()
    lam = 1e-2
    A = Fc.T @ Fc + lam * np.eye(V)
    s = np.linalg.solve(A, Fc.T @ tc)     # per-token scores
    b = t.mean() - F.mean(axis=0) @ s     # intercept

    # ---- blend with existing effective scores to avoid regression ----
    # existing per-token score = head_1@norm(embed[v]) - head_0@norm(embed[v])
    emb = w["embed.weight"].astype(np.float64)
    mu = emb.mean(axis=1, keepdims=True)
    sd = emb.std(axis=1, keepdims=True) + 1e-5
    g_old = (emb - mu) / sd               # LayerNorm output
    hw = w["head.weight"].astype(np.float64)
    s_old = g_old @ (hw[1] - hw[0])       # existing per-token score

    # normalize both to comparable scale, then blend
    def nrm(x):
        return x / (np.std(x) + 1e-8)
    s_new = nrm(s)
    s_old_n = nrm(s_old)
    alpha = 0.5
    s_final = alpha * s_new + (1.0 - alpha) * s_old_n
    # rescale so the linear model's intercept/scale is meaningful
    s_final = s_final * (np.std(s) + 1e-8)

    # ---- encode s_final into embed + head ----
    c = 100.0
    g = np.zeros((V, D), dtype=np.float64)
    g[:, 0] = s_final / c
    # remaining 30 dims: one value q, 29 values r  (q + 29r = -g0 ; q^2+29r^2 = D - g0^2)
    g0 = g[:, 0]
    # solve 29 r^2 + ... : use 30 free dims with 2 constraints -> set 30 equal
    # q + 30r = -g0 ; q^2 + 30r^2 = D - g0^2
    a_ = 930.0
    b_ = 60.0 * g0
    c_ = 2.0 * g0**2 - D
    disc = b_**2 - 4 * a_ * c_
    disc = np.maximum(disc, 0.0)
    r = (-b_ + np.sqrt(disc)) / (2 * a_)
    q = -g0 - 30.0 * r
    g[:, 1] = q
    g[:, 2:] = r[:, None]

    # safety: renormalize to zero-mean unit-var exactly
    g = g - g.mean(axis=1, keepdims=True)
    g = g / (g.std(axis=1, keepdims=True) + 1e-8)

    w["embed.weight"] = g.astype(np.float32)

    # head: head_1 - head_0 = c * e_0
    head = np.zeros((2, D), dtype=np.float64)
    head[1, 0] = c
    w["head.weight"] = head.astype(np.float32)
    w["head.bias"] = np.array([-b / 2.0, b / 2.0], dtype=np.float32)

    # zero out blocks and pos to keep identity (they already are, but be safe)
    for k in ["pos.weight"]:
        w[k] = np.zeros_like(w[k])
    for blk in ["block0", "block1"]:
        for sub in ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                    "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{blk}.{sub}"
            if key in w:
                w[key] = np.zeros_like(w[key])

    diag = {
        "s_mean": float(s_final.mean()),
        "s_std": float(s_final.std()),
        "s_absmax": float(np.abs(s_final).max()),
        "intercept": float(b),
        "c": c,
        "alpha": alpha,
        "train_acc_estimate": float(((F @ s_final + b) > 0).astype(np.float64).mean()),
    }
    return {"weights": w, "diagnostics": diag}
