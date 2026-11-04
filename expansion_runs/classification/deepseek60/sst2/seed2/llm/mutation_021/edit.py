import numpy as np

def edit(w, f):
    # ---- read feedback ----
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]       # 32

    mask = (X != 0).astype(np.float32)   # [N,64]
    ntok = np.maximum(mask.sum(1), 1.0)  # [N]

    # ---- 1) Build a BOW matrix (N x V) ----
    BOW = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)
    cols = X.reshape(-1)
    np.add.at(BOW, (rows, cols), 1.0)
    BOWn = BOW / ntok[:, None]           # normalized BOW

    # ---- 2) Fit a ridge linear classifier on BOWn (closed form) ----
    Y = np.where(y == 1, 1.0, -1.0)
    # center features for a fitted intercept
    mu = BOWn.mean(0, keepdims=True)
    Bc = BOWn - mu
    G = Bc.T @ Bc                        # V x V
    b = Bc.T @ Y                         # V
    I = np.eye(V)

    def fit(lam):
        s = np.linalg.solve(G + lam * I, b)
        c = Y.mean() - float(mu @ s)
        return s, c

    # 5-fold CV to pick lambda
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)
    best_lam, best_acc = 1.0, -1.0
    for lam in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        accs = []
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            mu_k = BOWn[tr].mean(0, keepdims=True)
            Bc_k = BOWn[tr] - mu_k
            Gk = Bc_k.T @ Bc_k
            bk = Bc_k.T @ Y[tr]
            sk = np.linalg.solve(Gk + lam * I, bk)
            ck = Y[tr].mean() - float(mu_k @ sk)
            pred = (BOWn[te] @ sk + ck) > 0
            accs.append((pred == (y[te] == 1)).mean())
        a = float(np.mean(accs))
        if a > best_acc:
            best_acc, best_lam = a, lam
    s, c = fit(best_lam)

    # ---- 3) Write the classifier into column 0 of embed, keep rows zero-mean/unit-var ----
    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = s
    # fill remaining columns with a small deterministic orthogonal-ish pattern
    # so that each row is exactly zero-mean and unit-variance.
    # Use a fixed orthonormal basis for the remaining D-1 dims.
    Q = np.zeros((D, D - 1), dtype=np.float64)
    for j in range(D - 1):
        Q[:j + 1, j] = 1.0
        Q[j + 1, j] = -(j + 1.0)
        Q[:, j] /= np.linalg.norm(Q[:, j])
    # make Q orthogonal to the all-ones direction implicitly by construction
    # (the columns above are already zero-sum)
    # assign tiny structured noise so rows are zero-mean/unit-var
    rng2 = np.random.RandomState(1234)
    Z = rng2.randn(V, D - 1) * 1e-3
    Z = Z - Z.mean(1, keepdims=True)
    Z = Z / (Z.std(1, keepdims=True) + 1e-12)
    E[:, 1:] = Z

    # enforce exact zero-mean / unit-variance per row (LN identity)
    E = E - E.mean(1, keepdims=True)
    E = E / (E.std(1, keepdims=True) + 1e-12)

    # ---- 4) Head reads column 0 ----
    head_w = np.zeros((2, D), dtype=np.float64)
    head_w[1, 0] = 1.0
    head_w[0, 0] = -1.0
    head_b = np.array([-c, c], dtype=np.float64)

    # ---- 5) Zero out transformer blocks / pos so the network is exactly the read-out ----
    for k in list(w.keys()):
        if k.startswith("block0.") or k.startswith("block1."):
            w[k] = np.zeros_like(w[k], dtype=np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for k in ["block0.norm1.weight", "block0.norm2.weight",
              "block1.norm1.weight", "block1.norm2.weight",
              "norm.weight"]:
        w[k] = np.ones_like(w[k], dtype=np.float32)
    for k in ["block0.norm1.bias", "block0.norm2.bias",
              "block1.norm1.bias", "block1.norm2.bias",
              "norm.bias"]:
        w[k] = np.zeros_like(w[k], dtype=np.float32)

    w["embed.weight"] = E.astype(np.float32)
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # ---- diagnostics ----
    train_pred = (BOWn @ s + c) > 0
    train_acc = float((train_pred == (y == 1)).mean())
    diag = {
        "method": "BOW ridge + LN-identity embedding",
        "best_lambda": float(best_lam),
        "cv_accuracy": float(best_acc),
        "train_accuracy": train_acc,
        "s_norm": float(np.linalg.norm(s)),
        "intercept": float(c),
        "embed_row_mean_maxabs": float(np.abs(E.mean(1)).max()),
        "embed_row_std_maxdev": float(np.abs(E.std(1) - 1.0).max()),
    }
    return {"weights": w, "diagnostics": diag}
