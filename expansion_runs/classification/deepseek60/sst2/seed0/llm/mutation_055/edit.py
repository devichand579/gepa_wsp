import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]       # 2000

    # ---- 1. bag-of-words counts (exclude pad id 0) ----
    mask = (X != 0)
    counts = np.zeros((N, V), dtype=np.float64)
    for j in range(T):
        col = X[:, j]
        valid = mask[:, j]
        np.add.at(counts, (np.arange(N)[valid], col[valid]), 1.0)
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    bow = counts / denom                   # length-normalised BOW

    # ---- 2. class means / discriminative direction ----
    mu0 = bow[y == 0].mean(axis=0)
    mu1 = bow[y == 1].mean(axis=0)
    d = mu1 - mu0                          # [V]

    # ---- 3. build a 32-dim projection W whose columns are zero-mean, unit-norm ----
    rng = np.random.RandomState(0)
    P = rng.randn(32, V)
    # centre each column (token) across the 32 feature dims
    P = P - P.mean(axis=0, keepdims=True)
    # inject the discriminative direction into the first row
    P[0] = d - d.mean()
    # orthonormalise the rows (Gram-Schmidt) so columns stay zero-mean
    Q, _ = np.linalg.qr(P.T)               # Q: [V,32], orthonormal columns
    W = Q.T                                # [32,V], rows orthonormal
    # re-centre each column and renormalise to unit norm
    W = W - W.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    norms[norms < 1e-8] = 1.0
    W = W / norms

    # ---- 4. pooled features under this embedding ----
    # pooled = mean_t LN(embed[token_t]) = W @ bow
    F = bow @ W.T                          # [N,32]

    # ---- 5. closed-form ridge classifier on F ----
    Y = np.zeros((N, 2), dtype=np.float64)
    Y[np.arange(N), y] = 1.0
    Y = Y - Y.mean(axis=0, keepdims=True)
    Fc = F - F.mean(axis=0, keepdims=True)
    lam = 1e-2
    A = Fc.T @ Fc + lam * np.eye(32)
    B = Fc.T @ Y
    H = np.linalg.solve(A, B).T            # [2,32]
    b = -H @ F.mean(axis=0)                # bias to centre logits

    # ---- 6. write weights ----
    w["embed.weight"] = W.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)
    for name in ["block0.qkv.weight", "block0.qkv.bias",
                 "block0.proj.weight", "block0.proj.bias",
                 "block0.ff1.weight", "block0.ff1.bias",
                 "block0.ff2.weight", "block0.ff2.bias",
                 "block1.qkv.weight", "block1.qkv.bias",
                 "block1.proj.weight", "block1.proj.bias",
                 "block1.ff1.weight", "block1.ff1.bias",
                 "block1.ff2.weight", "block1.ff2.bias"]:
        if name in w:
            w[name] = np.zeros_like(w[name], dtype=np.float32)
    for name in ["block0.norm1.weight", "block0.norm1.bias",
                 "block0.norm2.weight", "block0.norm2.bias",
                 "block1.norm1.weight", "block1.norm1.bias",
                 "block1.norm2.weight", "block1.norm2.bias",
                 "norm.weight", "norm.bias"]:
        if name in w:
            w[name] = np.ones_like(w[name], dtype=np.float32) if "weight" in name else np.zeros_like(w[name], dtype=np.float32)

    w["head.weight"] = H.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)

    # ---- diagnostics ----
    logits = F @ H.T + b
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    diag = {
        "train_accuracy": acc,
        "W_col_norm_mean": float(np.linalg.norm(W, axis=0).mean()),
        "head_norm": float(np.linalg.norm(H)),
        "feature_std": float(F.std()),
        "logit_std": float(logits.std()),
    }
    return {"weights": w, "diagnostics": diag}
