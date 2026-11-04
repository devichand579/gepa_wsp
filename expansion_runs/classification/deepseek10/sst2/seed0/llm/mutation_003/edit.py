import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, T = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build normalized BOW counts
    mask = (x != 0)
    L = mask.sum(axis=1).astype(np.float32)
    L = np.maximum(L, 1.0)
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        tokens = x[i][mask[i]]
        np.add.at(X[i], tokens, 1.0)
    X_norm = X / L[:, None]  # [N, V]

    # One-hot targets
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Ridge regression with bias: augment X_norm with a column of ones
    X_aug = np.hstack([X_norm, np.ones((N, 1), dtype=np.float32)])  # [N, V+1]

    # Choose lambda via a simple holdout split
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_train = 3000
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    X_tr = X_aug[train_idx]
    Y_tr = Y[train_idx]
    X_va = X_aug[val_idx]
    Y_va = Y[val_idx]

    best_lambda = 1.0
    best_acc = -1.0
    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        A = X_tr.T @ X_tr + lam * np.eye(V + 1, dtype=np.float32)
        B = X_tr.T @ Y_tr
        W_aug = np.linalg.solve(A, B)  # [V+1, 2]
        pred = X_va @ W_aug
        acc = np.mean(np.argmax(pred, axis=1) == np.argmax(Y_va, axis=1))
        if acc > best_acc:
            best_acc = acc
            best_lambda = lam

    # Final fit on all data
    A = X_aug.T @ X_aug + best_lambda * np.eye(V + 1, dtype=np.float32)
    B = X_aug.T @ Y
    W_aug = np.linalg.solve(A, B)  # [V+1, 2]
    W = W_aug[:V]      # [V, 2]
    b = W_aug[V]       # [2]

    # Scale factor to keep v_t[:2] small enough for unit variance construction
    max_abs = np.max(np.abs(W))
    c = max(10.0, 100.0 * max_abs)  # large enough

    # Build embeddings v_t for each token
    E = np.zeros((V, 32), dtype=np.float32)
    u = np.tile([1.0, -1.0], 15)  # 30-dim vector, mean 0, sum squares 30
    for t in range(V):
        w0 = W[t, 0] / c
        w1 = W[t, 1] / c
        b0 = -(w0 + w1) / 30.0
        a2 = (32.0 - w0*w0 - w1*w1 - 30.0*b0*b0) / 30.0
        if a2 < 0:
            a2 = 0.0
        a = np.sqrt(a2)
        v = np.empty(32, dtype=np.float32)
        v[0] = w0
        v[1] = w1
        v[2:] = b0 + a * u
        E[t] = v

    # Set weights
    w["embed.weight"][:] = E
    w["pos.weight"][:] = 0.0

    # Make blocks identity: zero residual projections and FFN outputs
    for blk in ["block0", "block1"]:
        w[f"{blk}.proj.weight"][:] = 0.0
        w[f"{blk}.proj.bias"][:] = 0.0
        w[f"{blk}.ff2.weight"][:] = 0.0
        w[f"{blk}.ff2.bias"][:] = 0.0
        # Also zero qkv and ff1 to be safe
        w[f"{blk}.qkv.weight"][:] = 0.0
        w[f"{blk}.qkv.bias"][:] = 0.0
        w[f"{blk}.ff1.weight"][:] = 0.0
        w[f"{blk}.ff1.bias"][:] = 0.0
        # Set norms to identity
        w[f"{blk}.norm1.weight"][:] = 1.0
        w[f"{blk}.norm1.bias"][:] = 0.0
        w[f"{blk}.norm2.weight"][:] = 1.0
        w[f"{blk}.norm2.bias"][:] = 0.0

    # Final norm identity
    w["norm.weight"][:] = 1.0
    w["norm.bias"][:] = 0.0

    # Head reads first two coordinates scaled by c
    w["head.weight"][:] = 0.0
    w["head.weight"][0, 0] = c
    w["head.weight"][1, 1] = c
    w["head.bias"][:] = b

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "val_accuracy": float(best_acc),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": b.tolist(),
        "embed_norm_mean": float(np.mean(np.linalg.norm(w["embed.weight"], axis=1))),
        "c_scale": float(c),
    }
    return {"weights": w, "diagnostics": diagnostics}
