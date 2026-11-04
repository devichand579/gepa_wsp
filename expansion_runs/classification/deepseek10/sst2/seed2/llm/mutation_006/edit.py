import numpy as np

def edit(w, f):
    # Extract training data
    x = f["x"]                     # (4000, 64) int64
    y = f["y"].astype(np.int64)    # (4000,)
    N, L = x.shape

    # Current embedding (2000, 32)
    embed = w["embed.weight"].astype(np.float32)

    # We will keep embed as is, zero positional embeddings
    w["pos.weight"][:] = 0.0

    # Set block0 to compute a random ReLU expansion
    # norm1 and norm2 become identity
    w["block0.norm1.weight"][:] = 1.0
    w["block0.norm1.bias"][:] = 0.0
    w["block0.norm2.weight"][:] = 1.0
    w["block0.norm2.bias"][:] = 0.0

    # Zero out attention (qkv, proj)
    w["block0.qkv.weight"][:] = 0.0
    w["block0.qkv.bias"][:] = 0.0
    w["block0.proj.weight"][:] = 0.0
    w["block0.proj.bias"][:] = 0.0

    # Random feed-forward expansion: 32 -> 64 -> 32
    rng = np.random.RandomState(42)
    ff1_w = rng.randn(64, 32).astype(np.float32) * 0.5
    ff1_b = rng.randn(64).astype(np.float32) * 0.1
    ff2_w = rng.randn(32, 64).astype(np.float32) * 0.5
    ff2_b = np.zeros(32, dtype=np.float32)

    w["block0.ff1.weight"][:] = ff1_w
    w["block0.ff1.bias"][:] = ff1_b
    w["block0.ff2.weight"][:] = ff2_w
    w["block0.ff2.bias"][:] = ff2_b

    # Keep block1 zero
    for name in ["block1.norm1.weight", "block1.norm1.bias",
                 "block1.qkv.weight", "block1.qkv.bias",
                 "block1.proj.weight", "block1.proj.bias",
                 "block1.norm2.weight", "block1.norm2.bias",
                 "block1.ff1.weight", "block1.ff1.bias",
                 "block1.ff2.weight", "block1.ff2.bias"]:
        w[name][:] = 0.0
    w["block1.norm1.weight"][:] = 1.0
    w["block1.norm2.weight"][:] = 1.0

    # Final norm identity
    w["norm.weight"][:] = 1.0
    w["norm.bias"][:] = 0.0

    # Compute new pooled features for all training examples
    # h = embed[ids]  (N, L, 32)
    h = embed[x]  # (N, L, 32)

    # block0: h1 = h + ff2(relu(ff1(h)))
    # norm1 is identity, so ff1 input is h
    ff1_out = h @ ff1_w.T + ff1_b  # (N, L, 64)
    relu_out = np.maximum(ff1_out, 0.0)
    ff2_out = relu_out @ ff2_w.T + ff2_b  # (N, L, 32)
    h1 = h + ff2_out

    # final norm (identity) -> h2 = h1
    h2 = h1

    # masked mean pool over non-pad tokens (ids != 0)
    mask = (x != 0).astype(np.float32)  # (N, L)
    mask_sum = mask.sum(axis=1, keepdims=True)  # (N, 1)
    mask_sum = np.maximum(mask_sum, 1.0)  # avoid division by zero
    pooled = (h2 * mask[..., None]).sum(axis=1) / mask_sum  # (N, 32)

    # Fit head via ridge regression: logits = pooled @ W.T + b
    # We use one-hot targets for two classes
    Y = np.zeros((N, 2), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Add bias column
    X = np.concatenate([pooled, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N, 33)

    # Ridge regression with small lambda
    lam = 1.0
    A = X.T @ X + lam * np.eye(33, dtype=np.float32)
    B = X.T @ Y
    W_full = np.linalg.solve(A, B)  # (33, 2)

    head_w = W_full[:32, :].T  # (2, 32)
    head_b = W_full[32, :]     # (2,)

    w["head.weight"][:] = head_w.astype(np.float32)
    w["head.bias"][:] = head_b.astype(np.float32)

    # Diagnostics
    train_logits = X @ W_full
    train_pred = np.argmax(train_logits, axis=1)
    train_acc = float((train_pred == y).mean())

    diag = {
        "train_accuracy": train_acc,
        "head_weight_norm": float(np.linalg.norm(head_w)),
        "head_bias": head_b.tolist(),
        "pooled_std_mean": float(pooled.std(axis=0).mean()),
    }

    return {"weights": w, "diagnostics": diag}
