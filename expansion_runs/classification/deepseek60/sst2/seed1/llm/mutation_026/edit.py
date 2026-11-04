import numpy as np
import math

def edit(w, f):
    # Extract full training pool
    X = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    F = f["features"]   # [4000, 32] float32, current pooled features
    N, D = F.shape

    # Ensure float64 for stable linear algebra
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    # Standardize features (zero mean, unit variance) to make shrinkage meaningful
    mu = F.mean(axis=0, keepdims=True)
    sd = F.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (F - mu) / sd

    # Class means
    Z0 = Z[y == 0]
    Z1 = Z[y == 1]
    n0, n1 = Z0.shape[0], Z1.shape[0]
    m0 = Z0.mean(axis=0)
    m1 = Z1.mean(axis=0)

    # Pooled within-class covariance
    S0 = (Z0 - m0).T @ (Z0 - m0)
    S1 = (Z1 - m1).T @ (Z1 - m1)
    S = (S0 + S1) / max(1, (n0 + n1 - 2))

    # Shrinkage toward diagonal
    # Choose shrinkage by simple cross-validation on the training pool.
    # We use a small grid; this is closed-form fitting, not iterative optimization.
    best_acc = -1.0
    best_alpha = 0.15
    best_w = None
    best_b = None

    # Precompute for CV: we can do a simple 5-fold split deterministically
    rng = np.random.RandomState(12345)
    idx = rng.permutation(N)
    n_folds = 5
    folds = np.array_split(idx, n_folds)

    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9]:
        # CV accuracy
        accs = []
        for k in range(n_folds):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            Z_tr = Z[tr_idx]
            y_tr = y[tr_idx]
            Z_val = Z[val_idx]
            y_val = y[val_idx]

            # Fit LDA on training fold
            Z0_tr = Z_tr[y_tr == 0]
            Z1_tr = Z_tr[y_tr == 1]
            m0_tr = Z0_tr.mean(axis=0)
            m1_tr = Z1_tr.mean(axis=0)
            S0_tr = (Z0_tr - m0_tr).T @ (Z0_tr - m0_tr)
            S1_tr = (Z1_tr - m1_tr).T @ (Z1_tr - m1_tr)
            S_tr = (S0_tr + S1_tr) / max(1, (Z_tr.shape[0] - 2))
            # Shrink
            S_tr_shrunk = (1 - alpha) * S_tr + alpha * np.diag(np.diag(S_tr))
            # Add tiny ridge for numerical stability
            S_tr_shrunk += 1e-6 * np.eye(D)
            # Direction
            w_dir = np.linalg.solve(S_tr_shrunk, (m1_tr - m0_tr))
            # Bias: threshold at midpoint of projected means
            b = -0.5 * (m1_tr + m0_tr) @ w_dir
            # Predict
            scores = Z_val @ w_dir + b
            pred = (scores > 0).astype(np.int64)
            acc = (pred == y_val).mean()
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_alpha = alpha

    # Fit final LDA on all data with best_alpha
    alpha = best_alpha
    S_shrunk = (1 - alpha) * S + alpha * np.diag(np.diag(S))
    S_shrunk += 1e-6 * np.eye(D)
    w_dir = np.linalg.solve(S_shrunk, (m1 - m0))
    b = -0.5 * (m1 + m0) @ w_dir

    # Transform back to original feature space:
    # Z = (F - mu) / sd  =>  w_dir @ Z + b = (w_dir / sd) @ F + (b - w_dir @ (mu/sd))
    w_orig = w_dir / sd.ravel()
    b_orig = b - np.sum(w_dir * (mu.ravel() / sd.ravel()))

    # The network computes logits = head.weight @ features + head.bias.
    # We want class 1 logit to be w_orig @ features + b_orig, and class 0 logit = 0.
    # So set head.weight[1] = w_orig, head.weight[0] = 0, head.bias[1] = b_orig, head.bias[0] = 0.
    # But the current head has shape [2, 32]. We'll set row 0 to zeros and row 1 to w_orig.
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[1, :] = w_orig.astype(np.float32)
    head_b = np.zeros(2, dtype=np.float32)
    head_b[1] = np.float32(b_orig)

    # Update weights
    w["head.weight"] = head_w
    w["head.bias"] = head_b

    # Optionally zero out other weights to keep the forward pass exactly as assumed.
    # The current weights already have zero blocks, but we can enforce it to be safe.
    for k in list(w.keys()):
        if k.startswith("block") or k == "pos.weight":
            w[k] = np.zeros_like(w[k], dtype=np.float32)
    # Norms: set to identity/zero as in the degenerate case
    for k in ["block0.norm1.weight", "block0.norm1.bias",
              "block0.norm2.weight", "block0.norm2.bias",
              "block1.norm1.weight", "block1.norm1.bias",
              "block1.norm2.weight", "block1.norm2.bias"]:
        if k in w:
            if k.endswith("weight"):
                w[k] = np.ones_like(w[k], dtype=np.float32)
            else:
                w[k] = np.zeros_like(w[k], dtype=np.float32)
    # Final norm: keep as identity (weight=1, bias=0) so features remain as cached
    if "norm.weight" in w:
        w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    if "norm.bias" in w:
        w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # Diagnostics
    # Compute training accuracy of the new head on the full pool
    scores = F @ w_orig + b_orig
    pred = (scores > 0).astype(np.int64)
    train_acc = float((pred == y).mean())
    # Cross-entropy on full pool
    # logits: class 0 = 0, class 1 = scores
    logits = np.stack([np.zeros_like(scores), scores], axis=1)
    # softmax
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    eps = 1e-12
    ce = -np.mean(np.log(probs[np.arange(N), y] + eps))

    diagnostics = {
        "best_alpha": float(best_alpha),
        "cv_accuracy": float(best_acc),
        "train_accuracy": train_acc,
        "train_ce": float(ce),
        "w_norm": float(np.linalg.norm(w_orig)),
        "bias": float(b_orig),
        "n_samples": int(N),
        "n_features": int(D),
    }

    return {"weights": w, "diagnostics": diagnostics}
