import numpy as np
import math

def edit(w, f):
    # Extract feedback pool
    X = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, T = X.shape

    # Build bag-of-words counts (excluding padding token 0)
    # We'll use all 2000 vocabulary entries.
    V = w["embed.weight"].shape[0]   # 2000
    counts = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        for t in range(T):
            tok = X[i, t]
            if tok != 0:
                counts[i, tok] += 1.0

    # Length-normalised counts (mean pooling)
    lengths = counts.sum(axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    Xn = counts / lengths   # [N, V]

    # Ridge regression with cross-validation to choose lambda
    # We'll do a simple 5-fold CV on the full pool.
    # To avoid iterative loops, we use closed-form ridge solutions.
    # We'll try a few lambda values and pick the best by CV accuracy.
    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_lambda = None
    best_cv_acc = -1.0
    best_w = None
    best_b = None

    # Precompute for efficiency
    # We'll use the normal equations: w = (X^T X + lambda I)^{-1} X^T y_centered
    # where y is mapped to {-1, +1} for regression.
    y_reg = y.astype(np.float32) * 2.0 - 1.0   # -1 for class 0, +1 for class 1

    # 5-fold CV
    n_folds = 5
    fold_size = N // n_folds
    indices = np.arange(N)
    # Deterministic shuffle
    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    for lam in lambdas:
        cv_accs = []
        for fold in range(n_folds):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_train = Xn[train_idx]
            y_train = y_reg[train_idx]
            X_val = Xn[val_idx]
            y_val = y[val_idx]

            # Add bias by centering
            X_mean = X_train.mean(axis=0, keepdims=True)
            y_mean = y_train.mean()
            Xc = X_train - X_mean
            yc = y_train - y_mean

            # Ridge closed form
            # (Xc^T Xc + lam I) w = Xc^T yc
            A = Xc.T @ Xc + lam * np.eye(V, dtype=np.float32)
            b_vec = Xc.T @ yc
            w_fold = np.linalg.solve(A, b_vec)
            b_fold = y_mean - X_mean @ w_fold

            # Predict on validation
            scores = X_val @ w_fold + b_fold
            preds = (scores > 0).astype(np.int64)
            acc = (preds == y_val).mean()
            cv_accs.append(acc)
        mean_acc = np.mean(cv_accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_lambda = lam

    # Fit final model on all data with best_lambda
    X_mean = Xn.mean(axis=0, keepdims=True)
    y_mean = y_reg.mean()
    Xc = Xn - X_mean
    yc = y_reg - y_mean
    A = Xc.T @ Xc + best_lambda * np.eye(V, dtype=np.float32)
    b_vec = Xc.T @ yc
    w_final = np.linalg.solve(A, b_vec)
    b_final = y_mean - X_mean @ w_final

    # Now we have a linear classifier on length-normalised counts:
    # score = Xn @ w_final + b_final
    # We need to implement this via embed.weight and head.weight.
    # The forward pass: pooled = mean_t LayerNorm(embed[token_t])
    # We want pooled to be a 32-dim vector such that head @ pooled + bias = score.
    # We can set embed.weight[t] = w_final[t] * u + alpha * v,
    # where u and v are fixed zero-mean, unit-variance, orthogonal vectors.
    # Then LayerNorm(embed[t]) ≈ (w_final[t]/alpha) * u + v.
    # So pooled ≈ (mean_t w_final[t] / alpha) * u + v.
    # Then head @ pooled = (mean_t w_final[t] / alpha) * (head @ u) + (head @ v).
    # We want this to equal mean_t w_final[t] + b_final.
    # So we set head @ u = alpha, head @ v = b_final.
    # We also need to handle the fact that LayerNorm is not exactly linear.
    # To minimize distortion, we choose alpha large enough so that |w_final[t]|/alpha is small.
    # Let's compute the max |w_final|.
    max_w = np.max(np.abs(w_final))
    # Choose alpha = 10 * max_w to make the nonlinearity small.
    alpha = 10.0 * max_w + 1e-6

    # Choose u and v as zero-mean, unit-variance, orthogonal vectors in R^32.
    # Simple choice: u = e1 - (1/32)*1, v = e2 - (1/32)*1, then orthogonalize and normalize.
    e1 = np.zeros(32, dtype=np.float32)
    e1[0] = 1.0
    e2 = np.zeros(32, dtype=np.float32)
    e2[1] = 1.0
    ones = np.ones(32, dtype=np.float32) / 32.0
    u = e1 - ones
    v = e2 - ones
    # Orthogonalize v against u
    v = v - (v @ u) / (u @ u) * u
    # Normalize to zero mean and unit variance
    u = u - u.mean()
    u = u / np.sqrt(np.mean(u**2) + 1e-12)
    v = v - v.mean()
    v = v - (v @ u) / (u @ u) * u
    v = v / np.sqrt(np.mean(v**2) + 1e-12)

    # Build embed.weight
    embed = np.zeros((V, 32), dtype=np.float32)
    for t in range(V):
        embed[t] = w_final[t] * u + alpha * v

    # Build head.weight
    # We want head @ pooled + bias = score.
    # pooled ≈ (mean_t w_final[t] / alpha) * u + v.
    # So head @ pooled = (mean_t w_final[t] / alpha) * (head @ u) + (head @ v).
    # We want this to equal mean_t w_final[t] + b_final.
    # So set head @ u = alpha, head @ v = b_final.
    # We have 2 output classes. We can set head.weight[1] - head.weight[0] = alpha * u + b_final * v?
    # Actually, we want logit_1 - logit_0 = score.
    # logit = head.weight @ pooled + head.bias.
    # So (head.weight[1] - head.weight[0]) @ pooled + (bias[1] - bias[0]) = score.
    # Let d = head.weight[1] - head.weight[0]. We want d @ pooled + db = score.
    # pooled = (mean_t w_final[t] / alpha) * u + v.
    # So d @ pooled = (mean_t w_final[t] / alpha) * (d @ u) + (d @ v).
    # We want this to equal mean_t w_final[t] + b_final.
    # So set d @ u = alpha, d @ v = b_final.
    # We can choose d = alpha * u + b_final * v. Then d @ u = alpha (since u,v orthonormal), d @ v = b_final.
    # Perfect.
    d = alpha * u + b_final * v
    # Set head.weight[1] = d, head.weight[0] = -d? Or we can set head.weight[0] = 0, head.weight[1] = d.
    # But we need to keep the shape [2,32]. Let's set head.weight[0] = 0, head.weight[1] = d.
    # Then logit_1 - logit_0 = d @ pooled + (bias[1]-bias[0]).
    # We can set bias[0] = 0, bias[1] = 0. Then difference is d @ pooled.
    # But we need the actual logits to be reasonable. The softmax will be fine.
    # However, the current head has non-zero weights. We can just overwrite.
    head_w = np.zeros((2, 32), dtype=np.float32)
    head_w[0] = 0.0
    head_w[1] = d
    head_b = np.zeros(2, dtype=np.float32)

    # Update weights
    w["embed.weight"] = embed.astype(np.float32)
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # Also zero out pos and all block weights to keep identity (they already are zero, but just in case)
    w["pos.weight"] = np.zeros_like(w["pos.weight"])
    for name in w:
        if name.startswith("block"):
            w[name] = np.zeros_like(w[name])
    # Set all norm weights to 1, biases to 0
    for name in w:
        if "norm" in name and name.endswith("weight"):
            w[name] = np.ones_like(w[name])
        if "norm" in name and name.endswith("bias"):
            w[name] = np.zeros_like(w[name])

    # Diagnostics
    diag = {
        "best_lambda": float(best_lambda),
        "cv_acc": float(best_cv_acc),
        "train_acc": float(((Xn @ w_final + b_final > 0).astype(np.int64) == y).mean()),
        "bias": float(b_final),
        "alpha": float(alpha),
        "max_w": float(max_w),
        "embed_norm_mean": float(np.mean(np.linalg.norm(embed, axis=1))),
        "embed_mean": float(np.mean(embed)),
        "head_d_norm": float(np.linalg.norm(d)),
    }

    return {"weights": w, "diagnostics": diag}
