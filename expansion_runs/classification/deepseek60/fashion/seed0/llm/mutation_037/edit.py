import numpy as np

def edit(w, f):
    # Extract full feedback features and labels
    X = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"].astype(np.int64)           # (2000,)
    N, D = X.shape

    # Current head weights and bias
    W_old = w["head.weight"].astype(np.float64)  # (10, 1568)
    b_old = w["head.bias"].astype(np.float64)    # (10,)

    # Compute Gram matrix G = X^T X  (1568 x 1568)
    G = X.T @ X

    # Small ridge to approximate projection onto row space of X
    lam = 1e-6
    G_reg = G + lam * np.eye(D, dtype=np.float64)

    # Projected weights: W_new = W_old @ G @ inv(G + lam I)
    # Solve G_reg @ A = G^T? Actually we need W_old @ G @ inv(G_reg).
    # Compute inv(G_reg) efficiently via solve
    inv_G_reg = np.linalg.inv(G_reg)
    W_new = W_old @ G @ inv_G_reg

    # Adjust bias so that the training logits are as close as possible to the old ones.
    # Old logits: X @ W_old.T + b_old
    # New logits: X @ W_new.T + b_new
    # We want X @ W_new.T + b_new ≈ X @ W_old.T + b_old
    # => b_new ≈ b_old + mean(X @ (W_old - W_new).T, axis=0)
    delta_logits = X @ (W_old - W_new).T  # (N, 10)
    b_new = b_old + np.mean(delta_logits, axis=0)

    # Optionally, we can also center the bias to zero mean if desired, but keeping it
    # consistent with the old logits is safer for preserving accuracy.
    # However, the old bias had huge values; after projection the weights are small,
    # so the bias will still be large. To avoid the bias dominating, we can also
    # regularize the bias by subtracting its mean? Actually, the bias is per-class.
    # The old bias had values like -998. If we keep it, the logits will be dominated
    # by the bias, which might be bad. But the old model had huge weights, so the
    # bias was balanced by the weights. After projection, the weights are small,
    # so the bias must also be scaled down. The above adjustment ensures that the
    # mean logit per class matches the old mean logit. But the old mean logits were
    # huge because of the bias. If we keep the bias huge, the new logits will be
    # huge, but the differences between classes will be small because the weights
    # are small. That would destroy accuracy. So we need to also scale the bias
    # down. Actually, the projection method preserves the logits on training data
    # if we use the exact projection (lam=0). But with lam>0, the logits change
    # slightly. The bias adjustment above only corrects the mean. To fully preserve
    # the logits, we should use the exact projection (lam=0) by using the pseudoinverse.
    # Let's do the exact projection using SVD to be safe.

    # Exact projection:
    # Compute SVD of X: X = U S V^T. Row space is spanned by V_r (columns of V with S>0).
    # Projection matrix P = V_r V_r^T.
    # W_new = W_old @ P.
    # Since X is (2000, 1568), we can compute V from SVD of X^T X? Actually, we can use
    # the Gram matrix G = X^T X. Its eigenvectors are V, eigenvalues are S^2.
    # So we can compute eigendecomposition of G.
    # G is 1568x1568. Compute eigenvalues and eigenvectors.
    # For numerical stability, use eigh.
    evals, evecs = np.linalg.eigh(G)
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    # Keep components with eigenvalue > threshold
    tol = 1e-8 * evals[0]
    rank = np.sum(evals > tol)
    V_r = evecs[:, :rank]  # (1568, rank)
    # Projection matrix P = V_r @ V_r.T
    # W_new = W_old @ P
    W_new = W_old @ V_r @ V_r.T

    # Now, to preserve the training logits exactly, we need to adjust the bias.
    # The old logits: X @ W_old.T + b_old
    # New logits: X @ W_new.T + b_new
    # We want X @ W_new.T + b_new = X @ W_old.T + b_old
    # => b_new = b_old + mean(X @ (W_old - W_new).T, axis=0)
    # But since W_new is the projection, X @ W_new.T = X @ W_old.T exactly (up to numerical error).
    # So b_new = b_old. Wait, if W_new = W_old @ P, then X @ W_new.T = X @ P^T @ W_old.T.
    # Since P is projection onto row space of X, X @ P^T = X. So X @ W_new.T = X @ W_old.T.
    # Therefore the logits are exactly the same! So we don't need to change the bias.
    # The bias can remain exactly as it was. The logits on training data will be identical.
    # So the training accuracy will remain 1.0.
    # However, the bias values are huge (e.g., -998). But that's fine because the logits
    # on training data are huge, and the argmax is correct. The huge bias is part of the
    # original solution. But does the huge bias hurt generalization? The bias is a constant
    # per class. It shifts the decision boundary. Since the features are not centered,
    # the bias is necessary to center the logits. The original bias was learned to fit
    # the training data. Keeping it exactly preserves the training logits. So we should
    # keep the bias as is. The only change is the weights, which are now the minimum norm
    # solution that produces the same training logits. This is exactly the projection method.
    # So we don't need to adjust the bias.

    # Let's just keep the bias as is.
    b_new = b_old.copy()

    # Update weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)

    # Diagnostics
    diag = {
        "head_weight_norm_before": float(np.linalg.norm(W_old)),
        "head_weight_norm_after": float(np.linalg.norm(W_new)),
        "head_weight_max_abs_before": float(np.max(np.abs(W_old))),
        "head_weight_max_abs_after": float(np.max(np.abs(W_new))),
        "head_bias_max_abs_before": float(np.max(np.abs(b_old))),
        "head_bias_max_abs_after": float(np.max(np.abs(b_new))),
        "rank": int(rank),
        "lambda_used": 0.0,
    }

    return {"weights": w, "diagnostics": diag}
