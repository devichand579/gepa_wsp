import numpy as np
import math

def edit(w, f):
    # Extract training data
    x = f["x"]  # [4000, 64]
    y = f["y"]  # [4000]
    N, L = x.shape
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # Build bag-of-words counts for non-padding tokens
    # mask: 1 if token != 0
    mask = (x != 0).astype(np.float32)  # [N, L]
    n_tokens = mask.sum(axis=1)  # [N]
    n_tokens = np.maximum(n_tokens, 1.0)  # avoid division by zero

    # Counts: [N, V]
    counts = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        for j in range(L):
            if x[i, j] != 0:
                counts[i, x[i, j]] += 1.0

    # Length-normalized counts (mean over non-pad tokens)
    X = counts / n_tokens[:, None]  # [N, V]

    # Center the data
    X_mean = X.mean(axis=0)  # [V]
    X_centered = X - X_mean  # [N, V]

    # Compute top-32 principal components using SVD on centered data
    # We only need the right singular vectors (components)
    # To save memory, we can compute the covariance matrix: [V, V] is 2000x2000, which is 4M floats ~16MB, fine.
    # But SVD on [N, V] with N=4000, V=2000 is also fine.
    # Let's use SVD on X_centered (4000x2000). It will be fast.
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Vt shape: [min(N,V), V] = [2000, 2000]? Actually N=4000, V=2000, so Vt is [2000, 2000].
    # The first 32 rows of Vt are the top 32 principal components.
    components = Vt[:D, :]  # [32, V]

    # Project data onto components: [N, 32]
    Z = X_centered @ components.T  # [N, 32]

    # Now fit LDA on Z
    # Class means
    Z0 = Z[y == 0]
    Z1 = Z[y == 1]
    mu0 = Z0.mean(axis=0)
    mu1 = Z1.mean(axis=0)
    # Pooled covariance
    cov0 = np.cov(Z0, rowvar=False)
    cov1 = np.cov(Z1, rowvar=False)
    n0 = Z0.shape[0]
    n1 = Z1.shape[0]
    cov = ((n0 - 1) * cov0 + (n1 - 1) * cov1) / (n0 + n1 - 2)
    # Regularize covariance
    alpha = 1e-2
    cov_reg = cov + alpha * np.eye(D)
    # LDA weight vector
    w_lda = np.linalg.solve(cov_reg, (mu1 - mu0))  # [32]
    # Bias: we want logit difference = w_lda · z + b
    # We can set bias so that threshold is at 0.5 probability.
    # The decision boundary is w_lda · z + b = 0.
    # We can set b = -0.5 * (w_lda · (mu0 + mu1))
    b_lda = -0.5 * np.dot(w_lda, mu0 + mu1)

    # Now we need to map this to the network.
    # The network computes: pooled = mean_i LN(embed[token_i])
    # We want pooled to be Z (or a scaled version) so that head @ pooled + bias gives the LDA logits.
    # We can set embed.weight such that LN(embed[token]) = components[:, token] + X_mean[token]? 
    # Wait: Z = (X - X_mean) @ components.T = sum_t (X_t - X_mean_t) * components[:, t]
    # X_t = count_t / n. So Z = sum_t (count_t / n) * components[:, t] - sum_t X_mean_t * components[:, t]
    # The second term is constant per document? No, sum_t X_mean_t * components[:, t] is a constant vector (independent of document).
    # So Z = (1/n) * sum_t count_t * components[:, t] - const.
    # The network's pooled = (1/n) * sum_t count_t * embed[token_t] (assuming LN is identity).
    # So if we set embed[token] = components[:, token], then pooled = (1/n) * sum_t count_t * components[:, token] = Z + const.
    # Then head @ pooled + bias = head @ Z + head @ const + bias.
    # We can absorb the constant into the bias.
    # So we set embed.weight[token] = components[:, token] (shape [V, 32]).
    # But we need LN(embed[token]) = embed[token] exactly. That requires each embed[token] to have zero mean and unit variance across the 32 dimensions.
    # The components are already centered? Not necessarily. We can normalize each token embedding to have zero mean and unit variance.
    # Let's do that: for each token, compute mean and std across the 32 dimensions, subtract mean, divide by std.
    # This will slightly change the components, but we can just use the normalized version as the new embed.
    # Then the pooled feature will be the average of these normalized vectors.
    # We can then fit the head on the actual pooled features computed with these new embeddings.
    # That's even better: we can compute the new pooled features for all training examples, then fit LDA on those.
    # Let's do that.

    # Step 1: Set embed.weight to components.T (shape [V, 32])
    embed_new = components.T.copy()  # [V, 32]
    # Normalize each token embedding to zero mean and unit variance
    mean_per_token = embed_new.mean(axis=1, keepdims=True)
    std_per_token = embed_new.std(axis=1, keepdims=True)
    std_per_token = np.maximum(std_per_token, 1e-6)
    embed_new = (embed_new - mean_per_token) / std_per_token
    # Now LN(embed_new[token]) = embed_new[token] exactly (since mean=0, std=1, LN with eps=1e-5 will be almost identity)
    # Actually LN normalizes across the 32 dimensions, so if mean=0 and std=1, LN(x) = x / sqrt(1 + eps) ≈ x.
    # So we can just use embed_new.

    # Step 2: Compute the new pooled features for all training examples using embed_new.
    # pooled = (1/n) * sum_{t} count_t * embed_new[token]
    # We can compute this efficiently:
    # counts is [N, V], embed_new is [V, 32]
    # pooled = (counts @ embed_new) / n_tokens[:, None]
    pooled_new = (counts @ embed_new) / n_tokens[:, None]  # [N, 32]

    # Step 3: Fit LDA on pooled_new
    P0 = pooled_new[y == 0]
    P1 = pooled_new[y == 1]
    mu0 = P0.mean(axis=0)
    mu1 = P1.mean(axis=0)
    cov0 = np.cov(P0, rowvar=False)
    cov1 = np.cov(P1, rowvar=False)
    n0 = P0.shape[0]
    n1 = P1.shape[0]
    cov = ((n0 - 1) * cov0 + (n1 - 1) * cov1) / (n0 + n1 - 2)
    alpha = 1e-2
    cov_reg = cov + alpha * np.eye(D)
    w_lda = np.linalg.solve(cov_reg, (mu1 - mu0))  # [32]
    b_lda = -0.5 * np.dot(w_lda, mu0 + mu1)

    # Step 4: Set head weights and bias.
    # We want logit0 - logit1 = w_lda · pooled + b_lda.
    # The network computes logits = head.weight @ pooled + head.bias.
    # We can set head.weight[0] = 0.5 * w_lda, head.weight[1] = -0.5 * w_lda
    # and head.bias[0] = 0.5 * b_lda, head.bias[1] = -0.5 * b_lda.
    # Then logit0 - logit1 = w_lda · pooled + b_lda.
    head_w = np.zeros((2, D), dtype=np.float32)
    head_w[0] = 0.5 * w_lda
    head_w[1] = -0.5 * w_lda
    head_b = np.zeros(2, dtype=np.float32)
    head_b[0] = 0.5 * b_lda
    head_b[1] = -0.5 * b_lda

    # Step 5: Update weights
    w["embed.weight"] = embed_new.astype(np.float32)
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # Ensure all other weights remain as they are (blocks are zero, norms are 1, etc.)
    # But we should ensure pos.weight is zero (it already is).
    # We can leave other weights unchanged.

    # Diagnostics
    # Compute training accuracy with new weights
    logits_new = pooled_new @ head_w.T + head_b  # [N, 2]
    preds = np.argmax(logits_new, axis=1)
    acc = np.mean(preds == y)
    ce = -np.mean(np.log(np.exp(logits_new[np.arange(N), y]) / np.sum(np.exp(logits_new), axis=1)))

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "embed_mean": float(embed_new.mean()),
        "embed_std": float(embed_new.std()),
        "head_w_norm": float(np.linalg.norm(head_w)),
        "b_lda": float(b_lda),
        "n_tokens": int(V),
        "n_samples": int(N)
    }

    return {"weights": w, "diagnostics": diagnostics}
