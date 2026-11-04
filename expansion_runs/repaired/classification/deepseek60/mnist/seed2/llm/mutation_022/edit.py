import numpy as np
import math

def edit(w, f):
    # Copy weights to avoid mutating input (though allowed)
    W1 = w['fc1.weight'].copy()   # [128, 784]
    b1 = w['fc1.bias'].copy()     # [128]
    W2 = w['fc2.weight'].copy()   # [10, 128]
    b2 = w['fc2.bias'].copy()     # [10]

    # Feedback data
    X = f['x'].reshape(f['x'].shape[0], -1).astype(np.float64)  # [2000, 784]
    y = f['y'].astype(np.int64)                                 # [2000]
    N = X.shape[0]
    H = f['features'].astype(np.float64)                        # [2000, 128] current ReLU features

    # 1. Optimize fc1 biases via Fisher ratio per unit.
    # Compute current pre-activations Z = X @ W1.T + b1
    Z = X @ W1.T + b1[None, :]  # [2000, 128]

    # For each unit, find threshold t that maximizes Fisher ratio of relu(Z - t).
    # We'll scan over sorted Z values. Since N=2000, this is fast.
    new_b1 = b1.copy()
    for j in range(128):
        z = Z[:, j]
        # Sort z and corresponding labels
        order = np.argsort(z)
        z_sorted = z[order]
        y_sorted = y[order]
        # We want to evaluate t at each z_sorted value (and possibly before first).
        # For a given t, h = max(0, z - t). We can compute class sums and sums of squares.
        # Initialize for t = z_sorted[0] - epsilon (all h=0) -> Fisher ratio 0.
        # We'll iterate through sorted z, adding the current z - t to the class sums.
        # Actually, it's easier to compute for each possible threshold t = z_sorted[k],
        # the values h_i = max(0, z_i - t). For i <= k, z_i <= t, so h_i = 0.
        # For i > k, h_i = z_i - t.
        # So we can compute class sums by iterating from the end.
        # Let's do a backward pass.
        class_sum = np.zeros(10)
        class_sq_sum = np.zeros(10)
        class_count = np.zeros(10)
        # Start with t = z_sorted[-1] (all h=0)
        best_ratio = -1.0
        best_t = z_sorted[-1]
        # We'll iterate k from N-1 down to 0, where t = z_sorted[k].
        # At step k, we include elements i > k. Initially k = N-1, no elements.
        # As we decrease k, we add element k+1? Let's do forward instead.
        # Forward: start with t = z_sorted[0] - 1e-6 (all h > 0? Actually if t < min z, all h = z - t).
        # Then decrease t? No, increasing t reduces h.
        # Let's do: for k from 0 to N-1, t = z_sorted[k]. Elements with index > k have z > t, so h = z - t.
        # Elements with index <= k have h = 0.
        # So we can compute class sums for elements > k by precomputing suffix sums.
        # Let's precompute suffix sums of z and z^2 per class.
        # But classes are mixed. We can compute for each class, the sorted z values and their indices.
        # Simpler: for each unit, loop over k from 0 to N-1, compute class sums from scratch? That's O(N^2) = 4M per unit, 128*4M = 512M, too slow.
        # Better: use cumulative sums. Since we only need class sums of (z - t) for z > t.
        # For a given t, sum_{i: z_i > t} (z_i - t) = sum_{i: z_i > t} z_i - t * count_{i: z_i > t}.
        # We can compute this per class by sorting z within each class? Or we can just sort all z and keep track of class labels.
        # Let's do a single pass over sorted z from largest to smallest.
        # Initialize sums for t = z_sorted[-1] (all h=0).
        # Then for k from N-1 down to 0:
        #   t = z_sorted[k]
        #   Add element k to the sums? Wait, if t = z_sorted[k], then for i > k, z_i > t, so h_i = z_i - t.
        #   For i = k, z_i = t, h_i = 0.
        #   So as we decrease k, we are increasing t? No, z_sorted is increasing. So z_sorted[k] is increasing with k.
        #   If we go from k = N-1 down to 0, t is decreasing. Then more elements have z > t.
        #   Let's go from k = N-1 down to 0. At k = N-1, t = max z. All h=0.
        #   At k = N-2, t = second max. Only the max element has z > t, so h = max_z - t.
        #   So we can maintain sums of (z - t) for elements already processed (those with index > k).
        #   When we move from k to k-1, we add element k to the set of elements with z > t? Actually, at step k, the set is indices > k.
        #   So we can start with empty set, and iterate k from N-1 down to 0:
        #     t = z_sorted[k]
        #     # The set of elements with z > t is exactly those we have added so far (indices > k).
        #     # Compute class sums: for each class c, sum_h[c] = sum_{i in set, y_i=c} (z_i - t)
        #     # = sum_z[c] - t * count[c]
        #     # We can maintain sum_z[c] and count[c] for the set.
        #     # Then compute Fisher ratio.
        #     # Then add element k to the set (for next iteration, where t will be smaller).
        #   This is O(N) per unit.
        sum_z = np.zeros(10)
        count = np.zeros(10)
        sum_z2 = np.zeros(10)  # not needed for Fisher? Actually we need within-class variance of h.
        # For h = z - t, the within-class variance is Var(z) within class, which is independent of t!
        # Wait: h_i = z_i - t for z_i > t, and 0 otherwise. The within-class variance of h depends on t because of the truncation.
        # So we cannot just use sum_z and count. We need sum of h^2 as well.
        # h^2 = (z - t)^2 = z^2 - 2tz + t^2.
        # So we need sum_z2 as well.
        sum_z = np.zeros(10)
        sum_z2 = np.zeros(10)
        count = np.zeros(10)
        best_ratio = -1.0
        best_t = z_sorted[-1]
        # We'll iterate k from N-1 down to 0.
        for k in range(N-1, -1, -1):
            t = z_sorted[k]
            # Current set: indices > k. We have already added indices from N-1 down to k+1.
            # Compute class means and variances of h for this set.
            # For classes with count > 0:
            # mean_h = (sum_z - t*count) / count
            # var_h = (sum_z2 - 2*t*sum_z + t^2*count) / count - mean_h^2
            # Actually, E[h^2] = (sum_z2 - 2*t*sum_z + t^2*count) / count
            # So var_h = E[h^2] - mean_h^2
            # We need overall mean and variance for Fisher ratio.
            # Fisher ratio = between-class variance / within-class variance.
            # Between-class variance = sum_c count_c * (mean_h_c - mean_h)^2 / N
            # Within-class variance = sum_c count_c * var_h_c / N
            # We can compute these.
            total_count = count.sum()
            if total_count > 0:
                mean_h_c = np.zeros(10)
                var_h_c = np.zeros(10)
                for c in range(10):
                    if count[c] > 0:
                        mean_h_c[c] = (sum_z[c] - t * count[c]) / count[c]
                        E_h2 = (sum_z2[c] - 2*t*sum_z[c] + t*t*count[c]) / count[c]
                        var_h_c[c] = E_h2 - mean_h_c[c]**2
                mean_h = (sum_z.sum() - t * total_count) / total_count
                between = 0.0
                within = 0.0
                for c in range(10):
                    if count[c] > 0:
                        between += count[c] * (mean_h_c[c] - mean_h)**2
                        within += count[c] * var_h_c[c]
                between /= total_count
                within /= total_count
                if within > 1e-12:
                    ratio = between / within
                else:
                    ratio = 0.0
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_t = t
            # Add element k to the set for next iteration (t will be smaller)
            c = y_sorted[k]
            z_val = z_sorted[k]
            sum_z[c] += z_val
            sum_z2[c] += z_val * z_val
            count[c] += 1
        # Also consider t = z_sorted[0] - epsilon (all elements included). This is the case where no truncation.
        # We can just check t = z_sorted[0] - 1e-6.
        t = z_sorted[0] - 1e-6
        # All elements included
        mean_h_c = np.zeros(10)
        var_h_c = np.zeros(10)
        for c in range(10):
            if count[c] > 0:
                mean_h_c[c] = (sum_z[c] - t * count[c]) / count[c]
                E_h2 = (sum_z2[c] - 2*t*sum_z[c] + t*t*count[c]) / count[c]
                var_h_c[c] = E_h2 - mean_h_c[c]**2
        total_count = count.sum()
        mean_h = (sum_z.sum() - t * total_count) / total_count
        between = 0.0
        within = 0.0
        for c in range(10):
            if count[c] > 0:
                between += count[c] * (mean_h_c[c] - mean_h)**2
                within += count[c] * var_h_c[c]
        between /= total_count
        within /= total_count
        if within > 1e-12:
            ratio = between / within
        else:
            ratio = 0.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_t = t
        # Update bias: we want new pre-activation Z' = Z - t, so new bias b' = b - t.
        # Because Z = X @ W1.T + b1. Z' = X @ W1.T + b' = Z - t => b' = b1 - t.
        new_b1[j] = b1[j] - best_t

    # 2. Compute new ReLU features with updated biases.
    Z_new = X @ W1.T + new_b1[None, :]
    H_new = np.maximum(0.0, Z_new)  # [2000, 128]

    # 3. Fit fc2 via regularized LDA on H_new.
    # We'll use a small ridge regularization on the within-class covariance.
    # Compute class means and pooled within-class covariance.
    classes = np.unique(y)
    n_classes = len(classes)
    d = H_new.shape[1]
    mean_all = H_new.mean(axis=0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    class_means = np.zeros((n_classes, d))
    for idx, c in enumerate(classes):
        Hc = H_new[y == c]
        mean_c = Hc.mean(axis=0)
        class_means[idx] = mean_c
        diff = Hc - mean_c
        Sw += diff.T @ diff
        diff_b = mean_c - mean_all
        Sb += len(Hc) * np.outer(diff_b, diff_b)
    # Regularize Sw
    alpha = 0.01
    Sw_reg = Sw + alpha * np.eye(d)
    # Solve for LDA directions: W_lda = Sw_reg^{-1} @ class_means.T? Actually for classification, we can use the linear discriminant function:
    # logits = H_new @ W_lda.T + b_lda, where W_lda = Sw_reg^{-1} @ class_means.T (shape d x n_classes)
    # and b_lda = -0.5 * sum(class_means * W_lda.T, axis=1) + log(priors)
    # But we need fc2.weight shape [10, 128] and fc2.bias [10].
    # So W2 = (Sw_reg^{-1} @ class_means.T).T  # [10, 128]
    # b2 = -0.5 * sum(class_means * W2, axis=1) + log(priors)
    # Since priors are roughly equal, we can just use -0.5 * sum(class_means * W2, axis=1).
    Sw_inv = np.linalg.inv(Sw_reg)
    W_lda = Sw_inv @ class_means.T  # [d, n_classes]
    W2_new = W_lda.T  # [n_classes, d]
    b2_new = -0.5 * np.sum(class_means * W2_new, axis=1)
    # Add log priors
    priors = np.array([np.sum(y == c) for c in classes], dtype=np.float64) / N
    b2_new += np.log(priors)

    # Ensure finite and within bounds
    W1 = W1.astype(np.float32)
    new_b1 = new_b1.astype(np.float32)
    W2_new = W2_new.astype(np.float32)
    b2_new = b2_new.astype(np.float32)

    # Clip to [-1000, 1000] just in case
    W1 = np.clip(W1, -1000, 1000)
    new_b1 = np.clip(new_b1, -1000, 1000)
    W2_new = np.clip(W2_new, -1000, 1000)
    b2_new = np.clip(b2_new, -1000, 1000)

    w['fc1.weight'] = W1
    w['fc1.bias'] = new_b1
    w['fc2.weight'] = W2_new
    w['fc2.bias'] = b2_new

    # Diagnostics
    # Compute training accuracy and cross-entropy on the full pool
    logits = H_new @ W2_new.T + b2_new[None, :]
    # Softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    # Cross-entropy
    log_probs = logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
    ce = -np.mean(log_probs[np.arange(N), y])

    diagnostics = {
        'train_accuracy': float(acc),
        'train_cross_entropy': float(ce),
        'fc1_bias_mean': float(new_b1.mean()),
        'fc1_bias_std': float(new_b1.std()),
        'fc2_weight_norm': float(np.linalg.norm(W2_new)),
        'fc2_bias_norm': float(np.linalg.norm(b2_new)),
        'alpha': alpha
    }

    return {'weights': w, 'diagnostics': diagnostics}
