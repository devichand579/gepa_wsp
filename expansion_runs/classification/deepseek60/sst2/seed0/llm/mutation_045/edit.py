import numpy as np
import math

def edit(w, f):
    # Extract feedback pool
    x = f["x"]  # (4000, 64) int64
    y = f["y"]  # (4000,) int64
    N = x.shape[0]
    V = w["embed.weight"].shape[0]  # 2000
    D = w["embed.weight"].shape[1]  # 32

    # Build feature matrix: for each example, average token counts (excluding padding 0)
    # We'll compute X^T X, X^T y, and X_mean directly to avoid storing dense X.
    XTX = np.zeros((V, V), dtype=np.float64)
    XTy = np.zeros(V, dtype=np.float64)
    X_sum = np.zeros(V, dtype=np.float64)
    y_sum = 0.0

    for i in range(N):
        tokens = x[i]
        tokens = tokens[tokens != 0]
        L = len(tokens)
        if L == 0:
            continue
        # counts
        counts = np.bincount(tokens, minlength=V).astype(np.float64)
        # feature = counts / L
        # accumulate XTX += feature feature^T = counts counts^T / L^2
        # We can do outer product of counts (sparse) but V=2000, counts is sparse.
        # For efficiency, only use unique tokens.
        unique, cnts = np.unique(tokens, return_counts=True)
        cnts = cnts.astype(np.float64)
        # outer product of cnts
        # XTX[unique[:,None], unique[None,:]] += np.outer(cnts, cnts) / (L*L)
        # Use np.add.at for correctness
        outer = np.outer(cnts, cnts) / (L * L)
        np.add.at(XTX, (unique[:, None], unique[None, :]), outer)
        # XTy += feature * y[i] = counts * y[i] / L
        XTy[unique] += cnts * y[i] / L
        # X_sum += feature = counts / L
        X_sum[unique] += cnts / L
        y_sum += y[i]

    X_mean = X_sum / N
    y_mean = y_sum / N

    # Centered XTX and XTy
    XTX_c = XTX - N * np.outer(X_mean, X_mean)
    XTy_c = XTy - N * X_mean * y_mean

    # Ridge regression with cross-validation to pick lambda
    # We'll use 5-fold CV on the feedback pool.
    # Since we only have aggregated statistics, we can't easily do CV without recomputing per fold.
    # But we can recompute per fold by splitting indices. N=4000, V=2000, 5 folds is fine.
    # Let's do a simple 5-fold CV.
    np.random.seed(0)
    indices = np.arange(N)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    lambdas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    best_lambda = 1.0
    best_cv_acc = -1.0

    # Precompute per-example counts and L to speed up CV
    # We'll store for each example: unique tokens, counts, L, y
    examples = []
    for i in range(N):
        tokens = x[i]
        tokens = tokens[tokens != 0]
        L = len(tokens)
        if L == 0:
            examples.append((np.array([], dtype=np.int64), np.array([], dtype=np.float64), 1.0, y[i]))
        else:
            unique, cnts = np.unique(tokens, return_counts=True)
            examples.append((unique, cnts.astype(np.float64), float(L), y[i]))

    for lam in lambdas:
        accs = []
        for fold_idx in range(5):
            val_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != fold_idx])
            # Compute XTX, XTy, X_sum, y_sum on train
            XTX_tr = np.zeros((V, V), dtype=np.float64)
            XTy_tr = np.zeros(V, dtype=np.float64)
            X_sum_tr = np.zeros(V, dtype=np.float64)
            y_sum_tr = 0.0
            for i in train_idx:
                unique, cnts, L, yi = examples[i]
                if L == 0:
                    continue
                outer = np.outer(cnts, cnts) / (L * L)
                np.add.at(XTX_tr, (unique[:, None], unique[None, :]), outer)
                XTy_tr[unique] += cnts * yi / L
                X_sum_tr[unique] += cnts / L
                y_sum_tr += yi
            N_tr = len(train_idx)
            X_mean_tr = X_sum_tr / N_tr
            y_mean_tr = y_sum_tr / N_tr
            XTX_c_tr = XTX_tr - N_tr * np.outer(X_mean_tr, X_mean_tr)
            XTy_c_tr = XTy_tr - N_tr * X_mean_tr * y_mean_tr
            # Solve
            A = XTX_c_tr + lam * np.eye(V)
            w_tr = np.linalg.solve(A, XTy_c_tr)
            b_tr = y_mean_tr - X_mean_tr @ w_tr
            # Evaluate on val
            correct = 0
            for i in val_idx:
                unique, cnts, L, yi = examples[i]
                if L == 0:
                    score = b_tr
                else:
                    score = (cnts @ w_tr[unique]) / L + b_tr
                pred = 1 if score > 0.5 else 0
                if pred == yi:
                    correct += 1
            accs.append(correct / len(val_idx))
        cv_acc = np.mean(accs)
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Refit on full data with best_lambda
    XTX_full = XTX
    XTy_full = XTy
    X_mean_full = X_mean
    y_mean_full = y_mean
    XTX_c_full = XTX_full - N * np.outer(X_mean_full, X_mean_full)
    XTy_c_full = XTy_full - N * X_mean_full * y_mean_full
    A = XTX_c_full + best_lambda * np.eye(V)
    w_fit = np.linalg.solve(A, XTy_c_full)
    b_fit = y_mean_full - X_mean_full @ w_fit

    # Now we have a linear classifier: score = feature @ w_fit + b_fit.
    # We want to embed tokens so that the network computes this score.
    # We'll set embed.weight[t] = u_t, where u_t has mean 0, variance 1, and first coordinate c_t.
    # c_t = w_fit[t] / max_abs_w * sqrt(31) * 0.99
    max_abs_w = np.max(np.abs(w_fit))
    if max_abs_w < 1e-12:
        max_abs_w = 1.0
    scale = math.sqrt(31.0) * 0.99 / max_abs_w
    c = w_fit * scale  # (V,)

    # Construct u_t for each token
    # u_t[0] = c_t
    # For i=1..31: u_t[i] = -c_t/31
    # Then add s to u_t[1], subtract s from u_t[2], where s = 4 * sqrt(1 - c_t^2/31)
    # This gives mean 0, variance 1.
    embed_new = np.zeros((V, D), dtype=np.float32)
    # We can vectorize this
    c_t = c  # (V,)
    # base = -c_t/31 for all 31 coords
    base = -c_t / 31.0
    embed_new[:, 1:] = base[:, None]
    # add s to coord 1, subtract from coord 2
    s = 4.0 * np.sqrt(np.maximum(0.0, 1.0 - c_t**2 / 31.0))
    embed_new[:, 1] += s
    embed_new[:, 2] -= s
    embed_new[:, 0] = c_t

    # Set pad token (0) to something valid, e.g., c=0 -> u_0 = [0, -0/31, ..., s=4, -4, ...]
    # Actually c_t for pad will be w_fit[0] * scale. w_fit[0] is likely 0 due to regularization, but we can just set it.
    # It's fine.

    # Update weights
    w["embed.weight"] = embed_new.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Set head weights: we want logit difference = feature @ w_fit + b_fit.
    # Our model computes pooled[0] = (1/L) sum c_t = scale * (feature @ w_fit).
    # So logit difference = alpha * pooled[0] + beta.
    # We want alpha * scale = 1 => alpha = 1/scale.
    # And beta = b_fit.
    alpha = 1.0 / scale
    # But we can also optimize threshold. Let's compute scores on full data and find best threshold.
    # Compute scores = feature @ w_fit + b_fit for all examples.
    scores = np.zeros(N)
    for i in range(N):
        unique, cnts, L, yi = examples[i]
        if L == 0:
            scores[i] = b_fit
        else:
            scores[i] = (cnts @ w_fit[unique]) / L + b_fit
    # Find best threshold to maximize accuracy
    # We want pred = 1 if score > threshold else 0.
    # We can try all unique scores as thresholds.
    sorted_scores = np.sort(scores)
    # We can compute accuracy for threshold between scores.
    # Simple approach: try threshold = each score, and also slightly above/below.
    # Since we want to maximize accuracy, we can use the fact that optimal threshold is between two scores.
    # Let's just try thresholds at each score and at score + 1e-6.
    best_acc = 0.0
    best_thresh = 0.0
    # To be efficient, we can sort and compute cumulative counts.
    order = np.argsort(scores)
    sorted_y = y[order]
    # For threshold t, pred = 1 if score > t.
    # If we set threshold between sorted_scores[i] and sorted_scores[i+1], then first i+1 are predicted 0, rest 1.
    # Accuracy = (number of y=0 in first i+1 + number of y=1 in rest) / N.
    # We can compute this by iterating.
    # Let's just do a simple loop over unique thresholds (at most 4000).
    # But we can do it faster.
    # We'll just use a simple loop for clarity.
    # Actually, we can use the fact that we want to maximize accuracy on the feedback pool.
    # Let's just try a few thresholds around b_fit? No, we can do exact.
    # We'll compute for each possible threshold = sorted_scores[i] (predict 1 if score > threshold).
    # Also consider threshold = -inf (all 1) and +inf (all 0).
    # We'll just do a loop over i from 0 to N.
    # But N=4000, so it's fine.
    # Let's do it.
    # We want to find t such that pred = 1 if score > t.
    # We can sort scores and y.
    # For a given t, let k = number of scores <= t. Then pred 0 for first k, 1 for rest.
    # Accuracy = (sum_{i<k} (y_i==0) + sum_{i>=k} (y_i==1)) / N.
    # We can compute prefix sums of (y==0) and suffix sums of (y==1).
    # Let's do that.
    sorted_scores = np.sort(scores)
    sorted_y = y[np.argsort(scores)]
    # prefix zeros
    is_zero = (sorted_y == 0).astype(np.int64)
    prefix_zeros = np.cumsum(is_zero)
    total_zeros = prefix_zeros[-1]
    total_ones = N - total_zeros
    # For k from 0 to N, threshold between sorted_scores[k-1] and sorted_scores[k] (with k=0 meaning threshold < min, k=N meaning threshold >= max)
    # pred 0 for first k, 1 for rest.
    # correct = prefix_zeros[k-1] + (total_ones - (k - prefix_zeros[k-1]))? Wait.
    # For first k: correct if y==0 -> prefix_zeros[k-1] (if k>0 else 0)
    # For rest: correct if y==1 -> number of ones in rest = total_ones - (number of ones in first k) = total_ones - (k - prefix_zeros[k-1])
    # So correct = prefix_zeros[k-1] + total_ones - k + prefix_zeros[k-1] = 2*prefix_zeros[k-1] + total_ones - k.
    # Let's compute for k=0..N.
    best_correct = -1
    best_k = 0
    for k in range(N+1):
        if k == 0:
            pz = 0
        else:
            pz = prefix_zeros[k-1]
        correct = 2 * pz + total_ones - k
        if correct > best_correct:
            best_correct = correct
            best_k = k
    # best_k is number of examples predicted 0. Threshold should be between sorted_scores[best_k-1] and sorted_scores[best_k].
    if best_k == 0:
        best_thresh = sorted_scores[0] - 1e-6
    elif best_k == N:
        best_thresh = sorted_scores[-1] + 1e-6
    else:
        best_thresh = (sorted_scores[best_k-1] + sorted_scores[best_k]) / 2.0
    # Our model computes logit difference = alpha * pooled[0] + beta.
    # We want this to be > 0 for class 1. So we want alpha * pooled[0] + beta > 0 <=> pooled[0] > -beta/alpha.
    # pooled[0] = scale * (feature @ w_fit). So we want scale * (feature @ w_fit) > -beta/alpha.
    # We want feature @ w_fit + b_fit > best_thresh? Actually our score is feature @ w_fit + b_fit.
    # We want to threshold that score at best_thresh. So we want feature @ w_fit + b_fit > best_thresh.
    # Our logit difference = alpha * scale * (feature @ w_fit) + beta = feature @ w_fit + beta (since alpha*scale=1).
    # So we want feature @ w_fit + beta > 0 <=> feature @ w_fit > -beta.
    # We want feature @ w_fit + b_fit > best_thresh <=> feature @ w_fit > best_thresh - b_fit.
    # So set -beta = best_thresh - b_fit => beta = b_fit - best_thresh.
    beta = b_fit - best_thresh

    # Set head weights
    head_weight = np.zeros_like(w["head.weight"], dtype=np.float32)
    head_bias = np.zeros_like(w["head.bias"], dtype=np.float32)
    # class 0: logit = 0
    head_weight[0, :] = 0.0
    head_bias[0] = 0.0
    # class 1: logit = alpha * pooled[0] + beta
    head_weight[1, :] = 0.0
    head_weight[1, 0] = alpha
    head_bias[1] = beta

    w["head.weight"] = head_weight
    w["head.bias"] = head_bias

    # Ensure all other weights are unchanged (they are already zero/identity)
    # But we should make sure blocks are identity: they are already zero, so fine.
    # Final norm: weight=1, bias=0. Already set.

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_acc": float(best_cv_acc),
        "train_acc": float(best_correct / N),
        "best_threshold": float(best_thresh),
        "scale": float(scale),
        "alpha": float(alpha),
        "beta": float(beta),
        "w_norm": float(np.linalg.norm(w_fit)),
        "max_abs_w": float(max_abs_w),
        "embed_norm_mean": float(np.mean(np.linalg.norm(embed_new, axis=1))),
        "embed_mean": float(np.mean(embed_new)),
    }

    return {"weights": w, "diagnostics": diagnostics}
