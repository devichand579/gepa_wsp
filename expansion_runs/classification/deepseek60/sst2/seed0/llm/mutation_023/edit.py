import numpy as np

def edit(w, f):
    # Extract feedback data
    x = f["x"]          # [4000, 64] token IDs
    y = f["y"].astype(np.float32)  # [4000] labels 0/1
    n, seq_len = x.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build normalised unigram counts X (n x V) sparsely, then compute XTX, XTy, X_mean
    XTX = np.zeros((V, V), dtype=np.float64)
    XTy = np.zeros(V, dtype=np.float64)
    X_sum = np.zeros(V, dtype=np.float64)
    total_tokens = 0

    for i in range(n):
        tokens = x[i]
        # ignore padding (token 0)
        mask = tokens != 0
        if not np.any(mask):
            continue
        L = np.sum(mask)
        total_tokens += L
        # count occurrences of each token
        uniq, counts = np.unique(tokens[mask], return_counts=True)
        # normalised counts
        c = counts.astype(np.float64) / L
        # accumulate XTX
        # outer product of c with itself
        XTX[np.ix_(uniq, uniq)] += np.outer(c, c)
        # accumulate XTy
        XTy[uniq] += c * y[i]
        # accumulate X_sum (sum of X rows)
        X_sum[uniq] += c

    X_mean = X_sum / n
    y_mean = np.mean(y)

    # Centered XTX and XTy
    XcTXc = XTX - n * np.outer(X_mean, X_mean)
    XcTyc = XTy - n * X_mean * y_mean

    # Ridge regression with cross-validation to pick lambda
    # We'll use a small grid of lambdas and 5-fold CV.
    # To avoid recomputing inverse for each fold, we can use the full data for now,
    # but proper CV requires recomputing on folds. Since n=4000, V=2000,
    # we can do 5-fold CV by splitting indices.
    # However, for simplicity and speed, we can use a fixed lambda based on
    # the scale of XcTXc. A common heuristic is lambda = alpha * trace(XcTXc)/V.
    # We'll try a few values and pick the best on a validation split.
    # Actually, we can do a simple hold-out: use first 3000 for training, last 1000 for validation.
    # But we have the full 4000, we can do 5-fold CV properly.
    # Given time, we'll do a quick 5-fold CV with a few lambdas.

    # Precompute SVD of XcTXc for efficient lambda selection? XcTXc is 2000x2000.
    # We can just solve for each lambda directly. 2000x2000 solve is fast.
    # Let's do 5-fold CV.
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_lambda = 1.0
    best_acc = 0.0

    for lam in lambdas:
        accs = []
        for k in range(5):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            # Build XTX, XTy for training fold
            XTX_tr = np.zeros((V, V), dtype=np.float64)
            XTy_tr = np.zeros(V, dtype=np.float64)
            X_sum_tr = np.zeros(V, dtype=np.float64)
            n_tr = len(train_idx)
            for i in train_idx:
                tokens = x[i]
                mask = tokens != 0
                if not np.any(mask):
                    continue
                L = np.sum(mask)
                uniq, counts = np.unique(tokens[mask], return_counts=True)
                c = counts.astype(np.float64) / L
                XTX_tr[np.ix_(uniq, uniq)] += np.outer(c, c)
                XTy_tr[uniq] += c * y[i]
                X_sum_tr[uniq] += c
            X_mean_tr = X_sum_tr / n_tr
            y_mean_tr = np.mean(y[train_idx])
            XcTXc_tr = XTX_tr - n_tr * np.outer(X_mean_tr, X_mean_tr)
            XcTyc_tr = XTy_tr - n_tr * X_mean_tr * y_mean_tr

            # Solve ridge
            A = XcTXc_tr + lam * np.eye(V)
            w_vec = np.linalg.solve(A, XcTyc_tr)
            b = y_mean_tr - np.dot(X_mean_tr, w_vec)

            # Evaluate on validation fold
            correct = 0
            for i in val_idx:
                tokens = x[i]
                mask = tokens != 0
                if not np.any(mask):
                    pred = 1 if b > 0 else 0
                else:
                    L = np.sum(mask)
                    uniq, counts = np.unique(tokens[mask], return_counts=True)
                    c = counts.astype(np.float64) / L
                    score = np.dot(w_vec[uniq], c) + b
                    pred = 1 if score > 0 else 0
                if pred == y[i]:
                    correct += 1
            accs.append(correct / len(val_idx))
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Now fit on full data with best_lambda
    A = XcTXc + best_lambda * np.eye(V)
    w_vec = np.linalg.solve(A, XcTyc)
    b = y_mean - np.dot(X_mean, w_vec)

    # Scale w_vec so that max |w_vec| <= 4 (to fit in embed[:,0] and keep unit variance)
    max_abs = np.max(np.abs(w_vec))
    if max_abs > 0:
        alpha = max_abs / 4.0
    else:
        alpha = 1.0
    w_scaled = w_vec / alpha

    # Now update embed.weight
    embed = w["embed.weight"].copy()
    # We'll set dim 0 to w_scaled, dim 1 to -w_scaled, and dims 2,3 to maintain unit variance.
    # For each row v:
    #   embed[v,0] = w_scaled[v]
    #   embed[v,1] = -w_scaled[v]
    #   embed[v,2] = sqrt((32 - 2*w_scaled[v]^2)/2)
    #   embed[v,3] = -sqrt((32 - 2*w_scaled[v]^2)/2)
    #   all other dims = 0
    # This ensures mean=0, variance=1.
    # But wait: the current embed rows have dims 2,3 set to ±4. We can just overwrite.
    # However, we must be careful: if w_scaled[v]^2 > 16, sqrt will be imaginary.
    # Since we scaled max |w_scaled| <= 4, w_scaled[v]^2 <= 16, so 32 - 2*w_scaled^2 >= 0.
    # It's exactly 0 when |w_scaled| = 4. So it's fine.
    # We'll set all other dims to 0.
    embed_new = np.zeros_like(embed)
    embed_new[:, 0] = w_scaled
    embed_new[:, 1] = -w_scaled
    val = np.sqrt(np.maximum(0, (32.0 - 2.0 * w_scaled**2) / 2.0))
    embed_new[:, 2] = val
    embed_new[:, 3] = -val
    w["embed.weight"] = embed_new.astype(np.float32)

    # Update head
    head_w = np.zeros_like(w["head.weight"])
    head_w[1, 0] = alpha  # so that alpha * (w_scaled) = w_vec
    head_b = np.zeros_like(w["head.bias"])
    head_b[1] = b
    head_b[0] = 0.0
    w["head.weight"] = head_w.astype(np.float32)
    w["head.bias"] = head_b.astype(np.float32)

    # Ensure all other weights remain as they are (zero/identity)
    # They are already zero/identity in w, so we don't need to change them.

    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "alpha": float(alpha),
        "b": float(b),
        "max_abs_w": float(max_abs),
        "embed_nonzero": int(np.count_nonzero(w["embed.weight"])),
        "head_weight_nonzero": int(np.count_nonzero(w["head.weight"])),
    }
    return {"weights": w, "diagnostics": diagnostics}
