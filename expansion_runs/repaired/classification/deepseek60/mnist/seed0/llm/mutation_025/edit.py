import numpy as np

def edit(w, f):
    # Copy weights so we do not mutate the input dict in place.
    W1 = np.array(w["fc1.weight"], dtype=np.float64, copy=True)
    b1 = np.array(w["fc1.bias"], dtype=np.float64, copy=True)
    W2 = np.array(w["fc2.weight"], dtype=np.float64, copy=True)
    b2 = np.array(w["fc2.bias"], dtype=np.float64, copy=True)

    X = np.asarray(f["x"], dtype=np.float64)          # [2000,1,28,28]
    y = np.asarray(f["y"], dtype=np.int64)            # [2000]
    H_old = np.asarray(f["features"], dtype=np.float64)  # [2000,128] ReLU(fc1(X))
    N, D = X.shape[0], X.shape[2] * X.shape[3]
    Xf = X.reshape(N, D)

    # Identify dead hidden units: columns that are zero for almost all examples.
    zero_frac = np.mean(H_old <= 1e-12, axis=0)
    dead = np.where(zero_frac > 0.999)[0]
    if dead.size == 0:
        # Nothing dead: just refit fc2 by ridge least-squares.
        H = H_old
    else:
        # Build closed-form LDA directions on raw pixels for the dead units.
        # Use class means and a ridge-regularized pooled within-class covariance.
        classes = np.unique(y)
        C = classes.size
        mu_all = Xf.mean(axis=0)
        mu_c = np.zeros((C, D), dtype=np.float64)
        for ci, c in enumerate(classes):
            mu_c[ci] = Xf[y == c].mean(axis=0)

        # Pooled within-class covariance.
        Sw = np.zeros((D, D), dtype=np.float64)
        for ci, c in enumerate(classes):
            Xc = Xf[y == c]
            Xc = Xc - mu_c[ci]
            Sw += Xc.T @ Xc
        Sw /= max(1, N - C)

        # Ridge regularize and invert via Cholesky.
        lam = 1e-3 * np.trace(Sw) / D
        Sw_reg = Sw + lam * np.eye(D, dtype=np.float64)
        L = np.linalg.cholesky(Sw_reg)
        # Solve Sw_reg * V = (mu_c - mu_all).T
        B = (mu_c - mu_all).T  # [D, C]
        V = np.linalg.solve(Sw_reg, B)  # [D, C]

        # LDA directions are columns of V.  We need len(dead) directions.
        # Take the first len(dead) directions, or pad by repeating if needed.
        k = dead.size
        if V.shape[1] >= k:
            dirs = V[:, :k].T  # [k, D]
        else:
            reps = int(np.ceil(k / V.shape[1]))
            dirs = np.tile(V.T, (reps, 1))[:k]

        # Normalize directions so their input-space norm is reasonable.
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

        # Set dead-unit weights and biases.
        # Bias is chosen so that roughly half of the training examples activate.
        for j, idx in enumerate(dead):
            proj = Xf @ dirs[j]
            # Use median as threshold, so about 50% activation.
            thr = np.median(proj)
            W1[idx] = dirs[j]
            b1[idx] = -thr

        # Recompute hidden features for all feedback examples.
        Z = Xf @ W1.T + b1[None, :]
        H = np.maximum(Z, 0.0)

    # Refit fc2 by ridge least-squares on the new hidden features.
    # Standardize features to make the ridge penalty scale-invariant.
    H_mean = H.mean(axis=0)
    H_std = H.std(axis=0)
    H_std[H_std < 1e-6] = 1.0
    Hs = (H - H_mean) / H_std

    # One-hot targets.
    C = 10
    Y = np.zeros((N, C), dtype=np.float64)
    Y[np.arange(N), y] = 1.0

    # Ridge least-squares: W = (Hs^T Hs + lambda I)^-1 Hs^T Y
    # Choose lambda from a simple scale heuristic.
    lam2 = 1e-2 * np.trace(Hs.T @ Hs) / Hs.shape[1]
    A = Hs.T @ Hs + lam2 * np.eye(Hs.shape[1], dtype=np.float64)
    Bmat = Hs.T @ Y
    W2s = np.linalg.solve(A, Bmat).T  # [10,128]
    b2s = Y.mean(axis=0) - W2s @ Hs.mean(axis=0)

    # Convert back from standardized features to original hidden features.
    W2 = W2s / H_std[None, :]
    b2 = b2s - W2 @ H_mean

    # Optional mild calibration: scale logits so their standard deviation is not extreme.
    # Compute logits on the full feedback pool and scale to a target std.
    logits = H @ W2.T + b2[None, :]
    cur_std = logits.std()
    if cur_std > 1e-6:
        target_std = 8.0
        scale = target_std / cur_std
        W2 *= scale
        b2 *= scale

    # Write back as float32, clip for safety.
    w["fc1.weight"] = np.clip(W1, -1000.0, 1000.0).astype(np.float32)
    w["fc1.bias"] = np.clip(b1, -1000.0, 1000.0).astype(np.float32)
    w["fc2.weight"] = np.clip(W2, -1000.0, 1000.0).astype(np.float32)
    w["fc2.bias"] = np.clip(b2, -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "dead_units_replaced": int(dead.size),
        "dead_indices": dead.tolist()[:20],
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias_norm": float(np.linalg.norm(w["fc2.bias"])),
        "hidden_zero_fraction_new": float(np.mean(H <= 1e-12)),
        "logit_std_before_scale": float(cur_std),
    }
    return {"weights": w, "diagnostics": diagnostics}
