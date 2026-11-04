import numpy as np
import math

def edit(w, f):
    # Extract training data
    X = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    N, L = X.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build bag-of-words matrix (N x V)
    BOW = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        ids = X[i]
        # count non-pad tokens (0 is pad)
        valid = ids[ids != 0]
        if len(valid) > 0:
            np.add.at(BOW[i], valid, 1.0)

    # Normalize by sequence length (number of non-pad tokens)
    lengths = np.sum(X != 0, axis=1).astype(np.float32)
    lengths[lengths == 0] = 1.0
    BOW_norm = BOW / lengths[:, None]   # [N, V]

    # Labels: 0 -> -1, 1 -> +1
    y_signed = np.where(y == 1, 1.0, -1.0).astype(np.float32)

    # 5-fold cross-validation to select ridge lambda
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    folds = np.array_split(indices, 5)

    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_lambda = 1.0
    best_cv_acc = -1.0

    for lam in lambdas:
        cv_acc = 0.0
        for k in range(5):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != k])
            X_tr = BOW_norm[train_idx]
            y_tr = y_signed[train_idx]
            X_val = BOW_norm[val_idx]
            y_val = y_signed[val_idx]

            # Ridge regression: solve (X^T X + lam I) w = X^T y
            # Use normal equations with small regularization
            A = X_tr.T @ X_tr + lam * np.eye(V, dtype=np.float32)
            b = X_tr.T @ y_tr
            # Solve using Cholesky or lstsq
            try:
                s_fold = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                s_fold = np.linalg.lstsq(A, b, rcond=None)[0]

            # Intercept: mean of y_tr - mean of X_tr @ s_fold
            c_fold = np.mean(y_tr) - np.mean(X_tr @ s_fold)

            # Predict on validation
            pred_val = X_val @ s_fold + c_fold
            pred_label = (pred_val > 0).astype(np.int64)
            true_label = (y_val > 0).astype(np.int64)
            cv_acc += np.mean(pred_label == true_label)

        cv_acc /= 5.0
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam

    # Fit final model on all data with best lambda
    lam = best_lambda
    A = BOW_norm.T @ BOW_norm + lam * np.eye(V, dtype=np.float32)
    b = BOW_norm.T @ y_signed
    try:
        s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(A, b, rcond=None)[0]

    c = np.mean(y_signed) - np.mean(BOW_norm @ s)

    # Now embed s into column 0 of embed.weight, and make each row zero-mean/unit-variance.
    # We need to construct a 2000x32 matrix E such that:
    #   E[:, 0] = s
    #   For each row v: mean(E[v, :]) = 0, std(E[v, :]) = 1
    # We can fill the remaining 31 columns with a deterministic pattern that satisfies the constraints.
    # Let s_v = s[v]. We need to choose 31 values x_1..x_31 such that:
    #   s_v + sum x_i = 0  => sum x_i = -s_v
    #   (s_v^2 + sum x_i^2) / 32 = 1 => sum x_i^2 = 32 - s_v^2
    # This is possible if 32 - s_v^2 >= 0. Since |s_v| is typically < 5, it's fine.
    # We can choose x_i = a for i=1..30 and x_31 = b.
    # Then 30a + b = -s_v
    # 30a^2 + b^2 = 32 - s_v^2
    # Solve for a, b. This gives two solutions. We'll pick the one with smaller absolute values.
    E = np.zeros((V, 32), dtype=np.float32)
    E[:, 0] = s.astype(np.float32)

    for v in range(V):
        s_v = float(s[v])
        # Solve for a, b
        # 30a + b = -s_v
        # 30a^2 + b^2 = 32 - s_v^2
        # Substitute b = -s_v - 30a
        # 30a^2 + (-s_v - 30a)^2 = 32 - s_v^2
        # 30a^2 + s_v^2 + 60 s_v a + 900 a^2 = 32 - s_v^2
        # 930 a^2 + 60 s_v a + 2 s_v^2 - 32 = 0
        A_coef = 930.0
        B_coef = 60.0 * s_v
        C_coef = 2.0 * s_v * s_v - 32.0
        disc = B_coef * B_coef - 4.0 * A_coef * C_coef
        if disc < 0:
            # Fallback: just use a simple pattern (should not happen for reasonable s_v)
            a = 0.0
            b = -s_v
        else:
            sqrt_disc = math.sqrt(disc)
            a1 = (-B_coef + sqrt_disc) / (2.0 * A_coef)
            a2 = (-B_coef - sqrt_disc) / (2.0 * A_coef)
            b1 = -s_v - 30.0 * a1
            b2 = -s_v - 30.0 * a2
            # Choose the solution with smaller max(abs(a), abs(b))
            if max(abs(a1), abs(b1)) <= max(abs(a2), abs(b2)):
                a, b = a1, b1
            else:
                a, b = a2, b2
        E[v, 1:31] = a
        E[v, 31] = b

    # Ensure exact zero mean and unit variance (numerical cleanup)
    E = E - E.mean(axis=1, keepdims=True)
    E = E / E.std(axis=1, keepdims=True)

    # Overwrite embed.weight
    w["embed.weight"] = E.astype(np.float32)

    # Zero out pos and all block weights to ensure identity path
    w["pos.weight"][:] = 0.0
    for block in ["block0", "block1"]:
        for name in ["norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                     "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{block}.{name}"
            if key in w:
                w[key][:] = 0.0
        # Set norm weights to 1, biases to 0
        w[f"{block}.norm1.weight"][:] = 1.0
        w[f"{block}.norm1.bias"][:] = 0.0
        w[f"{block}.norm2.weight"][:] = 1.0
        w[f"{block}.norm2.bias"][:] = 0.0

    # Final norm: set weight=1, bias=0
    w["norm.weight"][:] = 1.0
    w["norm.bias"][:] = 0.0

    # Head: read column 0 with intercept c
    # head.weight shape [2, 32]
    w["head.weight"][:] = 0.0
    w["head.weight"][0, 0] = -1.0
    w["head.weight"][1, 0] = 1.0
    w["head.bias"][0] = -c
    w["head.bias"][1] = c

    # Diagnostics
    diagnostics = {
        "method": "BOW ridge + LayerNorm-identity embedding",
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_cv_acc),
        "s_norm": float(np.linalg.norm(s)),
        "intercept": float(c),
        "embed_row_mean_maxabs": float(np.max(np.abs(E.mean(axis=1)))),
        "embed_row_std_maxdev": float(np.max(np.abs(E.std(axis=1) - 1.0))),
    }

    return {"weights": w, "diagnostics": diagnostics}
