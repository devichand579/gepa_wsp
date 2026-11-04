import numpy as np

def edit(w, f):
    # Extract training data
    X = f["x"]          # [4000, 64] int64
    y = f["y"]          # [4000] int64
    n, L = X.shape
    V = w["embed.weight"].shape[0]  # 2000

    # Build bag-of-words matrix BOW [n, V]
    BOW = np.zeros((n, V), dtype=np.float32)
    for i in range(n):
        ids = X[i]
        # exclude padding token 0
        valid = ids[ids != 0]
        if valid.size > 0:
            np.add.at(BOW[i], valid, 1.0)

    # Normalize by sequence length (number of non-padding tokens)
    lengths = np.maximum((X != 0).sum(axis=1), 1).astype(np.float32)
    BOW_norm = BOW / lengths[:, None]   # [n, V]

    # Labels: 0 -> -1, 1 -> +1
    y_signed = np.where(y == 1, 1.0, -1.0).astype(np.float32)

    # Ridge regression with intercept, selected by 5-fold CV
    # We solve: min ||BOW_norm @ s + c - y_signed||^2 + lambda * ||s||^2
    # Closed form: augment with a column of ones for intercept, but don't penalize intercept.
    # We'll use a small grid of lambdas.
    lambdas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_lambda = None
    best_cv_acc = -1.0
    best_s = None
    best_c = None

    # 5-fold cross-validation
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    # Deterministic shuffle for reproducibility
    rng = np.random.RandomState(42)
    rng.shuffle(indices)

    for lam in lambdas:
        cv_acc = 0.0
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
            X_tr = BOW_norm[train_idx]
            y_tr = y_signed[train_idx]
            X_val = BOW_norm[val_idx]
            y_val = y_signed[val_idx]

            # Add intercept column
            X_tr_aug = np.hstack([X_tr, np.ones((X_tr.shape[0], 1), dtype=np.float32)])
            X_val_aug = np.hstack([X_val, np.ones((X_val.shape[0], 1), dtype=np.float32)])

            # Ridge closed form: (X^T X + lambda I) w = X^T y
            # Do not penalize intercept: set lambda=0 for last column
            D = np.eye(X_tr_aug.shape[1], dtype=np.float32) * lam
            D[-1, -1] = 0.0
            A = X_tr_aug.T @ X_tr_aug + D
            b = X_tr_aug.T @ y_tr
            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
            s = sol[:-1]
            c = sol[-1]

            # Predict on validation
            pred = np.sign(X_val @ s + c)
            pred[pred == 0] = 1.0
            acc = np.mean(pred == y_val)
            cv_acc += acc
        cv_acc /= n_folds
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lambda = lam
            # Refit on full data with best lambda
            X_aug = np.hstack([BOW_norm, np.ones((n, 1), dtype=np.float32)])
            D = np.eye(X_aug.shape[1], dtype=np.float32) * best_lambda
            D[-1, -1] = 0.0
            A = X_aug.T @ X_aug + D
            b = X_aug.T @ y_signed
            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(A, b, rcond=None)[0]
            best_s = sol[:-1].astype(np.float32)
            best_c = float(sol[-1])

    # Now update weights
    # 1. Set embed.weight column 0 to best_s, and make each row zero-mean/unit-variance.
    #    We'll set column 0 = best_s, and fill remaining columns with small values to satisfy mean=0, var=1.
    embed = w["embed.weight"].copy()  # [2000, 32]
    # We'll construct new embed rows.
    new_embed = np.zeros_like(embed)
    for v in range(V):
        row = np.zeros(32, dtype=np.float32)
        row[0] = best_s[v]
        # We need mean 0 and variance 1 over the 32 dims.
        # Let the other 31 entries be a constant c such that:
        # sum = best_s[v] + 31*c = 0  => c = -best_s[v]/31
        # variance = (best_s[v]^2 + 31*c^2)/32 = 1
        # Solve for best_s[v] given c = -best_s[v]/31:
        # variance = (best_s[v]^2 + 31*(best_s[v]^2/31^2))/32 = best_s[v]^2*(1 + 1/31)/32 = best_s[v]^2*(32/31)/32 = best_s[v]^2/31
        # So we need best_s[v]^2/31 = 1 => best_s[v] = ±sqrt(31).
        # But we want best_s[v] to be the fitted weight, which is arbitrary.
        # So we cannot have both arbitrary best_s[v] and zero-mean/unit-variance with only one free parameter.
        # We need to adjust the other columns to satisfy mean=0 and variance=1 while keeping column 0 = best_s[v].
        # We have 31 other entries. We can set them all equal to c, then:
        # mean: best_s[v] + 31*c = 0 => c = -best_s[v]/31
        # variance: (best_s[v]^2 + 31*c^2)/32 = 1 => best_s[v]^2 + 31*(best_s[v]^2/31^2) = 32
        # => best_s[v]^2 (1 + 1/31) = 32 => best_s[v]^2 * (32/31) = 32 => best_s[v]^2 = 31
        # So best_s[v] must be ±sqrt(31). That's not arbitrary.
        # Therefore, we cannot have arbitrary column 0 and zero-mean/unit-variance with all other columns equal.
        # We need to use more degrees of freedom. We can set the other 31 columns to have mean 0 and variance 1 while column 0 is best_s[v].
        # Actually, we can just set column 0 = best_s[v], and then set the other 31 columns to be a zero-mean unit-variance vector that is orthogonal to column 0? Not necessary.
        # The requirement for LayerNorm to be identity is that the vector has mean 0 and variance 1.
        # We can achieve this by setting the other 31 entries to be a scaled version of a fixed zero-mean unit-variance vector, and then adjust the scale to make the overall variance 1.
        # Let's do this: let u be a fixed vector of length 31 with mean 0 and variance 1. For example, u = [1, -1, 0, 0, ...] normalized.
        # We want row = [best_s[v], alpha * u] such that mean(row) = 0 and var(row) = 1.
        # mean(row) = (best_s[v] + alpha * sum(u)) / 32 = best_s[v]/32 (since sum(u)=0).
        # So mean is not zero unless best_s[v]=0. That's a problem.
        # To have mean 0, we need the sum of all 32 entries to be 0. So if column 0 is best_s[v], the sum of the other 31 must be -best_s[v].
        # So we can set the other 31 entries to be a vector with sum = -best_s[v] and variance such that total variance is 1.
        # Let the other 31 entries be c + d * u, where u has mean 0 and variance 1 over 31 elements.
        # Then sum of other 31 = 31*c. So 31*c = -best_s[v] => c = -best_s[v]/31.
        # The variance of the other 31 is d^2. The total sum of squares = best_s[v]^2 + 31*(c^2 + d^2) = best_s[v]^2 + 31*c^2 + 31*d^2.
        # We want total variance = (sum of squares)/32 = 1 => sum of squares = 32.
        # So best_s[v]^2 + 31*(best_s[v]^2/31^2) + 31*d^2 = 32
        # => best_s[v]^2 (1 + 1/31) + 31*d^2 = 32
        # => best_s[v]^2 * (32/31) + 31*d^2 = 32
        # => 31*d^2 = 32 - best_s[v]^2 * (32/31)
        # => d^2 = (32/31) - best_s[v]^2 / 31 = (32 - best_s[v]^2)/31
        # So we need 32 - best_s[v]^2 >= 0 => |best_s[v]| <= sqrt(32) ~ 5.65.
        # But our fitted weights best_s can be larger than 5.65. In fact, the norm of best_s is around 60, so individual components can be >5.65.
        # If |best_s[v]| > sqrt(32), we cannot make the row zero-mean and unit-variance while keeping column 0 exactly best_s[v].
        # We could scale down best_s[v] and compensate by scaling up the head weights. That's fine! We can scale the entire weight vector s by a constant factor, and scale the head weights by the inverse factor. The product remains the same.
        # So we can scale s such that max |s[v]| <= sqrt(32) ~ 5.65. Then we can construct the rows.
        # Let's do that: find scale factor gamma such that gamma * max|s| <= 5.0 (safe). Then set embed column 0 = gamma * s, and head weights = (1/gamma) * original head weights.
        # The head currently reads column 0 with weight -1 and 1. We can set head.weight[0,0] = -1/gamma, head.weight[1,0] = 1/gamma, and head.bias[0] = -c/gamma? Wait, the intercept c is added after the linear combination. The logit difference is 2 * (gamma * s) @ BOW_norm + 2*c? Actually, original logit difference = 2 * (s @ BOW_norm + c). If we scale s by gamma, we need to scale the head weights by 1/gamma to keep the same logit. But the intercept c is independent. We can just set head.bias to c directly? Let's derive.
        # Current head: logit0 = -pooled[0] + b0, logit1 = pooled[0] + b1.
        # pooled[0] = (1/n) * BOW @ embed_col0 = BOW_norm @ embed_col0.
        # If embed_col0 = gamma * s, then pooled[0] = gamma * (BOW_norm @ s).
        # We want logit1 - logit0 = 2 * (BOW_norm @ s + c) = 2 * BOW_norm @ s + 2c.
        # With head weights: logit1 - logit0 = 2 * pooled[0] + (b1 - b0) = 2 * gamma * (BOW_norm @ s) + (b1 - b0).
        # So we need 2 * gamma = 2 => gamma = 1? No, we can set head.weight[1,0] = 1/gamma, head.weight[0,0] = -1/gamma, then logit1 - logit0 = (1/gamma)*pooled[0] - (-1/gamma)*pooled[0] = (2/gamma)*pooled[0] = (2/gamma)*gamma*(BOW_norm @ s) = 2*(BOW_norm @ s). Perfect.
        # And we set b1 - b0 = 2c. We can set b1 = c, b0 = -c. Or just b0 = -c, b1 = c.
        # So we can scale s arbitrarily and adjust head weights accordingly.
        # Let's choose gamma such that max |gamma * s| <= 5.0.
        max_abs_s = np.max(np.abs(best_s))
        if max_abs_s > 0:
            gamma = 5.0 / max_abs_s
        else:
            gamma = 1.0
        s_scaled = best_s * gamma

        # Now construct embed rows
        # We'll use a fixed pattern for the other 31 entries: we need sum = -s_scaled[v], and variance such that total variance = 1.
        # Let the other 31 entries be c + d * u, where u is a fixed vector of length 31 with mean 0 and variance 1.
        # We can choose u = [1, -1, 0, 0, ...] normalized? Let's just use a simple pattern: u = np.zeros(31); u[0]=1; u[1]=-1; then normalize to variance 1.
        # Actually, we can just set the other 31 entries to be a constant c plus a small perturbation. But we need variance exactly 1.
        # Let's compute c = -s_scaled[v] / 31.
        # We need d^2 = (32 - s_scaled[v]^2) / 31. Since |s_scaled[v]| <= 5, s_scaled[v]^2 <= 25, so 32 - s_scaled[v]^2 >= 7, so d^2 >= 7/31 > 0. Good.
        # We can set u = np.zeros(31); u[0] = 1; u[1] = -1; then u has mean 0, variance = (1+1)/31 = 2/31. We want variance 1, so scale u by sqrt(31/2).
        # Let's just use a simple u: u = np.ones(31) * 0? No, we need variance. Let's use u = np.arange(31) - 15.0; then subtract mean, divide by std.
        # Simpler: just set the other 31 entries to c + d * v, where v is a fixed vector with mean 0 and variance 1. We can use v = np.random.randn(31); v -= v.mean(); v /= v.std(). But we need deterministic. We can just use a fixed pattern.
        # Let's use v = np.zeros(31); v[0] = 1.0; v[1] = -1.0; v[2:] = 0.0; then v has mean 0, variance = 2/31. To make variance 1, multiply by sqrt(31/2).
        # Actually, we don't need to be perfect. The requirement is that LayerNorm is identity. LayerNorm computes (x - mean)/sqrt(var + eps). If we make mean=0 and var=1, then LayerNorm(x) = x. We can achieve this by setting the row to have mean 0 and var 1.
        # Let's just do it numerically for each row:
        # We have row[0] = s_scaled[v]. We need to choose row[1:] such that sum(row) = 0 and sum(row^2) = 32.
        # This is a system of two equations with 31 unknowns. We can set all row[1:] equal to a constant c, then:
        # sum = s_scaled[v] + 31*c = 0 => c = -s_scaled[v]/31.
        # sum of squares = s_scaled[v]^2 + 31*c^2 = s_scaled[v]^2 + 31*(s_scaled[v]^2/31^2) = s_scaled[v]^2 * (1 + 1/31) = s_scaled[v]^2 * (32/31).
        # We want this to equal 32. So s_scaled[v]^2 * (32/31) = 32 => s_scaled[v]^2 = 31 => |s_scaled[v]| = sqrt(31) ~ 5.568.
        # So if we set all other entries equal, we force |s_scaled[v]| = sqrt(31). That's too restrictive.
        # So we need to use the other 31 entries to absorb the variance. We can set them to c + d * v, where v is a vector with sum 0 and sum of squares 31.
        # Let v be a vector of length 31 with sum 0 and sum of squares 31. For example, v = [1, -1, 0, ..., 0] scaled? Sum of squares of [1,-1,0...] is 2. To get 31, scale by sqrt(31/2).
        # Then we set row[1:] = c * ones + d * v. We need sum(row[1:]) = -s_scaled[v] => 31*c = -s_scaled[v] => c = -s_scaled[v]/31.
        # Sum of squares of row[1:] = 31*c^2 + d^2 * 31 (since v has sum of squares 31 and sum 0, the cross term is 0).
        # Total sum of squares = s_scaled[v]^2 + 31*c^2 + 31*d^2 = s_scaled[v]^2 + 31*(s_scaled[v]^2/31^2) + 31*d^2 = s_scaled[v]^2 * (32/31) + 31*d^2.
        # We want this to equal 32. So 31*d^2 = 32 - s_scaled[v]^2 * (32/31) = (32/31)*(31 - s_scaled[v]^2).
        # So d^2 = (32/31^2)*(31 - s_scaled[v]^2) = (32/961)*(31 - s_scaled[v]^2).
        # Since |s_scaled[v]| <= 5, s_scaled[v]^2 <= 25, so 31 - s_scaled[v]^2 >= 6, so d^2 > 0. Good.
        # So we can construct the row exactly.

        # Let's define v vector of length 31: we can use v = np.zeros(31); v[0] = 1; v[1] = -1; then v has sum 0, sum of squares 2. We need sum of squares 31, so scale by sqrt(31/2).
        # Actually, we can just use a random v and orthogonalize, but deterministic is fine.
        # Let's just use v = np.arange(31) - 15.0; v -= v.mean(); v /= v.std(); then v has mean 0, variance 1, so sum of squares = 31. Perfect.
        v_base = np.arange(31, dtype=np.float32) - 15.0
        v_base -= v_base.mean()
        v_base /= v_base.std()  # now sum=0, sum of squares=31

        new_embed = np.zeros((V, 32), dtype=np.float32)
        for v in range(V):
            s_val = s_scaled[v]
            c = -s_val / 31.0
            d2 = (32.0 / 961.0) * (31.0 - s_val * s_val)
            if d2 < 0:
                d2 = 0.0
            d = np.sqrt(d2)
            row = np.empty(32, dtype=np.float32)
            row[0] = s_val
            row[1:] = c + d * v_base
            new_embed[v] = row

        # Verify mean and variance
        # mean = (s_val + 31*c)/32 = (s_val - s_val)/32 = 0
        # var = (s_val^2 + 31*c^2 + 31*d^2)/32 = (s_val^2 + s_val^2/31 + 31*d^2)/32 = (s_val^2*(32/31) + 31*d^2)/32 = (s_val^2*(32/31) + (32/31)*(31 - s_val^2))/32 = (32/31)*(s_val^2 + 31 - s_val^2)/32 = (32/31)*31/32 = 1. Perfect.

        w["embed.weight"] = new_embed.astype(np.float32)

        # Set pos to zero
        w["pos.weight"][:] = 0.0

        # Set all block weights to zero (they already are, but ensure)
        for key in w:
            if key.startswith("block"):
                w[key][:] = 0.0

        # Set final norm to identity (weight=1, bias=0)
        w["norm.weight"][:] = 1.0
        w["norm.bias"][:] = 0.0

        # Set head to read column 0 with appropriate scaling and intercept
        head_w = np.zeros((2, 32), dtype=np.float32)
        head_w[0, 0] = -1.0 / gamma
        head_w[1, 0] = 1.0 / gamma
        w["head.weight"] = head_w

        head_b = np.zeros(2, dtype=np.float32)
        head_b[0] = -best_c
        head_b[1] = best_c
        w["head.bias"] = head_b

        # Diagnostics
        diagnostics = {
            "best_lambda": best_lambda,
            "cv_accuracy": float(best_cv_acc),
            "gamma": float(gamma),
            "s_norm": float(np.linalg.norm(best_s)),
            "intercept": float(best_c),
            "embed_row_mean_maxabs": float(np.max(np.abs(new_embed.mean(axis=1)))),
            "embed_row_var_maxdev": float(np.max(np.abs(new_embed.var(axis=1) - 1.0))),
        }

        return {"weights": w, "diagnostics": diagnostics}
