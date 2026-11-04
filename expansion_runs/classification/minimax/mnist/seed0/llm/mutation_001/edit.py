def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"], dtype=np.int64).ravel()
    N, D = X.shape
    C = 10
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    W1 = np.asarray(w["fc1.weight"], dtype=np.float64)
    b1 = np.asarray(w["fc1.bias"], dtype=np.float64)
    W2 = np.asarray(w["fc2.weight"], dtype=np.float64)
    b2 = np.asarray(w["fc2.bias"], dtype=np.float64)

    rng = np.random.default_rng(12345)

    # ---------------- closed-form ridge helper (normal equations + analytic LOO) ----
    def ridge_fit(H, T, lam):
        """H:(n,p) design (no intercept), T:(n,k) targets. Returns (B, b0, loo_err)."""
        n, p = H.shape
        G = H.T @ H
        R = H.T @ T
        A = G + lam * np.eye(p)
        try:
            B = np.linalg.solve(A, R)
        except np.linalg.LinAlgError:
            B = np.linalg.lstsq(A, R, rcond=None)[0]
        r = T - H @ B
        b0 = r.mean(axis=0)
        # analytic leave-one-out via hat matrix diagonal (SVD based)
        try:
            U, s, Vt = np.linalg.svd(H, full_matrices=False)
            k = len(s)
            s2 = s ** 2
            d = np.zeros(n)
            for i in range(k):
                if s2[i] > 1e-12:
                    u = U[:, i]
                    d += (u ** 2) * (s2[i] / (s2[i] + lam))
            yhat = H @ B + b0
            e = yhat - T
            denom = np.maximum(1.0 - d, 1e-6)
            loo = (e / denom[:, None]) ** 2
            loo_err = float(loo.mean())
        except np.linalg.LinAlgError:
            loo_err = float(((H @ B + b0 - T) ** 2).mean())
        return B, b0, loo_err

    def pick_lam(H, T, lams):
        best = None
        for lam in lams:
            B, b0, err = ridge_fit(H, T, lam)
            if best is None or err < best[0]:
                best = (err, lam, B, b0)
        return best

    # ---------------- stage 1: linear classifier on pixels ----------------
    Xc = X - X.mean(axis=0)
    lams = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    err, lam, A, b_lin = pick_lam(Xc, Y, lams)
    lin_logits = Xc @ A + b_lin
    lin_acc = float((lin_logits.argmax(1) == y).mean())

    # ---------------- hidden unit design ----------------
    # (a) linear block: 16 rows = class score directions (10) + 6 discriminative dirs
    lin_dirs = A.T.copy()                       # (784, 10)
    # extra directions: class-mean differences (6 most separated class pairs)
    mu = np.zeros((C, D))
    for c in range(C):
        mu[c] = X[y == c].mean(axis=0)
    diffs = []
    for c in range(C):
        for c2 in range(c + 1, C):
            diffs.append((np.linalg.norm(mu[c] - mu[c2]), c, c2))
    diffs.sort(reverse=True)
    extra = []
    for _, c, c2 in diffs[:6]:
        extra.append(mu[c] - mu[c2])
    lin_dirs = np.concatenate([lin_dirs, np.array(extra)], axis=0)   # (16, 784)
    L = lin_dirs.shape[0]

    # scale so that the linear projections have comparable spread
    proj = X @ lin_dirs.T
    s_lin = proj.std(axis=0) + 1e-8
    lin_dirs = lin_dirs / s_lin[None, :]          # unit-ish scale

    # (b) prototype block: 14 prototypes (class-balanced) x 8 thresholds = 112
    P = 14
    protos = []
    per = max(1, P // C)
    for c in range(C):
        idx = np.where(y == c)[0]
        # pick prototypes spread over the class (farthest-point style, deterministic)
        sel = []
        cur = idx[0]
        for _ in range(per):
            sel.append(cur)
            d = ((X[idx] - X[cur]) ** 2).sum(axis=1)
            cur = idx[int(np.argmax(d))]
        protos.extend(sel)
    protos = protos[:P]
    V = X[protos]                                  # (P, 784)
    vnorm = (V ** 2).sum(axis=1) / 2.0              # (P,)
    sim = X @ V.T - vnorm[None, :]                  # (N, P) similarity ramps
    # thresholds: quantiles of the similarity distribution per prototype
    n_thr = 8
    thr = np.zeros((P, n_thr))
    for p in range(P):
        q = np.quantile(sim[:, p], np.linspace(0.15, 0.95, n_thr))
        thr[p] = q
    # build nonlinear hidden rows: relu(sim_p - thr_pk)
    nl_rows = []
    for p in range(P):
        for k in range(n_thr):
            v = V[p].copy()
            nl_rows.append((v, -vnorm[p] - thr[p, k]))
    nl_rows = nl_rows[:112]
    NL = len(nl_rows)

    # assemble fc1
    newW1 = np.zeros((128, 784))
    newb1 = np.zeros(128)
    # linear block rows 0..L-1 : need a_j.x + b_j > 0 for all x -> choose b_j
    for j in range(L):
        a = lin_dirs[j]
        vals = X @ a
        bmin = vals.min()
        newW1[j] = a
        newb1[j] = -bmin + 0.5 * (np.abs(bmin) + 1.0) + 1.0
    # nonlinear block rows L..L+NL-1
    for t, (v, b) in enumerate(nl_rows):
        newW1[L + t] = v
        newb1[L + t] = b

    # ---------------- compute hidden features with the NEW fc1 ----------------
    H = np.maximum(X @ newW1.T + newb1[None, :], 0.0)      # (N, 128)

    # split blocks
    Hlin = H[:, :L]
    Hnl = H[:, L:L + NL]

    # standardize nonlinear block (helps ridge conditioning); keep linear block raw
    mu_nl = Hnl.mean(axis=0)
    sd_nl = Hnl.std(axis=0) + 1e-8
    Hnl_s = (Hnl - mu_nl) / sd_nl

    # ---------------- stage 2: nonlinear block fits residual of linear model ----
    R = Y - lin_logits
    lams_nl = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    err_nl, lam_nl, Bn, b0n = pick_lam(Hnl_s, R, lams_nl)
    nl_pred = Hnl_s @ Bn + b0n
    nl_acc = float(((lin_logits + nl_pred).argmax(1) == y).mean())

    # ---------------- stage 3: refit linear block on remaining residual ----
    R2 = R - nl_pred
    err_lin, lam_lin, A2, b2_lin = pick_lam(Xc, R2, lams)
    final_logits = lin_logits + nl_pred + (Xc @ A2 + b2_lin)
    final_acc = float((final_logits.argmax(1) == y).mean())

    # ---------------- write back into fc2 (fold standardization + intercepts) ----
    # hidden feature j contributes: W2[c,j] * H[:,j] ; we must reproduce
    #   final_logits[:,c] = (Xc@A + b_lin) + (Hnl_s@Bn + b0n) + (Xc@A2 + b2_lin)
    # linear part: Xc@A = X@A - meanX@A  -> use raw X with adjusted bias
    meanX = X.mean(axis=0)
    W2n = np.zeros((C, 128))
    b2n = np.zeros(C)
    # linear block columns
    W2n[:, :L] = A.T + A2.T
    b2n += b_lin - meanX @ (A + A2) + b2_lin
    # nonlinear block columns (standardized)
    W2n[:, L:L + NL] = (Bn / sd_nl[None, :]).T
    b2n += b0n - (Bn / sd_nl[None, :]).T @ mu_nl

    # ---------------- safety: verify the reconstructed network ----------------
    def forward(W1m, b1m, W2m, b2m, Xin):
        Hh = np.maximum(Xin @ W1m.T + b1m[None, :], 0.0)
        return Hh @ W2m.T + b2m

    rec = forward(newW1, newb1, W2n, b2n, X)
    rec_acc = float((rec.argmax(1) == y).mean())
    rec_err = float(np.abs(rec - final_logits).max())

    # ---------------- write results ----------------
    outW1 = np.clip(newW1, -1000.0, 1000.0).astype(np.float32)
    outb1 = np.clip(newb1, -1000.0, 1000.0).astype(np.float32)
    outW2 = np.clip(W2n, -1000.0, 1000.0).astype(np.float32)
    outb2 = np.clip(b2n, -1000.0, 1000.0).astype(np.float32)

    # re-verify after clipping
    rec2 = forward(outW1.astype(np.float64), outb1.astype(np.float64),
                   outW2.astype(np.float64), outb2.astype(np.float64), X)
    rec2_acc = float((rec2.argmax(1) == y).mean())

    w["fc1.weight"] = outW1
    w["fc1.bias"] = outb1
    w["fc2.weight"] = outW2
    w["fc2.bias"] = outb2

    diag = {
        "hypothesis": "closed-form: 16 linear (large-bias) hidden units carrying a ridge pixel classifier + 112 prototype-similarity relu units; ridge readout with analytic LOO lambda selection",
        "train_acc_linear_only": round(lin_acc, 4),
        "train_acc_linear_plus_proto": round(nl_acc, 4),
        "train_acc_final_refit": round(final_acc, 4),
        "train_acc_reconstructed_net": round(rec2_acc, 4),
        "reconstruction_max_abs_logit_diff": round(rec_err, 8),
        "lam_linear": float(lam),
        "lam_nonlinear": float(lam_nl),
        "lam_linear_refit": float(lam_lin),
        "loo_mse_linear": round(err, 5),
        "loo_mse_nonlinear": round(err_nl, 5),
        "loo_mse_linear_refit": round(err_lin, 5),
        "hidden_block_sizes": [int(L), int(NL)],
        "fc1_row_absmax": [round(float(np.abs(outW1[:L]).max()), 4),
                           round(float(np.abs(outW1[L:]).max()), 4)],
        "fc1_bias_linear_min": round(float(outb1[:L].min()), 4),
        "fc1_bias_proto_range": [round(float(outb1[L:].min()), 4),
                                 round(float(outb1[L:].max()), 4)],
        "fc2_absmax": round(float(np.abs(outW2).max()), 4),
        "fc2_bias": [round(float(v), 4) for v in outb2],
        "feat_mean": round(float(H.mean()), 5),
        "feat_std": round(float(H.std()), 5),
        "feat_zero_frac": round(float((H == 0).mean()), 4),
        "logit_std": round(float(final_logits.std()), 4),
    }
    return {"weights": w, "diagnostics": diag}
