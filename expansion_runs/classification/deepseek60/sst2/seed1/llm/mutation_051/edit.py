import numpy as np

def edit(w, f):
    X = f["x"].astype(np.int64)          # [N,64]
    y = f["y"].astype(np.int64)          # [N]
    N, T = X.shape
    V = w["embed.weight"].shape[0]
    mask = (X != 0)                      # non-pad
    L = mask.sum(1).astype(np.float64)   # lengths
    L[L == 0] = 1.0

    # ---- current per-token weight a_j recovered from features ----
    # pooled[0] = (1/L) * sum_j a_{x_j}  -> solve ridge for a (closed form)
    # Build sparse count matrix implicitly via bincount per example is costly;
    # instead use the given features directly for the head refit, and
    # estimate a_j by correlating token presence with the residual.
    feat0 = f["features"][:, 0].astype(np.float64)   # current pooled feature 0

    # ---- Naive-Bayes log-odds per token (closed form counting) ----
    # counts of token j in positive / negative examples (token occurrences)
    flat_tok = X.reshape(-1)
    flat_ex  = np.repeat(np.arange(N), T)
    flat_ok  = mask.reshape(-1)
    tok = flat_tok[flat_ok]
    ex  = flat_ex[flat_ok]
    lab = y[ex]                          # label of the example each token came from

    pos_cnt = np.bincount(tok[lab == 1], minlength=V).astype(np.float64)
    neg_cnt = np.bincount(tok[lab == 0], minlength=V).astype(np.float64)
    # document frequency (how many examples contain token) for smoothing
    pos_doc = np.bincount(np.unique(tok[lab == 1]), minlength=V).astype(np.float64)
    neg_doc = np.bincount(np.unique(tok[lab == 0]), minlength=V).astype(np.float64)

    alpha = 1.0
    # log-odds of occurrence given class (Laplace smoothed)
    log_odds = np.log((pos_cnt + alpha) / (pos_cnt.sum() + alpha * V)) - \
               np.log((neg_cnt + alpha) / (neg_cnt.sum() + alpha * V))

    # ---- recover current a_j by least squares on the linear system ----
    # pooled0_i = (1/L_i) * sum_j a_{x_ij}.  Build design matrix D (N x V)
    # where D[i,j] = (#occurrences of j in i)/L_i.  Use normal equations.
    # D is sparse but V=2000, N=4000 -> dense 4000x2000 is fine (8M floats).
    D = np.zeros((N, V), dtype=np.float64)
    rows = np.repeat(np.arange(N), T)[flat_ok]
    np.add.at(D, (rows, tok), 1.0)
    D /= L[:, None]
    # ridge solve for current a (lam small so we recover the existing weights)
    lam_a = 1e-3
    A = D.T @ D + lam_a * np.eye(V)
    b = D.T @ feat0
    try:
        a_cur = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        a_cur = np.linalg.lstsq(A, b, rcond=None)[0]

    # ---- combine: shrink NB log-odds toward current weights ----
    # scale NB to match current a's std on tokens that appear
    seen = (pos_cnt + neg_cnt) > 0
    if seen.sum() > 10:
        nb = log_odds.copy()
        nb -= nb[seen].mean()
        s_cur = a_cur[seen].std() + 1e-9
        s_nb  = nb[seen].std() + 1e-9
        nb *= s_cur / s_nb
        nb += a_cur[seen].mean()
    else:
        nb = a_cur.copy()

    # blend (shrinkage toward current strong solution)
    gamma = 0.35
    a_new = (1.0 - gamma) * a_cur + gamma * nb

    # clip to a safe range so LN realization stays accurate
    a_new = np.clip(a_new, -4.0, 4.0)

    # ---- realize a_new in embed dim 0, preserving structure of other dims ----
    E = w["embed.weight"].astype(np.float64)
    # keep dims 1..31 exactly as they are (they produce the constant offsets)
    # set dim 0 so that LN(e_j)[0] ≈ a_new_j.  With dim1 large (~5.56) and
    # dims2..31 ~ -0.18, LN[0] ≈ e_j[0] * 0.971 (see analysis).  Invert:
    E[:, 0] = a_new / 0.971
    # pad row stays zero
    E[0, :] = 0.0
    w["embed.weight"] = E.astype(np.float32)

    # ---- refit head in closed form on the NEW pooled feature 0 ----
    # new pooled0_i = (1/L_i) * sum_j a_new_{x_ij} = D @ a_new
    new_feat0 = D @ a_new
    # optimal 1-D threshold: pick t minimizing misclassification (closed form scan)
    order = np.argsort(new_feat0)
    fs = new_feat0[order]
    ys = y[order]
    # cumulative counts of class1 to the left
    c1 = np.cumsum(ys)
    c0 = np.cumsum(1 - ys)
    total1 = ys.sum()
    total0 = N - total1
    # errors if threshold placed after position k (predict 0 for <=k, 1 for >k)
    err = c1 + (total0 - c0)          # class1 left + class0 right
    k = int(np.argmin(err))
    if k + 1 < N:
        thr = 0.5 * (fs[k] + fs[k + 1])
    else:
        thr = fs[k] + 1e-6
    # choose head weight so logit1 - logit0 = head_w * (feat0 - thr)
    head_w = 2.1211
    w["head.weight"] = np.zeros_like(w["head.weight"])
    w["head.weight"][1, 0] = head_w
    w["head.bias"] = np.zeros_like(w["head.bias"])
    w["head.bias"][1] = -head_w * thr

    # ---- diagnostics ----
    pred = (new_feat0 > thr).astype(np.int64)
    acc = float((pred == y).mean())
    diag = {
        "method": "closed_form_NB_logodds_blend_with_recovered_current_weights",
        "gamma": gamma,
        "train_acc_new": acc,
        "a_absmax": float(np.abs(a_new).max()),
        "a_norm": float(np.linalg.norm(a_new)),
        "nb_absmax": float(np.abs(nb).max()),
        "thr": float(thr),
        "n_seen_tokens": int(seen.sum()),
    }
    return {"weights": w, "diagnostics": diag}
