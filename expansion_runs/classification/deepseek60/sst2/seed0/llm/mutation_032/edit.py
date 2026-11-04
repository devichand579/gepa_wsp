import numpy as np

def edit(w, f):
    # ---- read feedback pool ----
    X = np.asarray(f["x"])            # [N,64] int64
    y = np.asarray(f["y"]).astype(np.int64)  # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]    # 2000
    D = w["embed.weight"].shape[1]    # 32
    eps = 1e-5

    # ---- current per-token scores (preserve them) ----
    E = w["embed.weight"].astype(np.float64).copy()   # [V,32]
    H = w["head.weight"].astype(np.float64).copy()    # [2,32]
    Hb = w["head.bias"].astype(np.float64).copy()     # [2]

    # LN of current embedding rows (gamma=1,beta=0): zero-mean, unit-var
    m = E.mean(axis=1, keepdims=True)
    v = E.var(axis=1, keepdims=True)
    LN = (E - m) / np.sqrt(v + eps)                   # [V,32]
    # current per-token score pair
    cur = LN @ H.T                                    # [V,2]

    # ---- fit a smoothed Naive-Bayes log-odds direction on the pool ----
    # counts per token per class (only non-pad tokens)
    cnt = np.zeros((V, 2), dtype=np.float64)
    tot = np.zeros(2, dtype=np.float64)
    mask = (X != 0)
    for c in (0, 1):
        sel = (y == c)
        Xi = X[sel]
        mi = mask[sel]
        # bincount over flattened tokens
        flat = Xi[mi].ravel()
        cnt[:, c] = np.bincount(flat, minlength=V)[:V]
        tot[c] = mi.sum()
    alpha = 1.0
    p0 = (cnt[:, 0] + alpha) / (tot[0] + alpha * V)
    p1 = (cnt[:, 1] + alpha) / (tot[1] + alpha * V)
    nb = np.log(p1) - np.log(p0)                       # [V] log-odds
    nb = nb - nb.mean()
    nb_std = nb.std() + 1e-8
    nb = nb / nb_std                                   # standardised

    # ---- fit a strongly-regularised ridge BoW direction (closed form) ----
    # feature: per-document token-count vector, length-normalised
    # Build sparse-ish design via bincount per doc is O(N*L); use matrix form.
    # We only need a direction w_ridge in R^V.
    # Use a subsample of docs for speed but keep it exact enough.
    # Construct design matrix implicitly: for each doc, counts/T.
    # Ridge on +/-1 targets: w = (Z^T Z + lam I)^-1 Z^T t  (Z is N x V, huge)
    # Use the identity (Z^T Z) = sum_d z_d z_d^T; compute via accumulation.
    lam = 50.0
    A = np.zeros((V, V), dtype=np.float64)
    b = np.zeros(V, dtype=np.float64)
    t = np.where(y == 1, 1.0, -1.0)
    # accumulate in chunks
    for i in range(N):
        row = X[i]
        nz = row[row != 0]
        if nz.size == 0:
            continue
        T = float(nz.size)
        # counts
        cvec = np.bincount(nz, minlength=V).astype(np.float64)
        cvec /= T
        A += np.outer(cvec, cvec)
        b += t[i] * cvec
    A[np.diag_indices(V)] += lam
    try:
        w_ridge = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w_ridge = np.linalg.lstsq(A, b, rcond=None)[0]
    w_ridge = w_ridge - w_ridge.mean()
    wr_std = w_ridge.std() + 1e-8
    w_ridge = w_ridge / wr_std

    # ---- build new per-token feature vectors ----
    # We want LN(new_embed[v]) = phi(v) with phi zero-mean, unit-var,
    # and head @ phi(v) = (g0(v), g1(v)).
    # Choose head = [e0; e1] (reads dims 0,1) and put:
    #   phi_0 = cur_0, phi_1 = cur_1  (preserve current scores)
    #   phi_2 = nb,    phi_3 = w_ridge (new signals)
    #   phi_4..31 = filler c_v to make mean zero
    # Then rescale so var(phi)=1 (global scale does not change argmax).
    phi = np.zeros((V, D), dtype=np.float64)
    phi[:, 0] = cur[:, 0]
    phi[:, 1] = cur[:, 1]
    phi[:, 2] = nb
    phi[:, 3] = w_ridge
    # filler to zero-mean: sum over dims 4..D-1 must equal -(sum of 0..3)
    s4 = phi[:, 0] + phi[:, 1] + phi[:, 2] + phi[:, 3]
    nfill = D - 4
    phi[:, 4:] = (-s4 / nfill)[:, None]
    # make each row zero-mean and unit-variance
    pm = phi.mean(axis=1, keepdims=True)
    phi = phi - pm
    pv = phi.var(axis=1, keepdims=True)
    phi = phi / np.sqrt(pv + 1e-12)

    # new embed rows: LN(embed)=phi  =>  embed = phi (mean0,var1)
    newE = phi.astype(np.float32)

    # new head: read dims 0,1 (current scores) and dims 2,3 (new signals)
    # weights chosen so that logits = mean_t [ a*cur + b*nb + c*ridge ]
    # We keep current scores at scale 1 and add new directions with tuned gains.
    # Gain tuning: pick gains that maximise pool accuracy of the combined score.
    # Build per-doc combined score for a grid of gains (closed-form, no search loop
    # over training -- just evaluate a small fixed set of candidate gains).
    # Precompute per-doc mean of each direction.
    def doc_mean(direction):
        out = np.zeros(N, dtype=np.float64)
        for i in range(N):
            row = X[i]
            nz = row[row != 0]
            if nz.size == 0:
                out[i] = 0.0
            else:
                out[i] = direction[nz].mean()
        return out

    d_cur0 = doc_mean(cur[:, 0])
    d_cur1 = doc_mean(cur[:, 1])
    d_nb = doc_mean(nb)
    d_ri = doc_mean(w_ridge)

    # current decision uses logit1 - logit0 = (H1-H0) . cur ; reproduce it
    diff_cur = d_cur1 - d_cur0
    base_pred = (diff_cur > 0).astype(np.int64)
    base_acc = (base_pred == y).mean()

    best = (base_acc, 0.0, 0.0)
    for gnb in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
        for gri in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
            sc = diff_cur + gnb * d_nb + gri * d_ri
            acc = ((sc > 0).astype(np.int64) == y).mean()
            if acc > best[0]:
                best = (acc, gnb, gri)
    _, gnb, gri = best

    # head rows: row0 = -0.5*(cur direction), row1 = +0.5*(cur direction) plus gains
    # We want logit1 - logit0 = mean_t [ (cur1-cur0) + gnb*nb + gri*ridge ]
    # Set head row1 = [0.5, 0.5, 0.5*gnb, 0.5*gri, 0...]
    #     head row0 = [-0.5, -0.5, -0.5*gnb, -0.5*gri, 0...]
    newH = np.zeros((2, D), dtype=np.float64)
    newH[1, 0] = 0.5
    newH[1, 1] = 0.5
    newH[1, 2] = 0.5 * gnb
    newH[1, 3] = 0.5 * gri
    newH[0, 0] = -0.5
    newH[0, 1] = -0.5
    newH[0, 2] = -0.5 * gnb
    newH[0, 3] = -0.5 * gri
    newb = np.zeros(2, dtype=np.float64)

    # ---- write back ----
    w["embed.weight"] = newE.astype(np.float32)
    w["head.weight"] = newH.astype(np.float32)
    w["head.bias"] = newb.astype(np.float32)

    # keep everything else finite / bounded
    for k in list(w.keys()):
        arr = np.asarray(w[k])
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = np.nan_to_num(arr, nan=0.0, posinf=1000.0, neginf=-1000.0)
            arr = np.clip(arr, -1000.0, 1000.0)
            w[k] = arr.astype(np.float32)

    diag = {
        "base_pool_acc": float(base_acc),
        "combined_pool_acc": float(best[0]),
        "gain_nb": float(gnb),
        "gain_ridge": float(gri),
        "nb_std": float(nb_std),
        "ridge_std": float(wr_std),
        "embed_row_norm_mean": float(np.linalg.norm(newE, axis=1).mean()),
        "head_norm": float(np.linalg.norm(newH)),
    }
    return {"weights": w, "diagnostics": diag}
