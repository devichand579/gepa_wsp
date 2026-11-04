import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # 0. helpers
    # ------------------------------------------------------------------
    def lin(x, W, b):
        # PyTorch-style linear: W [out,in], b [out]
        return x @ W.T + b

    def layer_norm(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * g + b

    def forward(ids, W):
        """Exact re-implementation of the described architecture."""
        N, T = ids.shape
        E = W["embed.weight"]
        P = W["pos.weight"]
        h = E[ids] + P[None, :, :]                       # [N,T,32]
        mask = (ids != 0)[:, :, None].astype(np.float64)
        for bname in ("block0", "block1"):
            z = layer_norm(h, W[bname + ".norm1.weight"], W[bname + ".norm1.bias"])
            qkv = lin(z, W[bname + ".qkv.weight"], W[bname + ".qkv.bias"])
            q, k, v = np.split(qkv, 3, axis=-1)
            att = np.zeros((N, T, T))
            for hd in range(2):
                sl = slice(hd * 16, (hd + 1) * 16)
                sc = np.einsum("ntd,nsd->nts", q[:, :, sl], k[:, :, sl]) / 4.0
                sc = np.where(mask[:, None, :, 0] > 0, sc, -1e9)
                sc = sc - sc.max(-1, keepdims=True)
                e = np.exp(sc)
                att += np.einsum("nts,nsd->ntd", e / e.sum(-1, keepdims=True), v[:, :, sl])
            h = h + lin(att, W[bname + ".proj.weight"], W[bname + ".proj.bias"])
            z2 = layer_norm(h, W[bname + ".norm2.weight"], W[bname + ".norm2.bias"])
            a = np.maximum(lin(z2, W[bname + ".ff1.weight"], W[bname + ".ff1.bias"]), 0.0)
            h = h + lin(a, W[bname + ".ff2.weight"], W[bname + ".ff2.bias"])
        h = layer_norm(h, W["norm.weight"], W["norm.bias"])
        pooled = (h * mask).sum(1) / np.maximum(mask.sum(1), 1.0)
        logits = lin(pooled, W["head.weight"], W["head.bias"])
        return pooled, logits

    # ------------------------------------------------------------------
    # 1. sanity: reproduce the cached activations with the given weights
    # ------------------------------------------------------------------
    W0 = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
    ids_all = np.asarray(f["x"], dtype=np.int64)
    y_all = np.asarray(f["y"], dtype=np.int64)
    bid = np.asarray(f["batch.ids"], dtype=np.int64)

    pooled_chk, logits_chk = forward(ids_all[bid], W0)
    feat_err = float(np.abs(pooled_chk - np.asarray(f["features"], dtype=np.float64)).max())
    log_err = float(np.abs(logits_chk - np.asarray(f["logits"], dtype=np.float64)).max())
    acc_chk = float(((logits_chk.argmax(1) == y_all[bid]).mean()))

    # ------------------------------------------------------------------
    # 2. feature construction
    #    We build an embedding whose rows are zero-mean / unit-std so that
    #    LayerNorm(embed[id]) == embed[id] exactly, and whose 32 dims split
    #    into two independent "one-hot-per-word" codebooks:
    #       dims  0..15 : codebook A  -> pooled dim j = count_{v_j}/n
    #       dims 16..31 : codebook B  -> pooled dim j = count_{v'_j}/n
    #    Each codebook direction is a distinct unit vector, so the two
    #    families are independent and can be mixed by the head.
    # ------------------------------------------------------------------
    V = 2000
    D = 32
    rng = np.random.default_rng(12345)

    # ---- choose the two word lists from closed-form statistics --------
    # word -> document frequency, and class-conditional means of the
    # *current* model's per-token sentiment signal (from cached features).
    df = np.zeros(V)
    df_pos = np.zeros(V)
    df_neg = np.zeros(V)
    for i in range(ids_all.shape[0]):
        row = ids_all[i]
        u = np.unique(row[row != 0])
        df[u] += 1
        if y_all[i] == 1:
            df_pos[u] += 1
        else:
            df_neg[u] += 1
    # token-level polarity signal from the current head direction
    hw = W0["head.weight"]
    dvec = hw[1] - hw[0]
    dvec = dvec / (np.linalg.norm(dvec) + 1e-12)
    Ecur = W0["embed.weight"]
    gcur = Ecur @ dvec                       # per-word scalar signal
    # class-conditional mean of g over documents containing the word
    sum_g = np.zeros(V); cnt_g = np.zeros(V)
    for i in range(ids_all.shape[0]):
        row = ids_all[i]
        u = np.unique(row[row != 0])
        sum_g[u] += gcur[u]; cnt_g[u] += 1
    mean_g = np.where(cnt_g > 0, sum_g / np.maximum(cnt_g, 1), 0.0)
    polarity = (df_pos * mean_g) / np.maximum(df_pos, 1) - (df_neg * mean_g) / np.maximum(df_neg, 1)
    dfreq = df / ids_all.shape[0]

    # rank words by |polarity| * log(1+df) : informative & frequent
    score = np.abs(polarity) * np.log1p(dfreq * ids_all.shape[0])
    order = np.argsort(-score)
    words = [int(t) for t in order if df[t] >= 5][:64]        # 64 signal words
    if len(words) < 64:
        words = [int(t) for t in order][:64]
    # codebook A: the 16 most informative words (count features)
    listA = words[:16]
    # codebook B: the next 16 (different words -> independent directions)
    listB = words[16:32]

    # ---- build embedding ---------------------------------------------
    # For codebook A: word v in listA gets row = dA_{idx} (unit vector);
    #   all other words get a filler row that is zero-mean/unit-std and
    #   orthogonal to every dA_j so it contributes nothing to the A dims.
    # For codebook B: same with dB_j.
    # We need each row to be zero-mean and unit-std (so LayerNorm is identity)
    # AND to have zero projection on the codebook directions it does not own.
    # Construction: row = sum over owned dims of c_j * d_j  + filler,
    # where filler is chosen to (a) zero the mean, (b) give unit variance,
    # (c) have zero inner product with all d_j.
    # Use an explicit orthonormal basis of R^32: e_0..e_31.
    #   codebook A uses dims 0..15, codebook B uses dims 16..31.
    #   For a word owning A-index i only: row = a_i * e_i + filler
    #   filler must live in the span of dims not owned -> but all dims are
    #   used by the two codebooks. Instead: filler lives in the *other*
    #   codebook's dims with zero-sum, and we then subtract the projection
    #   of the owned direction onto it. Simpler: use a 3-part construction:
    #   row = a_i*e_i + b_i*(e_{16} - e_{17}) style filler that is
    #   orthogonal to e_i and to every other owned direction.
    # Because codebook A owns dims 0..15 and B owns 16..31, a word owning
    # only A-index i can use filler in dims {16..31} with zero sum; a word
    # owning only B-index j uses filler in dims {0..15} with zero sum; a
    # word owning both uses no filler. Then:
    #   - mean over dims: filler has zero sum, owned dims contribute a_i
    #     (single) or a_i+b_j -> we subtract the mean explicitly.
    #   - variance: we rescale the whole row to unit std.
    #   - owned-direction projections are exactly a_i (resp. b_j) after
    #     the mean-subtraction? No: subtracting the mean changes the
    #     projection by mean*(sum of d) = 0 since d are unit basis vectors
    #     with a single nonzero -> projection of the mean vector onto e_i
    #     is mean_i. We account for it by solving the small 2x2 system
    #     below so that the *final* (mean-subtracted, rescaled) row has
    #     exactly the intended projection onto the owned dims.

    # Build per-word raw row, then post-process to satisfy:
    #   (1) row.mean() == 0
    #   (2) row.std() == 1
    #   (3) row[i] == target_A_i for owned A dims (exactly)
    #   (4) row[16+j] == target_B_j for owned B dims (exactly)
    # We use a simple, exact scheme: choose the filler so that after mean
    # subtraction the owned entries hit their targets.
    # Let owned A index i with target tA, owned B index j with target tB.
    # raw = tA*e_i + tB*e_{16+j} + filler, filler supported on free dims
    # with zero sum. After mean subtraction m = (tA+tB)/32 (filler sum 0):
    #   entry i  -> tA - m ; entry 16+j -> tB - m
    # We want final entry i == tA' etc. Choose tA,tB so that after
    # subtraction the values are the ones we want; equivalently set
    #   raw_i = tA' + m, raw_{16+j} = tB' + m with m = (raw_i+raw_{16+j})/32
    # Solve: raw_i = tA' + (raw_i+raw_{16+j})/32  -> 31 raw_i - raw_{16+j} = 32 tA'
    #        raw_{16+j} = tB' + (raw_i+raw_{16+j})/32 -> -raw_i + 31 raw_{16+j} = 32 tB'
    # => raw_i = (32*(31 tA' + tB'))/960 , raw_{16+j} = (32*(tA' + 31 tB'))/960
    # Then fill the remaining free dims with a zero-sum filler of the
    # required variance, then rescale is NOT allowed (would break targets).
    # Instead: choose filler magnitude to make variance exactly 1.
    # Variance of the final row (mean 0) = (sum of squares)/32.
    # sum of squares = raw_i^2 + raw_{16+j}^2 + sum(filler^2)
    # We need sum(filler^2) = 32 - raw_i^2 - raw_{16+j}^2 >= 0.
    # Filler dims: free dims (not owned). Use an equal-magnitude ± pattern
    # over free dims with zero sum (pairs +v,-v).

    Enew = np.zeros((V, D))
    # filler helper: distribute variance over free dims using +/- pairs
    def make_row(iA, tA, iB, tB, rng):
        row = np.zeros(D)
        if iA >= 0:
            row[iA] = 1.0
        if iB >= 0:
            row[16 + iB] = 1.0
        free = [d for d in range(D)
                if not (iA >= 0 and d == iA) and not (iB >= 0 and d == 16 + iB)]
        # raw values after mean-subtraction target solve
        if iA >= 0 and iB >= 0:
            ri = 32.0 * (31.0 * tA + tB) / 960.0
            rb = 32.0 * (tA + 31.0 * tB) / 960.0
        elif iA >= 0:
            ri = 32.0 * (31.0 * tA + 0.0) / 960.0
            rb = 0.0
        elif iB >= 0:
            ri = 0.0
            rb = 32.0 * (0.0 + 31.0 * tB) / 960.0
        else:
            ri = 0.0; rb = 0.0
        row[:] = 0.0
        if iA >= 0:
            row[iA] = ri
        if iB >= 0:
            row[16 + iB] = rb
        # variance budget
        ss = row @ row
        need = 32.0 - ss
        if need < 0:
            # shrink targets slightly (should not happen with |t|<=1)
            row *= np.sqrt(max(32.0 - need, 1e-9) / max(ss, 1e-9))
            need = 32.0 - row @ row
        if need > 1e-9 and len(free) >= 2:
            # use pairs (+v,-v) to keep zero sum
            npair = len(free) // 2
            v = np.sqrt(need / (2.0 * npair))
            for p in range(npair):
                row[free[2 * p]] += v
                row[free[2 * p + 1]] -= v
        # final mean subtraction & variance normalisation
        row = row - row.mean()
        s = row.std()
        if s > 1e-9:
            row = row / s
        return row

    # We want, for a word owning A-index i (target tA=1) the *final* row to
    # have entry i equal to +1 (so pooled dim i = count/n) and for B-index j
    # entry 16+j equal to +1.  Because of the mean-subtraction the final
    # entry is tA' - m_final; we instead solve directly by construction:
    # choose raw values so that after subtracting the mean the owned entries
    # equal the desired targets.  Solve the 2x2 system for raw values:
    #   raw_i - m = tA,  raw_b - m = tB,  m = (raw_i + raw_b + 0)/32
    #   => 31 raw_i - raw_b = 32 tA ;  -raw_i + 31 raw_b = 32 tB
    #   => raw_i = (32*(31 tA + tB))/960 , raw_b = (32*(tA + 31 tB))/960
    # (as coded).  Then final entries are exactly tA, tB.

    for idx, v in enumerate(listA):
        Enew[v] = make_row(idx, 1.0, -1, 0.0, rng)
    for idx, v in enumerate(listB):
        Enew[v] = make_row(-1, 0.0, idx, 1.0, rng)
    # neutral filler rows for all other words: zero-mean, unit-std, and
    # orthogonal to every codebook direction (support only on dims 16..31
    # for A-words? no: neutral words must not project on ANY owned dim)
    # Use dims that are never owned: none. So instead build neutral rows
    # as zero-sum patterns on ALL dims with tiny magnitude and then
    # explicitly project out the owned basis vectors.
    owned = np.zeros(D)
    owned[list(range(16))] = 1.0
    owned[16:32] = 1.0
    neutral = rng.normal(size=(V, D))
    neutral -= neutral.mean(1, keepdims=True)
    # project out owned dims (they are the standard basis -> just zero them)
    neutral[:, :32] = neutral[:, :32]
    for v in range(V):
        if v in listA or v in listB:
            continue
        r = neutral[v].copy()
        r[:16] = 0.0
        r[16:32] = 0.0
        # need non-zero variance -> use dims 0..15 with zero sum but that
        # would project onto owned A dims. Instead: neutral words get a
        # row that is *exactly* zero-mean/unit-std with zero entries on all
        # owned dims -> impossible unless we allow extra dims.
        # => Use a 3rd codebook region: we only have 32 dims, so instead
        # neutral words get rows with entries on owned dims that are all
        # EQUAL (so they shift the pooled dims by a constant per token,
        # which the head can absorb via the intercept) and zero-sum
        # deviation elsewhere.  Simplest: neutral row = c * ones(32) + dev
        # with dev zero-sum and supported on all dims, then normalise.
        r = rng.normal(size=D)
        r = r - r.mean()
        r = r / (np.linalg.norm(r) + 1e-12)
        r = r * 0.0
        # neutral: exactly zero vector is invalid (std 0). Use a vector
        # orthogonal to the 32 owned basis vectors -> impossible in R^32.
        # => Instead: give neutral words a row equal to a *permutation*
        # pattern that is zero-mean and whose owned-dim entries are all
        # equal (constant), so pooled owned dims get a constant offset
        # proportional to (n - count_signal)/n, which is a length feature.
        base = np.zeros(D)
        base[:] = 1.0 / np.sqrt(D)
        dev = rng.normal(size=D); dev -= dev.mean()
        dev /= (np.linalg.norm(dev) + 1e-12)
        r = base + 0.0 * dev
        r = r - r.mean()
        s = r.std()
        if s < 1e-9:
            r = np.zeros(D); r[0] = 1.0; r = r - r.mean()
        else:
            r = r / s
        Enew[v] = r

    # rows must be zero-mean & unit-std -> verify & fix
    Enew -= Enew.mean(1, keepdims=True)
    sd = Enew.std(1, keepdims=True)
    bad = (sd[:, 0] < 1e-6)
    if bad.any():
        Enew[bad] = 0.0
        Enew[bad, 0] = 1.0
        Enew[bad] -= Enew[bad].mean(1, keepdims=True)
        Enew[bad] /= Enew[bad].std(1, keepdims=True)
    # re-verify owned-dim exactness after the global mean/std fix
    # (the fix is a per-row affine map; recompute targets analytically)
    # -> do a final exact pass: for each word, set owned entries to targets
    #     and re-balance the rest.
    for v in range(V):
        row = Enew[v]
        iA = listA.index(v) if v in listA else -1
        iB = listB.index(v) if v in listB else -1
        tA = 1.0 if iA >= 0 else 0.0
        tB = 1.0 if iB >= 0 else 0.0
        # desired final entries on owned dims
        row = row - row.mean()
        s = row.std()
        if s > 1e-9:
            row = row / s
        # adjust owned entries to targets, then re-centre/rescale the rest
        others = [d for d in range(D)
                  if not (iA >= 0 and d == iA) and not (iB >= 0 and d == 16 + iB)]
        cur = row[others].mean() if len(others) else 0.0
        # we want final owned entries = targets after centring:
        # set raw owned = t + m where m = mean of raw row; solve simply by
        # setting owned raw = t + c and others raw = o + c with c chosen so
        # mean(raw)=0 -> c = -(tA*1 + tB*1 + sum(others_raw)) / 32
        raw = row.copy()
        if iA >= 0:
            raw[iA] = tA
        if iB >= 0:
            raw[16 + iB] = tB
        c = -(raw.sum()) / D
        raw += c
        raw = raw - raw.mean()
        s2 = raw.std()
        if s2 > 1e-9:
            raw = raw / s2
        Enew[v] = raw

    # ------------------------------------------------------------------
    # 3. assemble candidate weights and simulate
    # ------------------------------------------------------------------
    def build(embed_rows, headW, headB):
        W = {k: v.copy() for k, v in W0.items()}
        W["embed.weight"] = embed_rows
        W["pos.weight"] = np.zeros((64, D))
        for bname in ("block0", "block1"):
            W[bname + ".qkv.weight"] = np.zeros((96, D))
            W[bname + ".qkv.bias"] = np.zeros(96)
            W[bname + ".proj.weight"] = np.zeros((D, D))
            W[bname + ".proj.bias"] = np.zeros(D)
            W[bname + ".ff1.weight"] = np.zeros((64, D))
            W[bname + ".ff1.bias"] = np.zeros(64)
            W[bname + ".ff2.weight"] = np.zeros((D, 64))
            W[bname + ".ff2.bias"] = np.zeros(D)
            W[bname + ".norm1.weight"] = np.ones(D)
            W[bname + ".norm1.bias"] = np.zeros(D)
            W[bname + ".norm2.weight"] = np.ones(D)
            W[bname + ".norm2.bias"] = np.zeros(D)
        W["norm.weight"] = np.ones(D)
        W["norm.bias"] = np.zeros(D)
        W["head.weight"] = headW
        W["head.bias"] = headB
        return W

    # pooled features for the whole pool with the new embedding
    Wc = build(Enew, W0["head.weight"], W0["head.bias"])
    pooled_new, _ = forward(ids_all, Wc)
    pooled_old = np.asarray(f["features"], dtype=np.float64)

    # ------------------------------------------------------------------
    # 4. closed-form head fit (ridge on pooled features, target +-1)
    #    plus a "count" feature block derived analytically from x
    # ------------------------------------------------------------------
    N = ids_all.shape[0]
    # analytic count features for the 64 signal words (count/n and presence)
    counts = np.zeros((N, 64))
    lens = np.zeros(N)
    for i in range(N):
        row = ids_all[i]
        nz = row[row != 0]
        lens[i] = len(nz)
        u, c = np.unique(nz, return_counts=True)
        for uu, cc in zip(u, c):
            j = word_pos.get(int(uu))
            if j is not None:
                counts[i, j] = cc
    counts = counts / np.maximum(lens[:, None], 1.0)
    presence = (counts > 0).astype(np.float64)

    def fit_head(X, y, lams):
        # X: [N,d] features (already standardized), y in {0,1}
        t = 2.0 * y - 1.0
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        # closed-form ridge over a small lambda grid, chosen by 5-fold CV
        # (pure matrix algebra, no iterative optimisation)
        n = Xb.shape[0]
        idx = np.arange(n)
        folds = np.array_split(idx % 5, 5)
        best = None
        for lam in lams:
            accs = []
            for fo in folds:
                tr = np.setdiff1d(idx, fo)
                A = Xb[tr].T @ Xb[tr] + lam * np.eye(Xb.shape[1])
                b = Xb[tr].T @ t[tr]
                w = np.linalg.solve(A, b)
                s = Xb[fo] @ w
                accs.append(float(((s > 0).astype(np.int64) == y[fo]).mean()))
            cv = float(np.mean(accs))
            if best is None or cv > best[1]:
                best = (lam, cv)
        lam = best[0]
        A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
        b = Xb.T @ t
        w = np.linalg.solve(A, b)
        return w, lam, best[1]

    # ---- candidate feature sets -------------------------------------
    # (a) current pooled features (baseline, reproduces today's model)
    # (b) new pooled features (32 codebook dims)
    # (c) new pooled + count + presence features
    def standardize(X):
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True)
        sd[sd < 1e-9] = 1.0
        return (X - mu) / sd

    cands = {}
    cands["old_pooled"] = standardize(pooled_old)
    cands["new_pooled"] = standardize(pooled_new)
    cands["new_pooled+counts"] = np.hstack([standardize(pooled_new),
                                            standardize(counts)])
    cands["new_pooled+counts+pres"] = np.hstack([standardize(pooled_new),
                                                 standardize(counts),
                                                 standardize(presence)])
    cands["old+counts+pres"] = np.hstack([standardize(pooled_old),
                                          standardize(counts),
                                          standardize(presence)])

    lams = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    results = {}
    for name, X in cands.items():
        wv, lam, cv = fit_head(X, y_all, lams)
        s = X @ wv[:-1] + wv[-1]
        acc = float(((s > 0).astype(np.int64) == y_all).mean())
        results[name] = (acc, lam, cv, wv)

    # pick the best by *cross-validated* accuracy (not in-sample), with a
    # preference for the simpler feature set on ties.
    order_pref = ["old_pooled", "new_pooled", "new_pooled+counts",
                  "old+counts+pres", "new_pooled+counts+pres"]
    best_name = None
    for nm in order_pref:
        if best_name is None or results[nm][2] > results[best_name][2] + 1e-9:
            best_name = nm
    acc_best, lam_best, cv_best, wv_best = results[best_name]

    # ------------------------------------------------------------------
    # 5. realise the fitted head inside the architecture
    #    logits = head.weight @ pooled + head.bias  -> we need
    #    head.weight[1]-head.weight[0] = wv[:-1] (on standardized features)
    #    so we must undo the standardization inside the embedding:
    #    standardized pooled = (pooled - mu)/sd
    #    score = sum_j wv_j * (pooled_j - mu_j)/sd_j + wv_last
    #    => head diff row = wv[:-1]/sd  and bias diff = wv_last - sum(wv*mu/sd)
    #    The head has 2 rows; use antisymmetric rows (row1 = -row0) and
    #    put the constant into the bias difference.
    # ------------------------------------------------------------------
    Xb_name = best_name
    Xsel = cands[Xb_name]
    mu = Xsel.mean(0)
    sd = Xsel.std(0)
    sd[sd < 1e-9] = 1.0
    # NOTE: for the count/presence blocks the standardization constants
    # cannot be realised by the head alone (they are not pooled dims).
    # -> Only the "pooled" part of the feature set is realisable through
    #    head.weight (32 dims).  Extra count/presence columns are used
    #    only for *diagnostics / model selection*, and the realised head
    #    uses the pooled block; if the extra blocks help a lot we instead
    #    realise them by re-encoding them into the embedding (see below).
    npool = 32
    w_pool = wv_best[:npool]
    mu_pool = mu[:npool]
    sd_pool = sd[:npool]
    dvec_new = w_pool / sd_pool
    c_new = wv_best[-1] - float((w_pool * mu_pool / sd_pool).sum())

    # ---- realise count/presence extras inside the embedding ----------
    # We can encode additional count features by using *more* codebook
    # directions, but the embedding has only 32 dims and they are already
    # used by the two codebooks.  Instead: choose the final feature set to
    # be one that is fully realisable ("new_pooled") unless the extras
    # give a large CV gain, in which case we rebuild the embedding with
    # 3 codebooks (A:16 counts, B:16 presence) so that BOTH count and
    # presence features live in the 32 pooled dims.
    def build_two_codebooks(listA_, listB_, modeB):
        """modeB: 'count' or 'presence' semantics for the B codebook."""
        En = np.zeros((V, D))
        # A codebook: count semantics -> entry 1 for owner
        # B codebook: presence semantics -> entry 1 for owner, but the
        #   *pooled* value must be presence not count.  Presence is not
        #   obtainable by mean-pooling a per-token indicator (it is a sum
        #   of indicators, i.e. count).  So presence is NOT realisable this
        #   way; instead use B for a *second* count family (different
        #   words) which is what we already do.
        return En

    # ---- final construction: 2 codebooks of 16 words each -------------
    # Re-derive the codebook lists so that the 32 chosen words are the
    # most informative ones, and fit the head on the *realisable* pooled
    # features only (32 dims) -> guaranteed consistent with the network.
    Xreal = standardize(pooled_new)
    wv_r, lam_r, cv_r = fit_head(Xreal, y_all, lams)
    s_r = Xreal @ wv_r[:-1] + wv_r[-1]
    acc_r = float(((s_r > 0).astype(np.int64) == y_all).mean())
    # baseline: old pooled features refit (sanity: should match ~0.878)
    Xold = standardize(pooled_old)
    wv_o, lam_o, cv_o = fit_head(Xold, y_all, lams)
    s_o = Xold @ wv_o[:-1] + wv_o[-1]
    acc_o = float(((s_o > 0).astype(np.int64) == y_all).mean())

    use_new = (cv_r >= cv_o - 1e-9) and (acc_r >= acc_o - 1e-9)
    if use_new:
        Efin = Enew
        w_pool = wv_r[:npool]; mu_pool = mu[:npool]; sd_pool = sd[:npool]
        mu_pool = Xreal.mean(0); sd_pool = Xreal.std(0)
        sd_pool[sd_pool < 1e-9] = 1.0
        dvec_new = w_pool / sd_pool
        c_new = wv_r[-1] - float((w_pool * mu_pool / sd_pool).sum())
        chosen = "new_pooled"
        acc_final = acc_r; lam_final = lam_r; cv_final = cv_r
    else:
        Efin = W0["embed.weight"]
        mu_pool = Xold.mean(0); sd_pool = Xold.std(0)
        sd_pool[sd_pool < 1e-9] = 1.0
        w_pool = wv_o[:npool]
        dvec_new = w_pool / sd_pool
        c_new = wv_o[-1] - float((w_pool * mu_pool / sd_pool).sum())
        chosen = "old_pooled"
        acc_final = acc_o; lam_final = lam_o; cv_final = cv_o

    # head rows: antisymmetric, difference = dvec_new ; bias difference = c_new
    hd = dvec_new / 2.0
    hb = np.array([-c_new / 2.0, c_new / 2.0])
    Wfin = build(Efin, np.stack([-hd, hd]), hb)

    # ------------------------------------------------------------------
    # 6. verify the realised network on the pool and the reflection batch
    # ------------------------------------------------------------------
    pooled_fin, logits_fin = forward(ids_all, Wfin)
    pred = logits_fin.argmax(1)
    acc_pool = float((pred == y_all).mean())
    t = 2.0 * y_all - 1.0
    p1 = 1.0 / (1.0 + np.exp(-(logits_fin[:, 1] - logits_fin[:, 0])))
    ce_pool = float(-np.mean(y_all * np.log(np.clip(p1, 1e-9, 1)) +
                              (1 - y_all) * np.log(np.clip(1 - p1, 1e-9, 1))))
    logits_b, _ = forward(ids_all[bid], Wfin)
    acc_batch = float((logits_b.argmax(1) == y_all[bid]).mean())
    pb = 1.0 / (1.0 + np.exp(-(logits_b[:, 1] - logits_b[:, 0])))
    ce_batch = float(-np.mean(y_all[bid] * np.log(np.clip(pb, 1e-9, 1)) +
                               (1 - y_all[bid]) * np.log(np.clip(1 - pb, 1e-9, 1))))

    # safety: if the realised network is worse than the *current* one on the
    # pool, keep the current weights unchanged (return w untouched).
    cur_acc = float((np.asarray(f["logits"]).argmax(1) == y_all).mean())
    if acc_pool < cur_acc - 1e-9:
        diag = {"status": "kept_current", "cur_acc": cur_acc,
                "cand_acc": acc_pool, "feat_err": feat_err,
                "log_err": log_err, "batch_acc_chk": acc_chk}
        return {"weights": w, "diagnostics": diag}

    # ------------------------------------------------------------------
    # 7. write out (float32, finite, |.| <= 1000)
    # ------------------------------------------------------------------
    out = {}
    for k, v in w.items():
        arr = np.asarray(Wfin[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1000.0, 1000.0)
        out[k] = arr

    diag = {
        "status": "ok",
        "chosen": chosen,
        "forward_check_feat_maxerr": round(feat_err, 8),
        "forward_check_logit_maxerr": round(log_err, 8),
        "forward_check_batch_acc": round(acc_chk, 4),
        "current_pool_acc": round(cur_acc, 4),
        "new_pool_acc": round(acc_pool, 4),
        "new_pool_ce": round(ce_pool, 4),
        "new_batch_acc": round(acc_batch, 4),
        "new_batch_ce": round(ce_batch, 4),
        "cv_acc_new_pooled": round(cv_r, 4),
        "cv_acc_old_pooled": round(cv_o, 4),
        "in_sample_acc_new_pooled": round(acc_r, 4),
        "in_sample_acc_old_pooled": round(acc_o, 4),
        "ridge_lambda": lam_final,
        "codebookA_words": [int(t) for t in listA],
        "codebookB_words": [int(t) for t in listB],
        "head_diff_norm": round(float(np.linalg.norm(dvec_new)), 4),
        "head_bias_diff": round(float(c_new), 4),
        "embed_row_std_range": [round(float(Efin.std(1).min()), 4),
                                round(float(Efin.std(1).max()), 4)],
        "embed_row_mean_abs_max": round(float(np.abs(Efin.mean(1)).max()), 6),
        "candidate_accs": {k: round(float(v[0]), 4) for k, v in results.items()},
        "candidate_cvs": {k: round(float(v[2]), 4) for k, v in results.items()},
    }
    return {"weights": out, "diagnostics": diag}
