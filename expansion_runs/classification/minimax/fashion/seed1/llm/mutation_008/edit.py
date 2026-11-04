import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x [N,C,H,W], W [O,C,3,3], padding=1, stride=1, cross-correlation
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, O, H, Wd), dtype=np.float32)
        for u in range(3):
            for v in range(3):
                out += np.einsum('nchw,oc->nohw',
                                 xp[:, :, u:u + H, v:v + Wd],
                                 W[:, :, u, v].astype(np.float32))
        out += b.astype(np.float32)[None, :, None, None]
        return out

    def maxpool2(x):
        N, C, H, Wd = x.shape
        h2, w2 = H // 2, Wd // 2
        r = x[:, :, :h2 * 2, :w2 * 2].reshape(N, C, h2, 2, w2, 2)
        return r.max(axis=(3, 5))

    def forward(x, w1, b1, w2, b2, hw, hb):
        h1 = maxpool2(np.maximum(conv2d(x, w1, b1), 0.0))
        h2 = maxpool2(np.maximum(conv2d(h1, w2, b2), 0.0))
        feat = h2.reshape(h2.shape[0], -1)          # channel,row,col
        logits = feat @ hw.T + hb
        return feat, logits

    def ridge_fit(X, Y, lam):
        # closed form ridge with intercept (normal equations)
        N, P = X.shape
        xm = X.mean(axis=0)
        ym = Y.mean(axis=0)
        Xc = X - xm
        Yc = Y - ym
        G = (Xc.T @ Xc).astype(np.float64)
        B = (Xc.T @ Yc).astype(np.float64)
        d = np.trace(G) / max(P, 1)
        A = G + lam * max(d, 1e-12) * np.eye(P)
        W = np.linalg.solve(A, B)
        b = ym - xm @ W
        return W, b

    def gcv_lambda(X, Y, lams):
        # analytic GCV over a grid using one eigendecomposition (closed form)
        N, P = X.shape
        xm = X.mean(axis=0); ym = Y.mean(axis=0)
        Xc = (X - xm).astype(np.float64)
        Yc = (Y - ym).astype(np.float64)
        G = Xc.T @ Xc
        B = Xc.T @ Yc
        evals, evecs = np.linalg.eigh(G)
        evals = np.clip(evals, 0.0, None)
        Bt = evecs.T @ B
        d = max(np.trace(G) / max(P, 1), 1e-12)
        best = None
        for lam in lams:
            inv = 1.0 / (evals + lam * d)
            W = evecs @ (inv[:, None] * Bt)
            R = Yc - Xc @ W
            rss = float(np.sum(R * R))
            tr_h = float(np.sum(evals * inv))
            denom = max(1.0 - tr_h / N, 1e-6)
            gcv = (rss / N) / (denom * denom)
            if best is None or gcv < best[0]:
                best = (gcv, lam, W)
        return best[1], best[2], best[0]

    def acc(X, Y, W, b):
        pred = np.argmax(X @ W + b, axis=1)
        return float((pred == Y.argmax(axis=1)).mean())

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"]).astype(np.int64)
    N = x.shape[0]
    Y1h = np.zeros((N, 10), dtype=np.float64)
    Y1h[np.arange(N), y] = 1.0

    w1 = np.asarray(w["conv1.weight"], dtype=np.float32)
    b1 = np.asarray(w["conv1.bias"], dtype=np.float32)
    w2 = np.asarray(w["conv2.weight"], dtype=np.float32)
    b2 = np.asarray(w["conv2.bias"], dtype=np.float32)
    hw = np.asarray(w["head.weight"], dtype=np.float32)
    hb = np.asarray(w["head.bias"], dtype=np.float32)

    # ---------------- reproduce forward (chunked) ----------------
    def forward_all(x, w1, b1, w2, b2, hw, hb, chunk=250):
        feats = []
        lgs = []
        for i in range(0, x.shape[0], chunk):
            ft, lg = forward(x[i:i + chunk], w1, b1, w2, b2, hw, hb)
            feats.append(ft)
            lgs.append(lg)
        return np.concatenate(feats, 0), np.concatenate(lgs, 0)

    feat_cur, logit_cur = forward_all(x, w1, b1, w2, b2, hw, hb)
    ref_feat = np.asarray(f["features"], dtype=np.float32)
    ref_log = np.asarray(f["logits"], dtype=np.float32)
    ferr = float(np.max(np.abs(feat_cur - ref_feat))) if feat_cur.shape == ref_feat.shape else 1e9
    lerr = float(np.max(np.abs(logit_cur - ref_log))) if logit_cur.shape == ref_log.shape else 1e9
    fwd_ok = bool(ferr < 2e-3 and lerr < 2e-2)

    diag = {"fwd_feature_maxerr": ferr, "fwd_logit_maxerr": lerr, "fwd_ok": fwd_ok}

    # ---------------- candidate feature sets ----------------
    # conv1 pooled maps (needed to design extra conv2 channels)
    def conv1_pool(x, w1, b1, chunk=250):
        outs = []
        for i in range(0, x.shape[0], chunk):
            outs.append(maxpool2(np.maximum(conv2d(x[i:i + chunk], w1, b1), 0.0)))
        return np.concatenate(outs, 0)          # [N,16,14,14]

    p1 = conv1_pool(x, w1, b1)
    diag["conv1pool_shape"] = list(p1.shape)

    # design 16 extra conv2 channels: oriented local differences of conv1 maps
    # (nonlinear after relu -> information a linear head cannot synthesise)
    w2_ext = w2.copy()
    b2_ext = b2.copy()
    # kernel taps for horizontal / vertical difference on the 14x14 map
    kh = np.zeros((3, 3), dtype=np.float32)
    kh[1, 2] = 0.5; kh[1, 0] = -0.5
    kv = np.zeros((3, 3), dtype=np.float32)
    kv[2, 1] = 0.5; kv[0, 1] = -0.5
    for c in range(16):
        o = 16 + c
        w2_ext[o, c] = kh
    # channels 16..31 are currently (mostly) dead -> overwrite them
    # also keep original channels 0..15 untouched
    feat_ext, _ = forward_all(x, w1, b1, w2_ext, b2_ext, hw, hb)
    diag["ext_feature_mean"] = float(feat_ext.mean())
    diag["ext_feature_zerofrac"] = float((feat_ext == 0).mean())

    # ---------------- split for honest comparison ----------------
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    ntr = int(0.7 * N)
    tr, va = idx[:ntr], idx[ntr:]

    def evaluate(Xall):
        Xtr, Xva = Xall[tr], Xall[va]
        Ytr = Y1h[tr]
        lam, Wg, gcv = gcv_lambda(Xtr, Ytr, [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0])
        Wf, bf = ridge_fit(Xtr, Ytr, lam)
        a = acc(Xva, Y1h[va], Wf, bf)
        return a, lam, gcv

    base_va = float((np.argmax(logit_cur[va], axis=1) == y[va]).mean())
    diag["baseline_val_acc"] = base_va

    results = {}
    results["cur"] = evaluate(feat_cur)
    results["ext"] = evaluate(feat_ext)
    diag["val_acc_cur_refit"] = results["cur"][0]
    diag["val_acc_ext_refit"] = results["ext"][0]
    diag["lam_cur"] = results["cur"][1]
    diag["lam_ext"] = results["ext"][1]

    # choose best variant (baseline included)
    best_name = "cur"
    best_val = base_va
    if results["cur"][0] > best_val:
        best_name = "cur"; best_val = results["cur"][0]
    if results["ext"][0] > best_val:
        best_name = "ext"; best_val = results["ext"][0]

    # ---------------- final fit on the whole permitted pool ----------------
    if best_name == "ext":
        Xuse = feat_ext
        w2_new = w2_ext
        b2_new = b2_ext
    else:
        Xuse = feat_cur
        w2_new = w2
        b2_new = b2

    lam_f, Wg_f, gcv_f = gcv_lambda(Xuse, Y1h, [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0])
    Wf, bf = ridge_fit(Xuse, Y1h, lam_f)

    # blend a little of the original head for stability (convex combination)
    alpha = 0.0
    if best_name == "cur":
        alpha = 0.0
    # final logits check on the pool
    pred = np.argmax(Xuse @ Wf + bf, axis=1)
    train_acc = float((pred == y).mean())
    diag["chosen"] = best_name
    diag["final_lambda"] = lam_f
    diag["final_gcv"] = gcv_f
    diag["train_acc_after"] = train_acc
    diag["val_acc_estimate"] = best_val
    diag["head_weight_absmax"] = float(np.abs(Wf).max())
    diag["head_bias"] = [float(v) for v in bf]

    # ---------------- write out ----------------
    w2_out = np.asarray(w2_new, dtype=np.float32)
    b2_out = np.asarray(b2_new, dtype=np.float32)
    hw_out = np.asarray(Wf, dtype=np.float32)
    hb_out = np.asarray(bf, dtype=np.float32)

    for t in (w2_out, b2_out, hw_out, hb_out):
        t[np.abs(t) > 1000.0] = np.sign(t[np.abs(t) > 1000.0]) * 1000.0
        t[~np.isfinite(t)] = 0.0

    w["conv2.weight"] = w2_out
    w["conv2.bias"] = b2_out
    w["head.weight"] = hw_out
    w["head.bias"] = hb_out

    return {"weights": w, "diagnostics": diag}
