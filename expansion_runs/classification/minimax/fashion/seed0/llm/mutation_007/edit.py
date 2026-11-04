import numpy as np

def edit(w, f):
    X = np.asarray(f["x"], dtype=np.float64)          # (N,1,28,28)
    y = np.asarray(f["y"], dtype=np.int64)
    feats_cached = np.asarray(f["features"], dtype=np.float64)
    N = X.shape[0]

    # ---------------- forward pass in numpy (for verification + augmentation) ----------
    def conv2d(x, wt, b, pad=1):
        # x: (N,C,H,W) ; wt: (O,C,kh,kw) ; cross-correlation, stride 1
        N, C, H, Wd = x.shape
        O, _, kh, kw = wt.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        Hp, Wp = xp.shape[2], xp.shape[3]
        out = np.zeros((N, O, Hp - kh + 1, Wp - kw + 1), dtype=np.float64)
        for i in range(kh):
            for j in range(kw):
                patch = xp[:, :, i:i + Hp - kh + 1, j:j + Wp - kw + 1]   # (N,C,H,W)
                out += np.tensordot(patch, wt[:, :, i, j], axes=([1], [1])).transpose(0, 3, 1, 2)
        out += b.reshape(1, O, 1, 1)
        return out

    def maxpool2(x):
        N, C, H, Wd = x.shape
        H2, W2 = H // 2, Wd // 2
        r = x[:, :, :H2 * 2, :W2 * 2].reshape(N, C, H2, 2, W2, 2)
        return r.max(axis=(3, 5))

    def forward_feats(xin):
        c1 = np.maximum(conv2d(xin, w["conv1.weight"], w["conv1.bias"]), 0.0)
        p1 = maxpool2(c1)
        c2 = np.maximum(conv2d(p1, w["conv2.weight"], w["conv2.bias"]), 0.0)
        p2 = maxpool2(c2)
        return p2.reshape(p2.shape[0], -1)

    F = forward_feats(X)
    err = float(np.max(np.abs(F - feats_cached)))
    verified = err < 1e-3

    # ---------------- design matrix (original + shifted copies) ----------------
    def shift_images(xin, dy, dx):
        out = np.zeros_like(xin)
        ys = slice(max(0, dy), xin.shape[2] + min(0, dy))
        yd = slice(max(0, -dy), xin.shape[2] + min(0, -dy))
        xs = slice(max(0, dx), xin.shape[3] + min(0, dx))
        xd = slice(max(0, -dx), xin.shape[3] + min(0, -dx))
        out[:, :, yd, xd] = xin[:, :, ys, xs]
        return out

    shifts = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]
    if verified:
        blocks = [F]
        for (dy, dx) in shifts[1:]:
            blocks.append(forward_feats(shift_images(X, dy, dx)))
        A = np.concatenate(blocks, axis=0)
        grp = np.concatenate([np.full(N, i, dtype=np.int64) for i in range(len(blocks))])
    else:
        A = feats_cached.copy()
        grp = np.arange(N, dtype=np.int64)

    # standardize features (affine -> foldable into the linear head)
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z = (A - mu) / sd
    Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)   # intercept column
    d = Z.shape[1]

    Y1 = np.zeros((A.shape[0], 10))
    Y1[np.arange(A.shape[0]), np.tile(y, len(blocks))] = 1.0

    # ---------------- closed-form ridge + closed-form group LOO ----------------
    G = Z.T @ Z
    b = Z.T @ Y1
    evals, evecs = np.linalg.eigh(G)
    evals = np.maximum(evals, 0.0)

    Zorig = Z[:N]
    Y1orig = Y1[:N]
    groups = [np.where(grp == i)[0] for i in range(N)]

    lam_grid = [1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 1e4]
    best = None
    loo_table = []
    for lam in lam_grid:
        inv = 1.0 / (evals + lam)
        Minv = (evecs * inv) @ evecs.T
        # full-data fit
        Wfull = Minv @ b
        pred_full = Z @ Wfull
        acc_full = float(np.mean(pred_full[:N].argmax(1) == y))
        # group LOO on the original rows
        correct = 0
        for i in range(N):
            Gidx = groups[i]
            Xg = Z[Gidx]
            yg = Y1[Gidx]
            Mg_inv = Minv
            u = Xg @ Mg_inv                      # (g,d)
            v = Xg @ Mg_inv @ Xg.T               # (g,g)
            s = Xg @ Mg_inv @ b                  # (g,10)
            try:
                Ainv = np.linalg.inv(np.eye(len(Gidx)) - v)
            except np.linalg.LinAlgError:
                Ainv = np.linalg.pinv(np.eye(len(Gidx)) - v)
            # leave-group-out prediction for original row i
            coef = u @ Ainv                       # (g,d)
            pred_i = Zorig[i] @ Wfull - (Zorig[i] @ Mg_inv @ Xg.T) @ yg \
                     + (coef @ (b - Xg.T @ yg))
            correct += int(np.argmax(pred_i) == y[i])
        acc_loo = correct / float(N)
        loo_table.append((lam, acc_full, acc_loo))
        if best is None or acc_loo > best[1]:
            best = (lam, acc_loo, acc_full, Wfull)

    lam, acc_loo, acc_full, Wfull = best

    # ---------------- fold standardization back into head weights ----------------
    Wz = Wfull[:d - 1, :]        # (1568,10)
    bz = Wfull[d - 1, :]         # (10,)
    Wh = (Wz / sd[:, None]).T   # (10,1568)
    bh = bz - Wh @ mu

    # ---------------- safety: never worse than the current head on the pool -------
    cur_logits = feats_cached @ w["head.weight"].T + w["head.bias"]
    cur_acc = float(np.mean(cur_logits.argmax(1) == y))
    new_logits = feats_cached @ Wh.T + bh
    new_acc = float(np.mean(new_logits.argmax(1) == y))

    if new_acc < cur_acc - 0.005:
        # fall back to a conservative ridge on cached features only (no augmentation)
        Zc = np.concatenate([(feats_cached - mu) / sd, np.ones((N, 1))], axis=1)
        Gc = Zc.T @ Zc
        bc = Zc.T @ Y1[:N]
        ec, vc = np.linalg.eigh(Gc)
        ec = np.maximum(ec, 0.0)
        lam2 = lam
        W2 = (vc * (1.0 / (ec + lam2))) @ vc.T @ bc
        Wz2 = W2[:d - 1, :]
        bz2 = W2[d - 1, :]
        Wh = (Wz2 / sd[:, None]).T
        bh = bz2 - Wh @ mu
        new_logits = feats_cached @ Wh.T + bh
        new_acc = float(np.mean(new_logits.argmax(1) == y))
        acc_loo = float("nan")

    # ---------------- write out ----------------
    w["head.weight"] = np.clip(Wh, -1000.0, 1000.0).astype(np.float32)
    w["head.bias"] = np.clip(bh, -1000.0, 1000.0).astype(np.float32)

    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def ce(p, t):
        return float(-np.mean(np.log(np.maximum(p[np.arange(len(t)), t], 1e-12))))

    p_new = softmax(new_logits)
    p_cur = softmax(cur_logits)
    diag = {
        "conv_verified": bool(verified),
        "conv_max_abs_err": err,
        "n_aug_blocks": int(len(blocks)) if verified else 1,
        "n_rows": int(A.shape[0]),
        "lambda": float(lam),
        "loo_table": [[float(a), float(b), float(c)] for a, b, c in loo_table],
        "acc_current_head_pool": cur_acc,
        "acc_new_head_pool": new_acc,
        "acc_loo_group": float(acc_loo),
        "ce_new_pool": ce(p_new, y),
        "ce_current_pool": ce(p_cur, y),
        "head_w_std": float(np.std(Wh)),
        "head_w_absmax": float(np.max(np.abs(Wh))),
        "head_b": [float(v) for v in bh],
        "feature_zero_frac": float(np.mean(feats_cached == 0.0)),
    }
    return {"weights": w, "diagnostics": diag}
