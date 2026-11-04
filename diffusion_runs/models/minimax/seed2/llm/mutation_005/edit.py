import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def pad1(a):
        # a: (N,C,H,W) -> zero padded (N,C,H+2,W+2)
        N, C, H, Wd = a.shape
        p = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float32)
        p[:, :, 1:H + 1, 1:Wd + 1] = a
        return p

    def conv3x3(a, Wt):
        # a: (N,C,H,W), Wt: (O,C,3,3) cross-correlation, pad 1
        N, C, H, Wd = a.shape
        O = Wt.shape[0]
        ap = pad1(a)
        out = np.zeros((N, O, H, Wd), dtype=np.float32)
        for dr in range(3):
            for dc in range(3):
                # (N,C,H,W) slice
                sl = ap[:, :, dr:dr + H, dc:dc + Wd]
                # Wt[:, :, dr, dc] -> (O,C)
                out += np.tensordot(sl, Wt[:, :, dr, dc].astype(np.float32),
                                    axes=([1], [1])).transpose(0, 3, 1, 2)
        return out

    def mse(pred, tgt):
        return float(np.mean((pred - tgt) ** 2))

    # ---------------- data ----------------
    feats = np.asarray(f["features"], dtype=np.float32)          # (N,64,16,16)
    y = np.asarray(f["y"], dtype=np.float32)                      # (N,1,16,16)
    N = feats.shape[0]
    C = feats.shape[1]
    H, Wd = feats.shape[2], feats.shape[3]

    # design matrix rows: (example, pixel) ; columns: 9*64 kernel taps + bias
    # Build column blocks lazily: for tap (dr,dc) and channel c -> shifted feature
    fp = pad1(feats)                                              # (N,64,18,18)

    # Gram matrix (577 x 577) and rhs (577) accumulated in chunks
    n_tap = 9
    P = n_tap * C + 1
    G = np.zeros((P, P), dtype=np.float64)
    rhs = np.zeros((P,), dtype=np.float64)
    yy = 0.0

    chunk = 256
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        n = e - s
        # shifted features for this chunk: (n, 9, C, H, W)
        sh = np.empty((n, n_tap, C, H, Wd), dtype=np.float32)
        for dr in range(3):
            for dc in range(3):
                sh[:, dr * 3 + dc] = fp[s:e, :, dr:dr + H, dc:dc + Wd]
        # flatten to (n*H*W, 9*C) with tap-major ordering
        A = np.ascontiguousarray(
            sh.transpose(0, 3, 4, 1, 2).reshape(n * H * Wd, n_tap * C)
        ).astype(np.float64)
        b = np.ones((n * H * Wd, 1), dtype=np.float64)
        yt = y[s:e].reshape(n * H * Wd, 1).astype(np.float64)
        G += A.T @ A
        rhs += A.T @ yt
        yy += float(np.sum(yt * yt))
        del A, b, yt, sh

    # ---------------- ridge solve ----------------
    diag = np.diag(G).copy()
    scale = float(np.mean(diag[:n_tap * C]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    def solve(lam):
        Gr = G.copy()
        idx = np.arange(P)
        Gr[idx, idx] += lam * scale
        try:
            sol = np.linalg.solve(Gr, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(Gr, rhs, rcond=None)[0]
        return sol

    def mse_of(sol):
        # residual = yy - 2 sol.rhs + sol' G sol
        return (yy - 2.0 * float(sol @ rhs) + float(sol @ (G @ sol))) / (N * H * Wd)

    lam_grid = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3,
                1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
    best = None
    for lam in lam_grid:
        sol = solve(lam)
        m = mse_of(sol)
        if np.isfinite(m) and (best is None or m < best[1]):
            best = (sol, m, lam)
    sol, mse_ls, lam = best

    # ---------------- write back ----------------
    out_w = w["out.weight"]
    new_out_w = np.zeros_like(out_w)
    for dr in range(3):
        for dc in range(3):
            new_out_w[0, :, dr, dc] = sol[dr * 3 + dc * C: dr * 3 + dc * C + C]
    new_out_bias = np.array([sol[-1]], dtype=np.float32)

    # safety: finite, magnitude bounded
    new_out_w = np.nan_to_num(new_out_w, nan=0.0, posinf=0.0, neginf=0.0)
    new_out_bias = np.nan_to_num(new_out_bias, nan=0.0, posinf=0.0, neginf=0.0)
    new_out_w = np.clip(new_out_w, -1000.0, 1000.0)
    new_out_bias = np.clip(new_out_bias, -1000.0, 1000.0)

    # ---------------- verify with actual forward pass on the batch ----------------
    def forward(x, t, W):
        Nn = x.shape[0]
        tb = (t.astype(np.float32) / 31.0).reshape(Nn, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (Nn, 1, H, Wd))], axis=1)
        h1 = np.maximum(conv3x3(inp, W["conv1.weight"]) + W["conv1.bias"][None, :, None, None], 0)
        h2 = np.maximum(conv3x3(h1, W["conv2.weight"]) + W["conv2.bias"][None, :, None, None], 0)
        h3 = np.maximum(conv3x3(h2, W["conv3.weight"]) + W["conv3.bias"][None, :, None, None], 0)
        return conv3x3(h3, new_out_w) + new_out_bias[None, :, None, None]

    w2 = {k: v.copy() for k, v in w.items()}
    w2["out.weight"] = new_out_w.astype(np.float32)
    w2["out.bias"] = new_out_bias.astype(np.float32)

    diag = {"mse_train_closed_form": mse_ls, "ridge_lambda": lam}

    try:
        bx = np.asarray(f["batch.x"], dtype=np.float32)
        bt = np.asarray(f["batch.t"])
        by = np.asarray(f["batch.y"], dtype=np.float32)
        pred_old = np.asarray(f["layer.out.output"], dtype=np.float32)
        pred_new = forward(bx, bt, w2)
        diag["batch_mse_old"] = mse(pred_old, by)
        diag["batch_mse_new"] = mse(pred_new, by)
    except Exception as e:
        diag["batch_check_error"] = str(e)

    # also full-train-set check using cached features (exact, since out is last layer)
    try:
        pred_tr = conv3x3(feats, new_out_w) + new_out_bias[None, :, None, None]
        diag["train_mse_new_cached"] = mse(pred_tr, y)
        diag["train_mse_old_cached"] = mse(
            conv3x3(feats, out_w.astype(np.float32)) + out_w.newaxis if False else
            conv3x3(feats, out_w.astype(np.float32)) + np.asarray(w["out.bias"], dtype=np.float32)[None, :, None, None],
            y)
    except Exception as e:
        diag["train_check_error"] = str(e)

    diag["out_weight_abs_max"] = float(np.max(np.abs(new_out_w)))
    diag["out_bias"] = float(new_out_bias[0])

    return {"weights": w2, "diagnostics": diag}
