import numpy as np
import math

def edit(w, f):
    # ---------- copy weights ----------
    W = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}

    F = np.asarray(f['features'], dtype=np.float32)      # (2000,64,16,16)
    Y = np.asarray(f['y'], dtype=np.float32)             # (2000,1,16,16)
    N, C, H, Wd = F.shape

    # ---------- consistency check on the paired batch ----------
    ids = np.asarray(f['batch.ids']).astype(np.int64).ravel()
    inp0 = np.asarray(f['layer.out.input0'], dtype=np.float32)  # (128,64,16,16)
    try:
        chk = np.mean((F[ids] - inp0) ** 2)
    except Exception:
        chk = -1.0
    features_are_out_inputs = (chk >= 0.0 and chk < 1e-3)

    # ---------- current MSE on the full 2000 ----------
    pred = np.asarray(f['prediction'], dtype=np.float32)
    cur_mse = float(np.mean((pred - Y) ** 2))

    # ---------- build design matrix in chunks and accumulate normal eqs ----------
    # design row for pixel (n,i,j): [ for di in 0..2, dj in 0..2, c in 0..63: Fpad[n,c,i+di,j+dj] ] then bias
    K = 3
    P = C * K * K          # 576
    nparam = P + 1         # + bias

    A = np.zeros((nparam, nparam), dtype=np.float64)
    b = np.zeros((nparam,), dtype=np.float64)

    chunk = 100
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        Fc = F[s:e]                                        # (n,64,16,16)
        Yc = Y[s:e].reshape(e - s, -1)                     # (n, 256)
        n = e - s
        Fpad = np.zeros((n, C, H + 2, Wd + 2), dtype=np.float32)
        Fpad[:, :, 1:1 + H, 1:1 + Wd] = Fc
        # collect the 9 shifted versions, each (n,64,16,16) -> (n,16,16,64)
        blocks = []
        for di in range(K):
            for dj in range(K):
                g = Fpad[:, :, di:di + H, dj:dj + Wd]      # (n,64,16,16)
                blocks.append(np.transpose(g, (0, 2, 3, 1)))  # (n,16,16,64)
        D = np.concatenate(blocks, axis=3)                 # (n,16,16,576)
        D = D.reshape(n * H * Wd, P).astype(np.float32)    # (n*256, 576)
        # add bias column
        Db = np.concatenate([D, np.ones((D.shape[0], 1), dtype=np.float32)], axis=1)
        yv = Yc.reshape(-1).astype(np.float32)
        A += (Db.T.astype(np.float64) @ Db.astype(np.float64))
        b += (Db.T.astype(np.float64) @ yv.astype(np.float64))

    # ---------- ridge solve ----------
    lam = 1e-6 * (np.trace(A) / max(nparam, 1))
    A_reg = A + lam * np.eye(nparam)
    try:
        sol = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A_reg, b, rcond=None)[0]
    sol = sol.astype(np.float32)

    new_out_w = sol[:P].reshape(K, K, C).transpose(2, 0, 1).reshape(1, C, K, K).astype(np.float32)
    new_out_b = np.array([sol[P]], dtype=np.float32)

    # ---------- evaluate the refit on the full 2000 ----------
    def eval_mse(out_w, out_b):
        tot = 0.0
        cnt = 0
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            Fc = F[s:e]
            n = e - s
            Fpad = np.zeros((n, C, H + 2, Wd + 2), dtype=np.float32)
            Fpad[:, :, 1:1 + H, 1:1 + Wd] = Fc
            blocks = []
            for di in range(K):
                for dj in range(K):
                    g = Fpad[:, :, di:di + H, dj:dj + Wd]
                    blocks.append(np.transpose(g, (0, 2, 3, 1)))
            D = np.concatenate(blocks, axis=3).reshape(n * H * Wd, P)
            yv = Y[s:e].reshape(-1)
            pv = D @ out_w.reshape(-1) + out_b[0]
            tot += float(np.sum((pv - yv) ** 2))
            cnt += pv.size
        return tot / max(cnt, 1)

    refit_mse = eval_mse(new_out_w, new_out_b)

    diag = {
        'features_match_out_input': bool(features_are_out_inputs),
        'features_check_mse': float(chk),
        'current_mse': cur_mse,
        'refit_mse': float(refit_mse),
        'n_train': int(N),
        'n_params': int(nparam),
        'ridge': float(lam),
    }

    if refit_mse < cur_mse:
        W['out.weight'] = new_out_w
        W['out.bias'] = new_out_b
        diag['chosen'] = 'refit'
        diag['chosen_mse'] = float(refit_mse)
    else:
        diag['chosen'] = 'current'
        diag['chosen_mse'] = float(cur_mse)

    # ---------- sanitize ----------
    for k in list(W.keys()):
        arr = np.asarray(W[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1000.0, 1000.0)
        W[k] = arr.astype(np.float32)

    return {"weights": W, "diagnostics": diag}
