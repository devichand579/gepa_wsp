import numpy as np

def edit(w, f):
    # --- unpack ---
    feats = np.asarray(f["features"], dtype=np.float64)   # (2000,64,16,16)
    y     = np.asarray(f["y"],        dtype=np.float64)   # (2000,1,16,16)
    N, C, H, W = feats.shape
    assert C == 64 and H == 16 and W == 16

    out_w = np.asarray(w["out.weight"], dtype=np.float64)  # (1,64,3,3)
    out_b = np.asarray(w["out.bias"],   dtype=np.float64)  # (1,)

    # --- current training MSE on the cached features ---
    def conv_out(feat, ww, bb):
        # feat: (N,64,16,16), ww: (1,64,3,3) -> (N,1,16,16), zero-pad 1
        pad = np.zeros((N, C, H + 2, W + 2), dtype=np.float64)
        pad[:, :, 1:-1, 1:-1] = feat
        out = np.zeros((N, 1, H, W), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                patch = pad[:, :, i:i + H, j:j + W]        # (N,64,16,16)
                out += np.einsum('nchw,oc->nohw', patch, ww[:, :, i, j])
        return out + bb.reshape(1, 1, 1, 1)

    cur_pred = conv_out(feats, out_w, out_b)
    cur_mse = float(np.mean((cur_pred - y) ** 2))

    # --- build design matrix in chunks and accumulate Gram / rhs ---
    # Each output pixel uses the 3x3 neighbourhood of all 64 channels -> 576 feats + bias
    P = 64 * 9
    D = P + 1
    G = np.zeros((D, D), dtype=np.float64)
    rhs = np.zeros(D, dtype=np.float64)
    chunk = 100
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        n = e - s
        pad = np.zeros((n, C, H + 2, W + 2), dtype=np.float64)
        pad[:, :, 1:-1, 1:-1] = feats[s:e]
        # columns: for each (ci, di, dj) -> index ci*9 + di*3 + dj
        cols = []
        for ci in range(C):
            for di in range(3):
                for dj in range(3):
                    cols.append(pad[:, ci, di:di + H, dj:dj + W].reshape(n * H * W))
        X = np.stack(cols, axis=1)                       # (n*H*W, 576)
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # add bias
        t = y[s:e].reshape(n * H * W)
        G += X.T @ X
        rhs += X.T @ t

    # --- ridge solve (tiny lambda for numerical stability) ---
    lam = 1e-6 * (np.trace(G) / D + 1e-12)
    G_reg = G + lam * np.eye(D)
    try:
        sol = np.linalg.solve(G_reg, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(G_reg, rhs, rcond=None)[0]

    new_w = sol[:P].reshape(1, C, 3, 3)
    new_b = np.array([sol[P]])

    new_pred = conv_out(feats, new_w, new_b)
    new_mse = float(np.mean((new_pred - y) ** 2))

    diag = {
        "current_mse": cur_mse,
        "refit_mse": new_mse,
        "chosen": "refit" if new_mse < cur_mse else "current",
        "n_train": int(N),
        "n_params": int(D),
    }

    if new_mse < cur_mse:
        w["out.weight"] = new_w.astype(np.float32)
        w["out.bias"]   = new_b.astype(np.float32)

    # --- safety: keep everything finite and bounded ---
    for k, v in list(w.items()):
        a = np.asarray(v, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=1000.0, neginf=-1000.0)
        np.clip(a, -1000.0, 1000.0, out=a)
        w[k] = a

    return {"weights": w, "diagnostics": diag}
