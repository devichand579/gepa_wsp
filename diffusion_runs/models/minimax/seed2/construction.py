import numpy as np

def edit(w, f):
    # ---- gather permitted feedback -------------------------------------
    feats = np.asarray(f["features"], dtype=np.float64)      # [N,64,16,16] parent conv3 ReLU output
    y     = np.asarray(f["y"], dtype=np.float64)              # [N,1,16,16] target epsilon
    pred  = np.asarray(f["prediction"], dtype=np.float64)     # [N,1,16,16] current prediction
    t     = np.asarray(f["t"]).astype(np.int64).ravel()       # [N]
    N = feats.shape[0]

    # ---- design matrix: per-pixel conv3 features + bias ---------------
    X = feats.transpose(0, 2, 3, 1).reshape(N * 256, 64)      # [N*256, 64]
    Y = y.transpose(0, 2, 3, 1).reshape(N * 256, 1)          # [N*256, 1]
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # + bias column
    Y = Y - pred.reshape(N * 256, 1)                          # residual to correct

    # ---- ridge normal equations (closed form, no iterative optimizer) --
    XtX = X.T @ X
    XtY = X.T @ Y
    d = XtX.shape[0]
    tr = np.trace(XtX) / d
    lam = 1e-6 * max(tr, 1e-12)
    A = XtX + lam * np.eye(d)
    try:
        coef = np.linalg.solve(A, XtY)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]

    # ---- diagnostics: in-sample MSE of the closed-form fit -------------
    resid = X @ coef - Y
    mse_new = float(np.mean((resid + pred.reshape(N * 256, 1)) ** 2))
    mse_old = float(np.mean((Y + pred.reshape(N * 256, 1)) ** 2))

    # ---- write back into the fixed architecture ------------------------
    out_w = np.asarray(w["out.weight"], dtype=np.float64).copy()   # [1,64,3,3]
    out_b = np.asarray(w["out.bias"], dtype=np.float64).copy()     # [1]
    # conv3 output at (i,j) = sum_{c,dy,dx} W3[c,dy,dx] * F[c, i+dy-1, j+dx-1]
    # => prediction(i,j) = sum_c sum_{dy,dx} Wout[0,c,dy,dx] * F[c,i+dy-1,j+dx-1]
    # coefficient of F[c, i+dy-1, j+dx-1] is Wout[0,c,dy,dx]
    # so Wout[0,c,dy,dx] = coef[c, dy, dx] with dy,dx in {0,1,2} -> kernel index (dy-1, dx-1)
    Wc = coef[:64].reshape(64, 3, 3)
    # our design used feature order (c, dy, dx) with dy,dx in 0..2 -> map to kernel offsets
    # design row (i,j) uses F[c, i+dy-1, j+dx-1] for dy,dx in 0..2
    # kernel index (ky,kx) means F[c, i+ky-1, j+kx-1]  => ky = dy, kx = dx
    out_w[0] = np.transpose(Wc, (0, 1, 2))  # keep [c,ky,kx]
    out_b[0] = coef[64, 0]

    # safety: finite, bounded, float32
    out_w = np.nan_to_num(out_w, nan=0.0, posinf=0.0, neginf=0.0)
    out_b = np.nan_to_num(out_b, nan=0.0, posinf=0.0, neginf=0.0)
    out_w = np.clip(out_w, -1000.0, 1000.0).astype(np.float32)
    out_b = np.clip(out_b, -1000.0, 1000.0).astype(np.float32)

    w = dict(w)
    w["out.weight"] = out_w
    w["out.bias"] = out_b

    # ---- per-timestep diagnostics --------------------------------------
    # recompute predictions with the new out layer from cached features
    # (valid since out is the last layer and features are the parent conv3 output)
    F = feats.transpose(0, 2, 3, 1)                      # [N,16,16,64]
    P = np.zeros((N, 16, 16), dtype=np.float64)
    for ky in range(3):
        for kx in range(3):
            # F shifted so that P[i,j] += sum_c W[c,ky,kx] * F[c, i+ky-1, j+kx-1]
            sh = np.zeros_like(F)
            ys = slice(max(0, 1 - ky), 16 - ky)
            yd = slice(max(0, ky - 1), 16 - (ky - 1))
            xs = slice(max(0, 1 - kx), 16 - kx)
            xd = slice(max(0, kx - 1), 16 - (kx - 1))
            sh[:, ys, xs, :] = F[:, yd, xd, :]
            P += np.tensordot(sh, out_w[0, :, ky, kx].astype(np.float64), axes=([3], [0]))
    P = P + float(out_b[0])
    yv = y.transpose(0, 2, 3, 1)[..., 0]
    pv = pred.transpose(0, 2, 3, 1)[..., 0]
    per_t = {}
    for tt in np.unique(t):
        m = (t == tt)
        per_t[int(tt)] = {
            "mse_old": float(np.mean((yv[m] - pv[m]) ** 2)),
            "mse_new": float(np.mean((yv[m] - P[m]) ** 2)),
        }

    diagnostics = {
        "mse_old": mse_old,
        "mse_new_insample": mse_new,
        "ridge_lambda": float(lam),
        "n_design_rows": int(X.shape[0]),
        "coef_l2": float(np.linalg.norm(coef)),
        "out_w_abs_max": float(np.max(np.abs(out_w))),
        "per_timestep": per_t,
    }
    return {"weights": w, "diagnostics": diagnostics}
