import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def edit(w, f):
    # ---- pull arrays -----------------------------------------------------
    feats = np.asarray(f["features"], dtype=np.float32)      # (N,64,16,16)
    y     = np.asarray(f["y"],        dtype=np.float32)      # (N,1,16,16)
    N, C, H, W = feats.shape

    # ---- build padded feature volume for 3x3 conv readout ----------------
    # pad H,W by 1 (zero pad) -> (N,C,H+2,W+2)
    pad = np.zeros((N, C, H + 2, W + 2), dtype=np.float32)
    pad[:, :, 1:H + 1, 1:W + 1] = feats

    # patches: (N, H, W, C, 3, 3) via sliding window on padded volume
    win = sliding_window_view(pad, (3, 3), axis=(2, 3))      # (N,C,H,W,3,3)
    # reorder to (N,H,W,C,3,3)
    win = np.transpose(win, (0, 2, 3, 1, 4, 5))
    X = win.reshape(N * H * W, C * 9).astype(np.float64)     # (N*H*W, 576)
    Y = y.reshape(N * H * W).astype(np.float64)              # (N*H*W,)

    # ---- normal equations with bias column -------------------------------
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    G = Xb.T @ Xb                                            # (577,577)
    rhs = Xb.T @ Y                                           # (577,)

    # tiny ridge for numerical stability (does not bias much)
    reg = 1e-6 * np.trace(G) / G.shape[0]
    G[np.diag_indices_from(G)] += reg

    try:
        sol = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(G, rhs, rcond=None)[0]

    Wout = sol[:C * 9].reshape(C, 3, 3).astype(np.float32)   # (64,3,3)
    bout = np.float32(sol[C * 9])

    # ---- write back into the 1x64x3x3 out weight -------------------------
    w["out.weight"] = Wout.reshape(1, C, 3, 3).astype(np.float32)
    w["out.bias"]   = np.array([bout], dtype=np.float32)

    # ---- diagnostics ------------------------------------------------------
    pred = Xb @ sol
    mse_before = float(np.mean((Y - np.asarray(f["prediction"], dtype=np.float64).reshape(-1)) ** 2))
    mse_after  = float(np.mean((Y - pred) ** 2))
    diag = {
        "readout_mse_before": mse_before,
        "readout_mse_after": mse_after,
        "out_weight_norm": float(np.linalg.norm(Wout)),
        "out_bias": float(bout),
        "ridge": float(reg),
        "n_equations": int(Xb.shape[0]),
    }
    return {"weights": w, "diagnostics": diag}
