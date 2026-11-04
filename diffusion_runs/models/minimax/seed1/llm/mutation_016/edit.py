import numpy as np

def edit(w, f):
    F = np.asarray(f["features"], dtype=np.float64)      # [N,64,16,16] = input of 'out'
    Y = np.asarray(f["y"], dtype=np.float64)[:, 0]        # [N,16,16]
    N, C, H, Wd = F.shape
    P = H * Wd

    # ---- design matrix: rows = (example, pixel), cols = 576 patch taps ----
    # patch ordering: col = c*9 + dy*3 + dx  (matches weight[out,in,kr,kc] flattening)
    X = np.empty((N * P, C * 9), dtype=np.float64)
    for c in range(C):
        Fc = F[:, c]                                     # [N,H,W]
        for dy in range(3):
            for dx in range(3):
                # shifted feature: value used at output pixel (i,j) is F[:,c,i+dy-1,j+dx-1]
                sh = np.zeros_like(Fc)
                ys = slice(0, H - 2) if dy == 0 else (slice(1, H - 1) if dy == 1 else slice(2, H))
                yd = slice(2, H) if dy == 0 else (slice(1, H - 1) if dy == 1 else slice(0, H - 2))
                xs = slice(0, Wd - 2) if dx == 0 else (slice(1, Wd - 1) if dx == 1 else slice(2, Wd))
                xd = slice(2, Wd) if dx == 0 else (slice(1, Wd - 1) if dx == 1 else slice(0, Wd - 2))
                sh[:, yd, xd] = Fc[:, ys, xs]
                X[:, c * 9 + dy * 3 + dx] = sh.reshape(-1)

    Yv = Y.reshape(-1)

    # ---- ridge normal equations (closed form) ----
    lam = 1e-3
    G = X.T @ X
    b = X.T @ Yv
    G[np.diag_indices_from(G)] += lam * (np.trace(G) / G.shape[0] + 1e-12)
    sol = np.linalg.solve(G, b)

    # ---- write back ----
    Wout = np.asarray(w["out.weight"], dtype=np.float64).copy()
    Wout[0] = sol.reshape(C, 3, 3)
    bmean = float(np.mean(Yv))
    bout = np.array([bmean], dtype=np.float64)

    # ---- diagnostics: in-sample MSE of the new readout ----
    pred = (X @ sol + bmean).reshape(N, H, Wd)
    mse = float(np.mean((pred - Y) ** 2))

    w["out.weight"] = Wout.astype(np.float32)
    w["out.bias"] = bout.astype(np.float32)

    for k in w:
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(arr, -1000.0, 1000.0, out=arr)
        w[k] = arr

    return {"weights": w,
            "diagnostics": {"mse_new_insample": round(mse, 5),
                            "mse_old": 1.0058,
                            "out_w_absmean": float(np.abs(Wout).mean()),
                            "out_b": float(bmean)}}
