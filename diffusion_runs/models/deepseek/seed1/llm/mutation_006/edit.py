import numpy as np

def edit(w, f):
    F = np.asarray(f['features'], dtype=np.float32)   # (N,64,16,16) final hidden = out.input
    Y = np.asarray(f['y'], dtype=np.float32)          # (N,1,16,16)
    N, C, H, Wd = F.shape
    P = C * 9                                          # 576 conv taps
    A = np.zeros((P + 1, P + 1), dtype=np.float64)
    rhs = np.zeros(P + 1, dtype=np.float64)
    yy = 0.0

    chunk = 250
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        n = e - s
        Fc = F[s:e]
        Yc = Y[s:e]
        Fpad = np.zeros((n, C, H + 2, Wd + 2), dtype=np.float32)
        Fpad[:, :, 1:H + 1, 1:Wd + 1] = Fc
        patches = np.empty((n, C, 9, H, Wd), dtype=np.float32)
        k = 0
        for a in range(3):
            for b in range(3):
                patches[:, :, k, :, :] = Fpad[:, :, a:a + H, b:b + Wd]
                k += 1
        # X: (n*H*Wd, C*9) with column = c*9 + (a*3+b)
        X = patches.transpose(0, 3, 4, 1, 2).reshape(n * H * Wd, C * 9)
        yv = Yc.transpose(0, 2, 3, 1).reshape(n * H * Wd)
        # accumulate normal equations (float32 gemm -> float64 accumulate)
        A[:P, :P] += (X.T @ X).astype(np.float64)
        Xs = X.sum(axis=0).astype(np.float64)
        A[:P, P] += Xs
        A[P, :P] += Xs
        A[P, P] += float(X.shape[0])
        rhs[:P] += (X.T @ yv).astype(np.float64)
        rhs[P] += float(yv.sum())
        yy += float((yv.astype(np.float64) ** 2).sum())

    # small ridge for numerical stability (heavily overdetermined -> negligible bias)
    lam = 1e-3
    A_reg = A.copy()
    A_reg[:P, :P] += lam * np.eye(P, dtype=np.float64)
    A_reg[P, P] += lam
    try:
        theta = np.linalg.solve(A_reg, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A_reg, rhs, rcond=None)[0]

    theta = theta.astype(np.float32)
    out_w = theta[:P].reshape(1, C, 3, 3).astype(np.float32)
    out_b = theta[P:P + 1].astype(np.float32)

    # sanity: predicted training MSE with new readout
    pred_mse = float((yy - 2.0 * float(theta.astype(np.float64) @ rhs)
                      + float(theta.astype(np.float64) @ A @ theta.astype(np.float64)))
                     / (N * H * Wd))

    w['out.weight'] = np.clip(out_w, -1000.0, 1000.0).astype(np.float32)
    w['out.bias'] = np.clip(out_b, -1000.0, 1000.0).astype(np.float32)

    # ensure everything finite
    for kk in list(w.keys()):
        v = np.asarray(w[kk])
        if v.dtype.kind == 'f':
            w[kk] = np.nan_to_num(v, nan=0.0, posinf=1000.0, neginf=-1000.0).astype(np.float32)

    diag = {
        "out_weight_norm": float(np.linalg.norm(w['out.weight'])),
        "out_bias": float(w['out.bias'][0]),
        "ridge": lam,
        "train_mse_new_readout": pred_mse,
        "solved": True,
    }
    return {"weights": w, "diagnostics": diag}
