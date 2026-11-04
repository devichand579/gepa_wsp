import numpy as np
import math
from numpy.lib.stride_tricks import as_strided


def edit(w, f):
    # ---- pull feedback ----
    F = np.asarray(f["features"], dtype=np.float32)      # (N,64,16,16) == layer.out.input0
    y = np.asarray(f["y"], dtype=np.float32)             # (N,1,16,16)
    N = F.shape[0]
    H = W = 16
    C = 32                      # only channels 0..31 survive conv2/conv3 identity
    F = np.ascontiguousarray(F[:, :C])                   # (N,32,16,16)

    # zero-pad spatially by 1 (pad=1, stride=1)
    Fp = np.zeros((N, C, H + 2, W + 2), dtype=np.float32)
    Fp[:, :, 1:H + 1, 1:W + 1] = F

    # strided patch view: (N,16,16,C,3,3) -> per output pixel the (c,kr,kc) taps
    sN, sC, sH, sW = Fp.strides
    patches = as_strided(
        Fp,
        shape=(N, H, W, C, 3, 3),
        strides=(sN, sH, sW, sC, sH, sW),
        writeable=False,
    )

    D = C * 9 + 1
    G = np.zeros((D, D), dtype=np.float64)
    b = np.zeros(D, dtype=np.float64)
    yflat = y.reshape(N, H * W).astype(np.float64)

    chunk = 200
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        P = patches[s:e].reshape((e - s) * H * W, C * 9).astype(np.float64)
        yc = yflat[s:e].reshape(-1)
        G[:C * 9, :C * 9] += P.T @ P
        col = P.sum(axis=0)
        G[:C * 9, C * 9] += col
        G[C * 9, :C * 9] += col
        G[C * 9, C * 9] += P.shape[0]
        b[:C * 9] += P.T @ yc
        b[C * 9] += yc.sum()

    # tiny ridge for numerical stability (negligible with 5e5 samples / 289 params)
    lam = 1e-6
    G[:C * 9, :C * 9] += lam * np.eye(C * 9)
    sol = np.linalg.solve(G, b)

    out_w = np.zeros((1, 64, 3, 3), dtype=np.float32)
    out_w[0, :C] = sol[:C * 9].reshape(C, 3, 3).astype(np.float32)
    out_b = np.array([sol[C * 9]], dtype=np.float32)

    w["out.weight"] = out_w
    w["out.bias"] = out_b

    # ---- diagnostics ----
    pred = np.zeros((N, H * W), dtype=np.float64)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        P = patches[s:e].reshape((e - s) * H * W, C * 9).astype(np.float64)
        pred[s * H * W:e * H * W] = P @ sol[:C * 9] + sol[C * 9]
    mse = float(np.mean((pred - yflat.reshape(-1)) ** 2))

    diag = {
        "fit_mse": mse,
        "out_weight_norm": float(np.linalg.norm(out_w)),
        "out_bias": float(out_b[0]),
        "n_samples": int(N * H * W),
        "n_params": int(D),
    }
    return {"weights": w, "diagnostics": diag}
