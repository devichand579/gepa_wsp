import numpy as np


def _conv2d(x, W, b):
    """Cross-correlation conv2d, pad=1, stride 1. x:(N,C,H,W) W:(O,C,3,3) b:(O,)"""
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float32)
    xp[:, :, 1:-1, 1:-1] = x
    out = np.zeros((N, O, H, Wd), dtype=np.float32)
    for kr in range(3):
        for kc in range(3):
            out += np.tensordot(xp[:, :, kr:kr + H, kc:kc + Wd],
                                W[:, :, kr, kc], axes=([1], [1])).transpose(0, 3, 1, 2)
    out += b.reshape(1, O, 1, 1)
    return out


def _forward(w, x, t):
    """x:(N,1,16,16) t:(N,) -> (N,1,16,16) prediction, matching the stated forward."""
    N = x.shape[0]
    tb = (np.asarray(t, dtype=np.float32) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x.astype(np.float32), np.broadcast_to(tb, (N, 1, 16, 16))], axis=1)
    h = np.maximum(_conv2d(inp, w["conv1.weight"], w["conv1.bias"]), 0.0)
    h = np.maximum(_conv2d(h, w["conv2.weight"], w["conv2.bias"]), 0.0)
    h = np.maximum(_conv2d(h, w["conv3.weight"], w["conv3.bias"]), 0.0)
    return _conv2d(h, w["out.weight"], w["out.bias"])


def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}

    feats = np.asarray(f["features"], dtype=np.float32)      # (N,64,16,16)
    y = np.asarray(f["y"], dtype=np.float32)                 # (N,1,16,16)
    N, C, H, Wd = feats.shape
    P = H * Wd

    # ---- design matrix: rows = (example, output pixel), cols = 3x3 patch of 64 channels (+ bias)
    # patch extraction with zero padding (matches conv pad=1)
    Fp = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float32)
    Fp[:, :, 1:-1, 1:-1] = feats
    D = np.empty((N * P, C * 9 + 1), dtype=np.float32)
    k = 0
    for kr in range(3):
        for kc in range(3):
            # (N,C,H,W) -> (N,H,W,C) flattened per pixel
            blk = Fp[:, :, kr:kr + H, kc:kc + Wd].transpose(0, 2, 3, 1).reshape(N * P, C)
            D[:, k * C:(k + 1) * C] = blk
            k += 1
    D[:, -1] = 1.0
    T = y.transpose(0, 2, 3, 1).reshape(N * P, 1)

    # ---- ridge regression (closed form)
    lam = 1e-3
    G = D.T @ D
    reg = lam * float(np.trace(G)) / G.shape[0]
    G = G + reg * np.eye(G.shape[0], dtype=np.float32)
    b = D.T @ T
    coef = np.linalg.solve(G, b).astype(np.float32)   # (577,1)

    ow = coef[:C * 9].reshape(1, C, 3, 3).copy()      # [out,in,kr,kc]
    ob = coef[-1].reshape(1).copy()

    # ---- verify with an independent forward pass (same convention as evaluator)
    w["out.weight"] = ow
    w["out.bias"] = ob
    pred = _forward(w, np.asarray(f["x"], dtype=np.float32), np.asarray(f["t"]))
    mse = float(np.mean((pred - y) ** 2))
    mse_old = float(np.mean((np.asarray(f["prediction"], dtype=np.float32) - y) ** 2))

    # ---- safety: finite, bounded, shape preserved
    for k2 in w:
        a = w[k2]
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(a, -1000.0, 1000.0, out=a)
        w[k2] = a.astype(np.float32)

    return {"weights": w,
            "diagnostics": {"mse_new_insample": mse, "mse_old": mse_old,
                            "out_w_absmean": float(np.mean(np.abs(ow))),
                            "out_b": float(ob[0]),
                            "ridge": float(reg)}}
