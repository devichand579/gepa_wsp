import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float64)          # [N,1,16,16]
    y = np.asarray(f["y"], dtype=np.float64)          # [N,1,16,16] target eps
    t = np.asarray(f["t"]).astype(np.int64).ravel()   # [N]
    N = x.shape[0]
    H = x.shape[2]; Wd = x.shape[3]
    abar = np.asarray(f["alphabar"], dtype=np.float64).ravel()

    # ---------------- helpers ----------------
    def pad1(a):
        # a: [N,C,H,W] -> zero padded [N,C,H+2,W+2]
        p = np.zeros((a.shape[0], a.shape[1], a.shape[2] + 2, a.shape[3] + 2), dtype=a.dtype)
        p[:, :, 1:-1, 1:-1] = a
        return p

    def conv3x3(a, Wt, bias):
        # a: [N,Cin,H,W], Wt: [Cout,Cin,3,3] (cross-correlation), bias: [Cout]
        ap = pad1(a)
        Nn, Cin, Hh, Ww = ap.shape
        Cout = Wt.shape[0]
        out = np.zeros((Nn, Cout, Hh - 2, Ww - 2), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                sl = ap[:, :, i:i + Hh - 2, j:j + Ww - 2]
                out += np.tensordot(sl, Wt[:, :, i, j], axes=([1], [1]))
        out += bias[None, :, None, None]
        return out

    def shift(a, dy, dx):
        # shift feature map by (dy,dx) with zero fill
        out = np.zeros_like(a)
        ys0, ys1 = max(0, dy), min(H, H + dy)
        yd0, yd1 = ys0 - dy, ys1 - dy
        xs0, xs1 = max(0, dx), min(Wd, Wd + dx)
        xd0, xd1 = xs0 - dx, xs1 - dx
        out[:, :, yd0:yd1, xd0:xd1] = a[:, :, ys0:ys1, xs0:xs1]
        return out

    # ---------------- design frozen feature stack ----------------
    # conv1: 2 -> 32.  ch0 = x_t (center), ch1..ch7 = shifted copies of x_t,
    #         ch8 = t/31 ramp, ch9 = t/31 * x_t (center), ch10..17 = shifted t*x
    C1 = 32
    w1 = np.zeros((C1, 2, 3, 3), dtype=np.float64)
    b1 = np.zeros(C1, dtype=np.float64)
    w1[0, 0, 1, 1] = 1.0                      # x_t center
    # shifted copies of x_t -> channels 1..7
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1)]
    for k, (dy, dx) in enumerate(shifts):
        w1[1 + k, 0, 1 - dy, 1 - dx] = 1.0      # so that conv output = x shifted by (dy,dx)
    w1[8, 1, 1, 1] = 1.0                       # t/31
    w1[9, 1, 1, 1] = 1.0                       # t/31 (for product channel)
    w1[9, 0, 1, 1] = 1.0                       # + x_t center -> t*x product
    for k, (dy, dx) in enumerate(shifts[:5]):
        w1[10 + k, 1, 1 - dy, 1 - dx] = 1.0
        w1[10 + k, 0, 1 - dy, 1 - dx] = 1.0

    # conv2: 32 -> 64.  ch0..7 = shifted x (identity), ch8 = t ramp,
    #         ch9 = t*x product, ch10..14 = shifted t*x, ch15..31 = 0
    C2 = 64
    w2 = np.zeros((C2, C1, 3, 3), dtype=np.float64)
    b2 = np.zeros(C2, dtype=np.float64)
    for k in range(8):
        w2[k, 1 + k, 1, 1] = 1.0
    w2[8, 8, 1, 1] = 1.0
    w2[9, 9, 1, 1] = 1.0
    for k in range(5):
        w2[10 + k, 10 + k, 1, 1] = 1.0

    # conv3: 64 -> 64.  ch0..7 = shifted x, ch8 = t ramp, ch9 = t*x,
    #         ch10..14 = shifted t*x, rest 0
    C3 = 64
    w3 = np.zeros((C3, C2, 3, 3), dtype=np.float64)
    b3 = np.zeros(C3, dtype=np.float64)
    for k in range(8):
        w3[k, k, 1, 1] = 1.0
    w3[8, 8, 1, 1] = 1.0
    w3[9, 9, 1, 1] = 1.0
    for k in range(5):
        w3[10 + k, 10 + k, 1, 1] = 1.0

    # ---------------- run frozen stack exactly as the network would ----------------
    tb = (t.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x, np.broadcast_to(tb, (N, 1, H, Wd))], axis=1)

    h1 = np.maximum(conv3x3(inp, w1, b1), 0.0)
    h2 = np.maximum(conv3x3(h1, w2, b2), 0.0)
    h3 = np.maximum(conv3x3(h2, w3, b3), 0.0)   # [N,64,16,16] = out-layer input

    # ---------------- closed-form ridge readout ----------------
    # design matrix rows: (N*H*W) samples, columns: 64*9 spatial taps + bias
    F = h3.transpose(0, 2, 3, 1).reshape(N * H * W, C3)   # [M,64]
    T = y.transpose(0, 2, 3, 1).reshape(N * H * W, 1)     # [M,1]

    # spatial tap matrix: out.weight[0,c,i,j] multiplies h3[c, y+i-1, x+j-1]
    # -> for sample at (y,x), tap (i,j) sees h3[c, y+i-1, x+j-1] = F shifted by (1-i, 1-j)
    M = N * H * W
    A = np.zeros((M, C3 * 9 + 1), dtype=np.float64)
    # build shifted feature columns
    cols = []
    for i in range(3):
        for j in range(3):
            dy, dx = 1 - i, 1 - j
            sh = shift(h3, dy, dx).transpose(0, 2, 3, 1).reshape(M, C3)
            cols.append(sh)
    A[:, :C3 * 9] = np.concatenate(cols, axis=1)
    A[:, C3 * 9] = 1.0

    # ridge (closed form), mild regularization
    lam = 1e-3 * M
    G = A.T @ A
    G[np.arange(C3 * 9, C3 * 9 + 1), np.arange(C3 * 9, C3 * 9 + 1)] += lam
    rhs = A.T @ T
    sol = np.linalg.solve(G, rhs)          # [C3*9+1, 1]

    out_w = np.zeros((1, C3, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out_w[0, :, i, j] = sol[i * 3 + j, 0]
    out_b = np.array([sol[C3 * 9, 0]], dtype=np.float64)

    # ---------------- verify on training data ----------------
    pred = conv3x3(h3, out_w, out_b)
    mse_new = float(np.mean((pred - y) ** 2))
    mse_old = float(np.mean((np.asarray(f["prediction"], dtype=np.float64) - y) ** 2))

    # per-timestep MSE (new)
    per_t = {}
    for tt in range(32):
        m = t == tt
        if m.sum() > 0:
            per_t[str(tt)] = round(float(np.mean((pred[m] - y[m]) ** 2)), 4)

    # ---------------- write back ----------------
    w["conv1.weight"] = w1.astype(np.float32)
    w["conv1.bias"] = b1.astype(np.float32)
    w["conv2.weight"] = w2.astype(np.float32)
    w["conv2.bias"] = b2.astype(np.float32)
    w["conv3.weight"] = w3.astype(np.float32)
    w["conv3.bias"] = b3.astype(np.float32)
    w["out.weight"] = out_w.astype(np.float32)
    w["out.bias"] = out_b.astype(np.float32)

    # safety: finite, |v| <= 1000
    for k in list(w.keys()):
        arr = np.asarray(w[k], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1000.0, neginf=-1000.0)
        np.clip(arr, -1000.0, 1000.0, out=arr)
        w[k] = arr

    diag = {
        "mse_before": round(mse_old, 5),
        "mse_after": round(mse_new, 5),
        "per_timestep_mse": per_t,
        "out_weight_absmax": round(float(np.abs(out_w).max()), 4),
        "out_bias": round(float(out_b[0]), 4),
        "n_samples": int(M),
        "n_regressors": int(C3 * 9 + 1),
    }
    return {"weights": w, "diagnostics": diag}
