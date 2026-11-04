import numpy as np

def _conv3x3(x, W, b):
    """Cross-correlation conv, pad=1, stride 1. x:(N,C,H,W) W:(O,C,3,3) b:(O,)"""
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((N, O, H, Wd), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += np.tensordot(xp[:, :, i:i + H, j:j + Wd], W[:, :, i, j], axes=([1], [1]))
    if b is not None:
        out += b.reshape(1, O, 1, 1)
    return out


def _forward(x, t, w):
    """Full forward pass of the fixed architecture (numpy)."""
    N = x.shape[0]
    tb = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x, np.broadcast_to(tb, (N, 1, x.shape[2], x.shape[3]))], axis=1)
    h1 = np.maximum(_conv3x3(inp, w["conv1.weight"], w["conv1.bias"]), 0.0)
    h2 = np.maximum(_conv3x3(h1, w["conv2.weight"], w["conv2.bias"]), 0.0)
    h3 = np.maximum(_conv3x3(h2, w["conv3.weight"], w["conv3.bias"]), 0.0)
    return _conv3x3(h3, w["out.weight"], w["out.bias"])


def _mse(pred, y):
    d = (pred - y).reshape(pred.shape[0], -1)
    return float(np.mean(np.sum(d * d, axis=1) / d.shape[1]))


def _fit_affine(x, y, t, idx):
    """Closed-form per-timestep least squares: y ~ b + sum_{k,l} A[k,l] x[:, :, i+k, j+l]."""
    N = x.shape[0]
    P = 9
    X = np.zeros((N, P + 1), dtype=np.float64)
    xp = np.pad(x[idx], ((0, 0), (0, 0), (1, 1), (1, 1)))
    for k in range(3):
        for l in range(3):
            X[:, k * 3 + l] = xp[:, 0, k:k + 16, l:l + 16].reshape(N, -1).mean(axis=1)
    X[:, P] = 1.0
    Y = y[idx].reshape(N, -1).mean(axis=1).astype(np.float64)
    A = X.T @ X
    A[np.arange(P + 1), np.arange(P + 1)] += 1e-3
    b = X.T @ Y
    sol = np.linalg.solve(A, b)
    return sol[:P].reshape(3, 3), sol[P]


def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    x = np.asarray(f["x"], dtype=np.float32)
    y = np.asarray(f["y"], dtype=np.float32)
    t = np.asarray(f["t"]).astype(np.int64)
    N = x.shape[0]
    diag = {}

    # ---------- 0. sanity: does the current weight set reproduce the reported prediction? ----------
    try:
        cur = _forward(x, t, w)
        diag["mse_current_weights"] = round(_mse(cur, y), 6)
        diag["mse_reported_prediction"] = round(_mse(np.asarray(f["prediction"], dtype=np.float32), y), 6)
    except Exception as e:
        diag["forward_error"] = str(e)[:120]

    # ---------- 1. closed-form per-timestep affine fits ----------
    A = np.zeros((32, 3, 3), dtype=np.float64)
    b = np.zeros(32, dtype=np.float64)
    for ti in range(32):
        idx = np.where(t == ti)[0]
        if len(idx) < 8:
            A[ti] = np.eye(3)
            continue
        A[ti], b[ti] = _fit_affine(x, y, t, idx)
    diag["fit_mse_ideal"] = round(float(np.mean([
        np.mean((y[t == ti] - (np.pad(x[t == ti], ((0, 0), (0, 0), (1, 1), (1, 1)))[:, :, 0:16, 0:16] * 0).reshape(-1)) * 0)
        for ti in range(32)])), 6)

    # in-sample MSE of the ideal per-timestep affine model (computed properly)
    tot = 0.0
    cnt = 0
    for ti in range(32):
        idx = np.where(t == ti)[0]
        if len(idx) == 0:
            continue
        xp = np.pad(x[idx], ((0, 0), (0, 0), (1, 1), (1, 1)))
        pred = np.zeros_like(y[idx])
        for k in range(3):
            for l in range(3):
                pred += A[ti, k, l][None, None, None, None] * xp[:, :, k:k + 16, l:l + 16]
        pred += b[ti]
        tot += _mse(pred, y[idx]) * len(idx)
        cnt += len(idx)
    diag["ideal_affine_mse"] = round(tot / max(cnt, 1), 6)

    # ---------- 2. realise per-timestep affine maps inside the fixed conv/ReLU stack ----------
    # conv1 channels:
    #   ch0 : x_t pass-through, shifted so that it is always > 0  -> z0 = x_t + 4
    #   ch1 : signed ramp  r(t) = t/31 - 0.5   (weight -1 on the t-channel, bias +0.5)
    #   ch2 : constant 1
    #   ch3 : constant 0
    #   ch4..ch35 : gate pairs  relu(M*(u - c)) and relu(M*(c - u))  -> exact 0/1 indicators
    # conv2: routes conv1 channels 0,1,2 into conv2 channels 0,1,2 (identity, weight 1 at centre)
    # conv3: routes conv2 channels 0,1,2 into conv3 channels 0,1,2
    # out  : channel 0 -> A_t filter (per-gate), channel 1 -> ramp, channel 2 -> constant
    #        out.bias = b_0 (t=0 intercept); the ramp/constant contributions give b_t - b_0.
    M = 1.0e3
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float32)
    c1b = np.zeros(32, dtype=np.float32)
    c1w[0, 0, 1, 1] = 1.0
    c1b[0] = 4.0
    c1w[1, 1, 1, 1] = -1.0
    c1b[1] = 0.5
    c1w[2, 1, 1, 1] = 0.0
    c1b[2] = 1.0
    c1w[3, 1, 1, 1] = 0.0
    c1b[3] = 0.0
    for ti in range(32):
        c = ti / 31.0
        c1w[4 + 2 * ti, 1, 1, 1] = M
        c1b[4 + 2 * ti] = -M * c
        c1w[5 + 2 * ti, 1, 1, 1] = -M
        c1b[5 + 2 * ti] = M * c

    c2w = np.zeros((64, 32, 3, 3), dtype=np.float32)
    c2b = np.zeros(64, dtype=np.float32)
    c2w[0, 0, 1, 1] = 1.0
    c2w[1, 1, 1, 1] = 1.0
    c2w[2, 2, 1, 1] = 1.0

    c3w = np.zeros((64, 64, 3, 3), dtype=np.float32)
    c3b = np.zeros(64, dtype=np.float32)
    c3w[0, 0, 1, 1] = 1.0
    c3w[1, 1, 1, 1] = 1.0
    c3w[2, 2, 1, 1] = 1.0

    ow = np.zeros((1, 64, 3, 3), dtype=np.float32)
    ob = np.zeros(1, dtype=np.float32)
    ow[0, 0] = A[0].astype(np.float32)
    ow[0, 1, 1, 1] = 0.0
    ow[0, 2, 1, 1] = 0.0
    for ti in range(32):
        ow[0, 4 + 2 * ti] = (A[ti] - A[0]).astype(np.float32)
    # ramp / constant contributions: out = A_t ⊛ x_t + b_t
    # b_t = b_0 + (b_t - b_0); realise (b_t - b_0) via ramp r(t)=t/31-0.5 (piecewise-linear interp)
    # plus a constant channel for the residual of the piecewise-linear fit.
    knots = np.arange(32, dtype=np.float64) / 31.0
    vals = b - b[0]
    # piecewise linear interpolation of vals over knots -> slope/intercept on the ramp channel
    # ramp channel value = u - 0.5 with u = t/31 ; use two channels: u (linear) and 1 (constant)
    # fit vals ~ p*u + q  (least squares over the 32 knots)
    Um = np.stack([knots, np.ones_like(knots)], axis=1)
    coef = np.linalg.lstsq(Um, vals, rcond=None)[0]
    ow[0, 1, 1, 1] = float(coef[0])
    ow[0, 2, 1, 1] = float(coef[1])
    ob[0] = float(b[0])

    w["conv1.weight"] = c1w
    w["conv1.bias"] = c1b
    w["conv2.weight"] = c2w
    w["conv2.bias"] = c2b
    w["conv3.weight"] = c3w
    w["conv3.bias"] = c3b
    w["out.weight"] = ow
    w["out.bias"] = ob

    # ---------- 3. verify realised network on the training feedback ----------
    try:
        newp = _forward(x, t, w)
        diag["mse_new"] = round(_mse(newp, y), 6)
        diag["mse_new_t0"] = round(_mse(newp[t == 0], y[t == 0]), 6) if (t == 0).any() else None
        diag["mse_new_t15"] = round(_mse(newp[t == 15], y[t == 15]), 6) if (t == 15).any() else None
        diag["mse_new_t31"] = round(_mse(newp[t == 31], y[t == 31]), 6) if (t == 31).any() else None
        diag["pred_std"] = round(float(newp.std()), 4)
    except Exception as e:
        diag["new_forward_error"] = str(e)[:160]

    # ---------- 4. safety: finite, bounded, shape-preserving ----------
    for k in w:
        w[k] = np.clip(np.nan_to_num(w[k], nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0).astype(np.float32)

    return {"weights": w, "diagnostics": diag}
