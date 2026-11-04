import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _conv3x3(x, W, b):
    """Cross-correlation conv, stride 1, pad 1. x:[N,C,H,W] W:[O,C,3,3] b:[O]."""
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float32)
    xp[:, :, 1:-1, 1:-1] = x
    # im2col via sliding windows: [N,C,H,W,3,3]
    win = sliding_window_view(xp, (3, 3), axis=(2, 3))  # [N,C,H,W,3,3]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * Wd, C * 9)
    Wf = W.reshape(O, C * 9).astype(np.float32)
    out = cols @ Wf.T + b.astype(np.float32)
    return out.reshape(N, H, Wd, O).transpose(0, 3, 1, 2)


def edit(w, f):
    rng = np.random.RandomState(1234)

    # ---------------- 1. Rebuild a clean, well-conditioned feature extractor ----------------
    # conv1: 2 -> 32, 3x3.  Channel 0 = x_t, channel 1 = t/31 (constant in space).
    # We build per-pixel ReLU units of (a*x_t + B*t + b) plus a few spatial filters,
    # so the readout can express a t-gated, locally-mixing denoiser.
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float32)
    c1b = np.zeros((32,), dtype=np.float32)

    # 16 "per-pixel" units: center tap on x_t, t-gate on channel 1, varied thresholds.
    slopes = np.linspace(0.6, 2.4, 16).astype(np.float32)
    tcoefs = np.linspace(-1.5, 1.5, 16).astype(np.float32)
    thresh = np.linspace(-0.6, 0.6, 16).astype(np.float32)
    for k in range(16):
        c1w[k, 0, 1, 1] = slopes[k]
        c1w[k, 1, 1, 1] = tcoefs[k]
        c1b[k] = -thresh[k] * slopes[k]

    # 16 "spatial" units: small random 3x3 filters on x_t (local pattern detectors).
    for k in range(16, 32):
        c1w[k, 0] = rng.randn(3, 3).astype(np.float32) * 0.5
        c1w[k, 1, 1, 1] = rng.randn() * 0.5
        c1b[k] = rng.randn() * 0.1

    # conv2: 32 -> 64, identity copy (channels 0..31 duplicated to 32..63).
    c2w = np.zeros((64, 32, 3, 3), dtype=np.float32)
    c2b = np.zeros((64,), dtype=np.float32)
    for k in range(32):
        c2w[k, k, 1, 1] = 1.0
        c2w[k + 32, k, 1, 1] = 1.0

    # conv3: 64 -> 64, identity.
    c3w = np.zeros((64, 64, 3, 3), dtype=np.float32)
    c3b = np.zeros((64,), dtype=np.float32)
    for k in range(64):
        c3w[k, k, 1, 1] = 1.0

    # ---------------- 2. Compute features on the full 2000 training examples ----------------
    x = np.asarray(f["x"], dtype=np.float32)          # [2000,1,16,16]
    t = np.asarray(f["t"], dtype=np.float32)          # [2000]
    y = np.asarray(f["y"], dtype=np.float32)          # [2000,1,16,16]

    N = x.shape[0]
    tch = (t / 31.0).reshape(N, 1, 1, 1).astype(np.float32)
    inp = np.concatenate([x, np.broadcast_to(tch, (N, 1, 16, 16))], axis=1)

    h1 = np.maximum(_conv3x3(inp, c1w, c1b), 0.0)
    h2 = np.maximum(_conv3x3(h1, c2w, c2b), 0.0)
    h3 = np.maximum(_conv3x3(h2, c3w, c3b), 0.0)      # [2000,64,16,16]

    # ---------------- 3. Closed-form ridge least squares for the out layer ----------------
    # out(i,j) = sum_{c,u,v} W[c,u,v] * h3[c, i+u-1, j+v-1] + bias
    # Build normal equations A = X^T X (577x577), b = X^T y (577), chunked to bound memory.
    P = 64 * 9 + 1
    A = np.zeros((P, P), dtype=np.float64)
    B = np.zeros((P,), dtype=np.float64)

    h3p = np.zeros((N, 64, 18, 18), dtype=np.float32)
    h3p[:, :, 1:-1, 1:-1] = h3
    win = sliding_window_view(h3p, (3, 3), axis=(2, 3))   # [N,64,16,16,3,3]

    chunk = 200
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        cols = win[s:e].transpose(0, 2, 3, 1, 4, 5).reshape(-1, 64 * 9)
        cols = np.concatenate([cols, np.ones((cols.shape[0], 1), dtype=np.float32)], axis=1)
        tgt = y[s:e].reshape(-1)
        cols64 = cols.astype(np.float64)
        A += cols64.T @ cols64
        B += cols64.T @ tgt.astype(np.float64)

    lam = 1e-2 * np.trace(A) / P
    A[np.diag_indices_from(A)] += lam
    sol = np.linalg.solve(A, B)

    out_w = sol[:-1].reshape(1, 64, 3, 3).astype(np.float32)
    out_b = np.array([sol[-1]], dtype=np.float32)

    # ---------------- 4. Write back ----------------
    w["conv1.weight"] = c1w
    w["conv1.bias"] = c1b
    w["conv2.weight"] = c2w
    w["conv2.bias"] = c2b
    w["conv3.weight"] = c3w
    w["conv3.bias"] = c3b
    w["out.weight"] = out_w
    w["out.bias"] = out_b

    # ---------------- 5. Diagnostics ----------------
    pred = (_conv3x3(h3, out_w, out_b)).reshape(-1)
    mse = float(np.mean((pred - y.reshape(-1)) ** 2))
    diag = {
        "train_mse_after_fit": mse,
        "ridge_lambda": float(lam),
        "out_weight_norm": float(np.linalg.norm(out_w)),
        "out_bias": float(out_b[0]),
        "n_equations": int(N * 256),
        "n_params": int(P),
    }
    return {"weights": w, "diagnostics": diag}
