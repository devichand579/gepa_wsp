import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv3x3(x, W, b):
        """x: [N,C,H,W], W: [O,C,3,3], b: [O] -> [N,O,H,W] (cross-correlation, pad 1)."""
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.zeros((N, C, H + 2, Wd + 2), dtype=np.float32)
        xp[:, :, 1:-1, 1:-1] = x
        out = np.zeros((N, O, H, Wd), dtype=np.float32)
        for kr in range(3):
            for kc in range(3):
                out += np.tensordot(xp[:, :, kr:kr + H, kc:kc + Wd], W[:, :, kr, kc], axes=([1], [1]))
        if b is not None:
            out += b.reshape(1, O, 1, 1)
        return out

    def relu(x):
        return np.maximum(x, 0.0)

    def forward(x, t, ww):
        """replicate the fixed architecture"""
        N = x.shape[0]
        tb = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(tb, (N, 1, x.shape[2], x.shape[3]))], axis=1)
        h1 = relu(conv3x3(inp, ww["conv1.weight"], ww["conv1.bias"]))
        h2 = relu(conv3x3(h1, ww["conv2.weight"], ww["conv2.bias"]))
        h3 = relu(conv3x3(h2, ww["conv3.weight"], ww["conv3.bias"]))
        o = conv3x3(h3, ww["out.weight"], ww["out.bias"])
        return h3, o

    def mse_of(pred, y):
        d = pred - y
        return float(np.mean(d * d))

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float32)          # [N,1,16,16]
    t = np.asarray(f["t"]).astype(np.int64)           # [N]
    y = np.asarray(f["y"], dtype=np.float32)          # [N,1,16,16]
    ab = np.asarray(f["alphabar"], dtype=np.float64)  # [32]
    N = x.shape[0]
    H = Wd = 16

    # ---------------- design matrix for the last layer ----------------
    # row for (n, pixel p): features from 3x3 neighbourhood of F[n,:,p] (zero padded)
    # column index: c*9 + kr*3 + kc   (c: channel, kr/kc: kernel offset)
    def build_design(F):
        """F: [N,64,16,16] -> D: [N*256, 576] float32"""
        Nn, C = F.shape[0], F.shape[1]
        Fp = np.zeros((Nn, C, H + 2, Wd + 2), dtype=np.float32)
        Fp[:, :, 1:-1, 1:-1] = F
        cols = []
        for kr in range(3):
            for kc in range(3):
                cols.append(Fp[:, :, kr:kr + H, kc:kc + Wd])   # [N,C,H,W]
        D = np.stack(cols, axis=2)                            # [N,C,9,H,W]
        D = D.transpose(0, 1, 3, 4, 2).reshape(Nn * H * Wd, C * 9)
        return np.ascontiguousarray(D)

    def fit_out(F, y, lam=1e-3):
        """ridge least squares for out.weight/out.bias given features F and targets y."""
        Nn = F.shape[0]
        D = build_design(F)
        Y = y.reshape(Nn * H * Wd).astype(np.float64)
        G = (D.T @ D).astype(np.float64)
        bvec = (D.T @ Y).astype(np.float64)
        # augment bias
        G2 = np.zeros((577, 577), dtype=np.float64)
        G2[:576, :576] = G
        G2[576, 576] = float(Nn * H * Wd)
        G2[:576, 576] = G.sum(axis=1)
        G2[576, :576] = G2[:576, 576]
        r2 = np.zeros(577, dtype=np.float64)
        r2[:576] = bvec
        r2[576] = Y.sum()
        scale = np.trace(G2) / 577.0
        G2 += lam * scale * np.eye(577)
        try:
            sol = np.linalg.solve(G2, r2)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(G2, r2, rcond=None)[0]
        Wout = sol[:576].reshape(1, 64, 3, 3).astype(np.float32)
        bout = np.array([sol[576]], dtype=np.float32)
        return Wout, bout

    def eval_variant(ww, F=None):
        """returns (mse, features) using given weights; recompute features if needed"""
        if F is None:
            F, _ = forward(x, t, ww)
        pred = conv3x3(F, ww["out.weight"], ww["out.bias"])
        return mse_of(pred, y), F

    base = {k: np.asarray(v, dtype=np.float32).copy() for k, v in w.items()}

    # ---------------- variant A: refit only the last layer on frozen features ----------------
    featA = np.asarray(f["features"], dtype=np.float32)   # [2000,64,16,16] = out's input
    Wo, bo = fit_out(featA, y, lam=1e-3)
    A = dict(base)
    A["out.weight"] = Wo
    A["out.bias"] = bo
    mseA = mse_of(conv3x3(featA, Wo, bo), y)

    # ---------------- variant B: t-gated linear feature bank ----------------
    # per-timestep optimal gain for a linear predictor of eps from x_t
    x0 = np.asarray(f["clean"], dtype=np.float64)
    var0 = float(np.mean(x0 * x0)) - float(np.mean(x0)) ** 2
    var0 = max(var0, 1e-3)
    ab_t = ab[t]                                     # [N]
    vx = ab_t * var0 + (1.0 - ab_t)
    gain = np.sqrt(np.maximum(1.0 - ab_t, 0.0)) / np.maximum(vx, 1e-6)
    gain = np.clip(gain, 0.0, 1.0)

    B = dict(base)
    C1 = 32
    W1 = np.zeros((C1, 2, 3, 3), dtype=np.float32)
    b1 = np.zeros((C1,), dtype=np.float32)
    # 16 channels: gated identity on x_t  ->  W1[k,0,1,1] = gain(t), bias small positive
    # 16 channels: t-encoding (constant channel) -> W1[16+k,1,1,1] = c_k, bias d_k
    half = 16
    for k in range(half):
        W1[k, 0, 1, 1] = 0.0
        b1[k] = 0.0
    # t-encoding basis: use a smooth set of basis functions of u=t/31
    u = (t.astype(np.float64) / 31.0)
    basis = [np.ones_like(u), u, u ** 2, u ** 3, np.sin(np.pi * u), np.cos(np.pi * u),
             np.sin(2 * np.pi * u), np.cos(2 * np.pi * u)]
    nb = len(basis)
    for j in range(nb):
        idx = half + (j % (C1 - half))
        # spread basis across the 16 t-channels with different scales
        W1[idx, 1, 1, 1] = 0.0
    # simpler & explicit: t channels encode u and u^2 with a few scales
    for j in range(C1 - half):
        idx = half + j
        s = 1.0 + 0.5 * j
        W1[idx, 1, 1, 1] = 0.0
    # (t-channel weights are set below after we know the needed structure)
    # --- gated x_t channels: the gain must be a *function of t*; conv1 sees the t channel
    #     so implement gain(t) via a 2-channel product approximated by the readout:
    #     we instead give conv1 raw x_t and raw t and let the LAST layer (linear in
    #     features) form the product only if features are products -> they are not.
    #     => use the analytic gain directly as a per-example scale applied to the
    #        *input* of conv1 is impossible (weights are shared across examples).
    #     => instead: give conv1 the raw x_t channels (identity) and t channels, and
    #        additionally pre-multiply x_t by gain(t) is impossible.
    #  => Practical approach: identity x_t channels + t channels; the readout then
    #     fits a *global* linear map; t-dependence of the gain is approximated by
    #     the t-channel features interacting through the (nonlinear) downstream layers.
    for k in range(half):
        W1[k, 0, 1, 1] = 1.0
        b1[k] = 0.05
    for j in range(C1 - half):
        idx = half + j
        W1[idx, 1, 1, 1] = (1.0 + 0.35 * j) if (j % 2 == 0) else -(0.4 + 0.2 * j)
        b1[idx] = 0.05
    B["conv1.weight"] = W1
    B["conv1.bias"] = b1

    # conv2 / conv3 : near-identity 3x3 (delta) so features stay ~linear in x_t
    def delta_w(Cin, Cout, eps=0.02, seed=0):
        rng = np.random.default_rng(seed)
        Wd_ = rng.normal(0.0, eps, size=(Cout, Cin, 3, 3)).astype(np.float32)
        for o in range(Cout):
            Wd_[o, o % Cin, 1, 1] = 1.0
        return Wd_
    B["conv2.weight"] = delta_w(32, 64, 0.02, 1)
    B["conv2.bias"] = np.full((64,), 0.05, dtype=np.float32)
    B["conv3.weight"] = delta_w(64, 64, 0.02, 2)
    B["conv3.bias"] = np.full((64,), 0.05, dtype=np.float32)

    # forward with new early layers -> new features
    _, _ = forward(x[:8], t[:8], B)   # smoke test
    featB, _ = forward(x, t, B)
    WoB, boB = fit_out(featB, y, lam=1e-3)
    B["out.weight"] = WoB
    B["out.bias"] = boB
    mseB = mse_of(conv3x3(featB, WoB, boB), y)

    # ---------------- variant C: identity-ish random features + refit ----------------
    # (fallback: keep original early layers, refit out)  == variant A

    # ---------------- choose ----------------
    diag = {"mse_before": None, "mse_A_lastlayer_refit": round(mseA, 6),
            "mse_B_gated_rebuild": round(mseB, 6), "var0": round(var0, 5)}

    # baseline mse from the supplied prediction
    pred0 = np.asarray(f["prediction"], dtype=np.float32)
    diag["mse_before"] = round(mse_of(pred0, y), 6)

    if mseB < mseA:
        chosen = B
        diag["chosen"] = "B"
    else:
        chosen = A
        diag["chosen"] = "A"

    out = {}
    for k, v in base.items():
        out[k] = np.ascontiguousarray(np.clip(chosen[k], -1000.0, 1000.0).astype(np.float32))
    return {"weights": out, "diagnostics": diag}
