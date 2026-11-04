import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------
    # 0. helpers
    # ------------------------------------------------------------------
    def conv2d(x, W, b, pad=1):
        """x: (N,C,H,W), W: (O,C,3,3) cross-correlation, zero padding."""
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        # (N,C,H+2,W+2) -> (N,C,3,3,H,W)
        s = xp.strides
        patches = np.lib.stride_tricks.as_strided(
            xp,
            shape=(N, C, 3, 3, H, Wd),
            strides=(s[0], s[1], s[2], s[3], s[2], s[3]),
            writeable=False)
        # einsum over (C,3,3)
        out = np.einsum('nckhw,ock->nohw', patches, W, optimize=True)
        if b is not None:
            out += b.reshape(1, O, 1, 1)
        return out

    def relu(a):
        return np.maximum(a, 0.0)

    # ------------------------------------------------------------------
    # 1. training data
    # ------------------------------------------------------------------
    x = np.asarray(f["x"], dtype=np.float64)          # (N,1,16,16)
    t = np.asarray(f["t"], dtype=np.int64)            # (N,)
    y = np.asarray(f["y"], dtype=np.float64)          # (N,1,16,16) target eps
    N = x.shape[0]
    H = Wd = x.shape[2]
    tau = (t.astype(np.float64) / 31.0)[:, None, None]   # (N,1,1)

    # ------------------------------------------------------------------
    # 2. design: feature map  F[n,c,i,j]  (32 channels)  and out-layer taps
    #    channel 2k   : ReLU( x_t + g_k*tau + b_k )      -> gives v_k
    #    channel 2k+1 : ReLU( -x_t - g_k*tau - b_k )    -> gives -v_k part
    #    so out can form v_k = ch2k - ch2k+1  and |v_k| = ch2k + ch2k+1
    # ------------------------------------------------------------------
    K = 16                       # number of pairs
    # gamma_k, beta_k : kink location of the hinge feature as a function of t
    #   gamma spans a range so that kink = -gamma_k*tau - beta_k covers
    #   useful thresholds of x_t at different noise levels.
    gamma = np.linspace(-1.0, 1.0, K)
    beta = np.linspace(-0.6, 0.6, K)

    # feature construction (on all training examples)
    xt = x[:, 0]                                     # (N,H,W)
    F = np.empty((N, 2 * K, H, Wd), dtype=np.float64)
    for k in range(K):
        v = xt + gamma[k] * tau[:, 0][:, None] + beta[k]   # (N,H,W)
        F[:, 2 * k] = np.maximum(v, 0.0)
        F[:, 2 * k + 1] = np.maximum(-v, 0.0)

    # out-layer taps: 32 channels x 9 positions (+1 bias) = 289 unknowns
    # patch extraction with stride tricks -> (N, H, W, 32*9)
    s = F.strides
    patches = np.lib.stride_tricks.as_strided(
        F,
        shape=(N, 2 * K, 3, 3, H, Wd),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]),
        writeable=False)
    P = patches.transpose(0, 4, 5, 1, 2, 3).reshape(N, H * Wd, 2 * K * 9)

    # ------------------------------------------------------------------
    # 3. ridge normal equations for the out layer
    # ------------------------------------------------------------------
    Y = y[:, 0].reshape(N, H * Wd).T                 # (HW, N)
    # subsample examples for the Gram matrix (keep all pixels)
    use = np.arange(N)
    if N > 900:
        use = np.linspace(0, N - 1, 900).astype(np.int64)
    Pu = P[use].reshape(-1, 2 * K * 9)
    Yu = Y[:, use].T.reshape(-1)

    G = Pu.T @ Pu
    rhs = Pu.T @ Yu
    # ridge on the tap block, no penalty on the bias
    lam = 1e-3 * float(np.trace(G)) / max(G.shape[0], 1)
    G = G + lam * np.eye(G.shape[0])
    beta_out = np.linalg.solve(G, rhs)

    # ------------------------------------------------------------------
    # 4. assemble new weights
    # ------------------------------------------------------------------
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float64)
    b1 = np.zeros(32, dtype=np.float64)
    for k in range(K):
        c = 2 * k
        # channel c : +x_t (center tap) + gamma_k*tau + beta_k
        W1[c, 0, 1, 1] = 1.0
        W1[c, 1, 1, 1] = gamma[k]          # tau channel is spatially constant
        b1[c] = beta[k]
        # channel c+1 : -(x_t + gamma_k*tau + beta_k)
        W1[c + 1, 0, 1, 1] = -1.0
        W1[c + 1, 1, 1, 1] = -gamma[k]
        b1[c + 1] = -beta[k]

    W2 = np.zeros((64, 32, 3, 3), dtype=np.float64)
    b2 = np.zeros(64, dtype=np.float64)
    for c in range(32):
        W2[c, c, 1, 1] = 1.0             # identity pass-through (no bias)

    W3 = np.zeros((64, 64, 3, 3), dtype=np.float64)
    b3 = np.zeros(64, dtype=np.float64)
    for c in range(32):
        W3[c, c, 1, 1] = 1.0             # identity pass-through (no bias)

    Wo = np.zeros((1, 64, 3, 3), dtype=np.float64)
    bo = np.zeros(1, dtype=np.float64)
    for c in range(32):
        Wo[0, c, 1, 1] = beta_out[c * 9 + 4]   # center tap of channel c
    bo[0] = beta_out[-1]

    # ------------------------------------------------------------------
    # 5. verify with the exact forward pass on the training data
    # ------------------------------------------------------------------
    def forward(xin, tin, W1, b1, W2, b2, W3, b3, Wo, bo):
        tauin = (tin.astype(np.float64) / 31.0)[:, None, None, None]
        inp = np.concatenate([xin, np.broadcast_to(tauin, xin.shape)], axis=1)
        h1 = relu(conv2d(inp, W1, b1))
        h2 = relu(conv2d(h1, W2, b2))
        h3 = relu(conv2d(h2, W3, b3))
        o = conv2d(h3, Wo, bo)
        return o

    pred = forward(x, t, W1, b1, W2, b2, W3, b3, Wo, bo)
    mse = float(np.mean((pred - y) ** 2))
    base = float(np.mean((np.zeros_like(y) - y) ** 2))

    # per-timestep mse
    per_t = {}
    for tt in range(32):
        m = t == tt
        if m.any():
            per_t[str(tt)] = round(float(np.mean((pred[m] - y[m]) ** 2)), 4)

    # ------------------------------------------------------------------
    # 6. write back (float32, finite, |.| <= 1000)
    # ------------------------------------------------------------------
    def put(name, arr):
        a = np.asarray(arr, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(a, -1000.0, 1000.0, out=a)
        w[name] = a

    put("conv1.weight", W1)
    put("conv1.bias", b1)
    put("conv2.weight", W2)
    put("conv2.bias", b2)
    put("conv3.weight", W3)
    put("conv3.bias", b3)
    put("out.weight", Wo)
    put("out.bias", bo)

    return {"weights": w,
            "diagnostics": {
                "train_mse_new": round(mse, 5),
                "train_mse_zero_baseline": round(base, 5),
                "mse_improvement": round(base - mse, 5),
                "per_t_mse": per_t,
                "n_fit_examples": int(len(use)),
                "n_out_taps": int(2 * K * 9 + 1),
                "ridge_lambda": float(lam),
            }}
