import numpy as np

def edit(w, f):
    # ------------------------------------------------------------------ data
    x = np.asarray(f["x"], dtype=np.float64)          # [N,1,16,16]
    y = np.asarray(f["y"], dtype=np.float64)          # [N,1,16,16]
    t = np.asarray(f["t"]).astype(np.int64).ravel()   # [N]
    N = x.shape[0]
    H = W_ = x.shape[2]
    T = 32

    # ------------------------------------------------- closed-form per-t fit
    # y ~ alpha_t * x + beta_t   (per-pixel regression, per timestep)
    alpha = np.zeros(T); beta = np.zeros(T)
    for j in range(T):
        m = (t == j)
        if not m.any():
            continue
        xj = x[m].ravel(); yj = y[m].ravel()
        xm = xj.mean(); ym = yj.mean()
        vx = xj.var()
        alpha[j] = ((xj - xm) * (yj - ym)).mean() / vx if vx > 1e-12 else 0.0
        beta[j] = ym - alpha[j] * xm

    # ------------------------------------------------- build the new network
    # conv1 : 2 -> 32
    #   ch 0 : +x  rail   (s=1)      ch 1 : -x rail  (s=-1)
    #   ch 2 : +x  rail   (s=2)      ch 3 : -x rail  (s=-2)
    #   ch 4..35 : step basis  ReLU(t/31 - u_k)   (u_k = k/31, k=0..31)
    # conv2 : 32 -> 64  : rails of x (s=1,2) + replicated step basis
    # conv3 : 64 -> 64  : rails of x (s=1,2) + replicated step basis
    # out   : 64 -> 1   : closed-form ridge fit on the produced features

    c = 4.0                     # rail offset (large vs |x|)
    nstep = T                   # step channels
    # step channel k : ReLU(t/31 - k/31)  -> 0 for t<=k, else (t-k)/31
    # indicator for t=j : 31*(step_{j-1} - step_j)  (step_{-1} == 1)
    # out layer will use: alpha_j = sum_j a_j * ind_j(t)

    W1 = np.zeros((32, 2, 3, 3)); b1 = np.zeros(32)
    W1[0, 0, 1, 1] = 1.0;  b1[0] = c
    W1[1, 0, 1, 1] = -1.0; b1[1] = c
    W1[2, 0, 1, 1] = 2.0;  b1[2] = c
    W1[3, 0, 1, 1] = -2.0; b1[3] = c
    for k in range(nstep):
        W1[4 + k, 1, 1, 1] = 1.0
        b1[4 + k] = -k / 31.0
    # channels 4..35 exceed 32 -> handled below by using 28 step channels
    # (timesteps 0..31 need 32 indicators; we use 28 knots + 4 coarse ones)
    # -> rebuild with exactly 32 channels: 4 rails + 28 steps
    nstep = 28
    W1 = np.zeros((32, 2, 3, 3)); b1 = np.zeros(32)
    W1[0, 0, 1, 1] = 1.0;  b1[0] = c
    W1[1, 0, 1, 1] = -1.0; b1[1] = c
    W1[2, 0, 1, 1] = 2.0;  b1[2] = c
    W1[3, 0, 1, 1] = -2.0; b1[3] = c
    for k in range(nstep):
        W1[4 + k, 1, 1, 1] = 1.0
        b1[4 + k] = -k / 31.0

    # ---- conv2 : rails of x from conv1 rails, plus step replication
    # conv1 rail pairs: (0,1) -> s=1 ; (2,3) -> s=2
    W2 = np.zeros((64, 32, 3, 3)); b2 = np.zeros(64)
    # conv2 ch 0 : +x (s=1)  = conv1[0] - conv1[1]
    W2[0, 0, 1, 1] = 1.0;  W2[0, 1, 1, 1] = -1.0; b2[0] = c
    # conv2 ch 1 : -x (s=1)
    W2[1, 0, 1, 1] = -1.0; W2[1, 1, 1, 1] = 1.0;  b2[1] = c
    # conv2 ch 2 : +x (s=2)
    W2[2, 2, 1, 1] = 1.0;  W2[2, 3, 1, 1] = -1.0; b2[2] = c
    # conv2 ch 3 : -x (s=2)
    W2[3, 2, 1, 1] = -1.0; W2[3, 3, 1, 1] = 1.0;  b2[3] = c
    # conv2 ch 4..31 : replicate conv1 step channels (identity)
    for k in range(nstep):
        W2[4 + k, 4 + k, 1, 1] = 1.0
    # conv2 ch 32..63 : replicate step channels again (spare capacity)
    for k in range(nstep):
        W2[32 + k, 4 + k, 1, 1] = 1.0

    # ---- conv3 : rails of x from conv2 rails, plus step replication
    W3 = np.zeros((64, 64, 3, 3)); b3 = np.zeros(64)
    W3[0, 0, 1, 1] = 1.0;  W3[0, 1, 1, 1] = -1.0; b3[0] = c
    W3[1, 0, 1, 1] = -1.0; W3[1, 1, 1, 1] = 1.0;  b3[1] = c
    W3[2, 2, 1, 1] = 1.0;  W3[2, 3, 1, 1] = -1.0; b3[2] = c
    W3[3, 2, 1, 1] = -1.0; W3[3, 3, 1, 1] = 1.0;  b3[3] = c
    for k in range(nstep):
        W3[4 + k, 4 + k, 1, 1] = 1.0
        W3[32 + k, 32 + k, 1, 1] = 1.0

    # ------------------------------------------------- forward (numpy conv)
    def conv2d(xin, Wc, bc):
        # xin [N,Cin,H,W], Wc [Cout,Cin,3,3] -> cross-correlation, pad 1
        N, Cin, Hh, Ww = xin.shape
        Cout = Wc.shape[0]
        xp = np.pad(xin, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = np.zeros((N, Cout, Hh, Ww))
        for kr in range(3):
            for kc in range(3):
                out += np.tensordot(xp[:, :, kr:kr + Hh, kc:kc + Ww],
                                   Wc[:, :, kr, kc], axes=([1], [1]))
        out += bc[None, :, None, None]
        return out

    def forward(xin, tin):
        N = xin.shape[0]
        tt = (tin.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([xin, np.broadcast_to(tt, (N, 1, H, W_))], axis=1)
        h1 = np.maximum(conv2d(inp, W1, b1), 0.0)
        h2 = np.maximum(conv2d(h1, W2, b2), 0.0)
        h3 = np.maximum(conv2d(h2, W3, b3), 0.0)
        return h3

    # ------------------------------------------------- closed-form out layer
    # out is 3x3 conv over h3 -> prediction.  Fit Wout[1,64,3,3], bout by
    # ridge least squares using the im2col design matrix (chunked).
    def im2col(h):
        N, C, Hh, Ww = h.shape
        hp = np.pad(h, ((0, 0), (0, 0), (1, 1), (1, 1)))
        cols = np.empty((N, C * 9, Hh * Ww))
        idx = 0
        for kr in range(3):
            for kc in range(3):
                cols[:, idx:idx + C, :] = hp[:, :, kr:kr + Hh, kc:kc + Ww].reshape(N, C, -1)
                idx += C
        return cols

    # accumulate normal equations in chunks
    P = 64 * 9 + 1
    AtA = np.zeros((P, P)); Atb = np.zeros(P)
    chunk = 100
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        h3 = forward(x[s:e], t[s:e])
        A = im2col(h3)                                  # [n, 576, 256]
        n = A.shape[0]
        B = y[s:e].reshape(n, 1, 256)
        # rows: (n*256) x 576
        Ar = A.transpose(0, 2, 1).reshape(n * 256, 576)
        Br = B.reshape(n * 256, 1)
        AtA[:576, :576] += Ar.T @ Ar
        AtA[:576, 576] += Ar.sum(axis=0)
        AtA[576, :576] += Ar.sum(axis=0)
        Atb[:576] += Ar.T @ Br[:, 0]
        Atb[576] += Br.sum()
    lam = 1e-3 * np.trace(AtA[:576, :576]) / 576.0 + 1e-8
    AtA[np.arange(576), np.arange(576)] += lam
    sol = np.linalg.solve(AtA, Atb)
    Wout = sol[:576].reshape(1, 64, 3, 3)
    bout = np.array([sol[576]])

    # ------------------------------------------------- assemble output weights
    new = {}
    for k, v in w.items():
        new[k] = np.array(v, dtype=np.float32, copy=True)
    new["conv1.weight"] = W1.astype(np.float32)
    new["conv1.bias"] = b1.astype(np.float32)
    new["conv2.weight"] = W2.astype(np.float32)
    new["conv2.bias"] = b2.astype(np.float32)
    new["conv3.weight"] = W3.astype(np.float32)
    new["conv3.bias"] = b3.astype(np.float32)
    new["out.weight"] = Wout.astype(np.float32)
    new["out.bias"] = bout.astype(np.float32)

    # ------------------------------------------------- sanity check (MSE)
    def predict(xin, tin, Wc1, bc1, Wc2, bc2, Wc3, bc3, Wo, bo):
        N = xin.shape[0]
        tt = (tin.astype(np.float64) / 31.0).reshape(N, 1, 1, 1)
        inp = np.concatenate([xin, np.broadcast_to(tt, (N, 1, H, W_))], axis=1)
        h1 = np.maximum(conv2d(inp, Wc1, bc1), 0.0)
        h2 = np.maximum(conv2d(h1, Wc2, bc2), 0.0)
        h3 = np.maximum(conv2d(h2, Wc3, bc3), 0.0)
        return conv2d(h3, Wo, bo)

    mse_new = float(np.mean((predict(x, t, W1, b1, W2, b2, W3, b3, Wout, bout) - y) ** 2))
    mse_old = float(np.mean((np.asarray(f["prediction"], dtype=np.float64) - y) ** 2))
    mse_lin = float(np.mean((alpha[t][:, None, None, None] * x
                             + beta[t][:, None, None, None] - y) ** 2))

    # clip / sanitize
    for k in new:
        new[k] = np.clip(np.nan_to_num(new[k], nan=0.0, posinf=0.0, neginf=0.0),
                         -1000.0, 1000.0).astype(np.float32)

    diag = {
        "mse_old": round(mse_old, 5),
        "mse_new_fit": round(mse_new, 5),
        "mse_linear_pert": round(mse_lin, 5),
        "alpha0": round(float(alpha[0]), 4), "alpha31": round(float(alpha[31]), 4),
        "out_w_absmean": round(float(np.abs(Wout).mean()), 6),
        "ridge_lambda": float(lam),
    }
    return {"weights": new, "diagnostics": diag}
