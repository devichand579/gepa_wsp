import numpy as np

# ---------------------------------------------------------------- helpers
def _conv2d(x, W, b):
    """x: (N,C,H,W) float32; W: (O,C,3,3) -> (N,O,H,W) cross-correlation, pad 1."""
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    # im2col via stride tricks: (N,C,3,3,H,W)
    s = xp.strides
    patches = np.lib.stride_tricks.as_strided(
        xp, shape=(N, C, 3, 3, H, Wd),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
    cols = patches.reshape(N, C * 9, H * Wd)          # (N, C*9, HW)
    Wf = W.reshape(O, C * 9).T                        # (C*9, O)
    out = np.einsum('ncp,po->nop', cols, Wf, optimize=True)
    out = out.reshape(N, O, H, Wd)
    if b is not None:
        out += b.reshape(1, O, 1, 1)
    return out


def _fwd(x, t, w):
    """Exact forward pass of the fixed architecture (NCHW)."""
    N = x.shape[0]
    tb = (t.astype(np.float32) / 31.0).reshape(N, 1, 1, 1)
    inp = np.concatenate([x, np.broadcast_to(tb, (N, 2, x.shape[2], x.shape[3]))], axis=1)
    h1 = np.maximum(_conv2d(inp, w['conv1.weight'], w['conv1.bias']), 0.0)
    h2 = np.maximum(_conv2d(h1, w['conv2.weight'], w['conv2.bias']), 0.0)
    h3 = np.maximum(_conv2d(h2, w['conv3.weight'], w['conv3.bias']), 0.0)
    return _conv2d(h3, w['out.weight'], w['out.bias'])


def _fit_lstsq(A, b, lam=1e-6):
    """A: (M,K) float32/float64, b: (M,) -> x (K,) via normal equations (ridge)."""
    A = A.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    G = A.T @ A
    G.flat[:: G.shape[0] + 1] += lam * max(1.0, np.trace(G) / max(1, G.shape[0]))
    return np.linalg.solve(G, A.T @ b)


def edit(w, f):
    w = {k: np.array(v, dtype=np.float32, copy=True) for k, v in w.items()}
    x = np.asarray(f['x'], dtype=np.float32)
    y = np.asarray(f['y'], dtype=np.float32)
    t = np.asarray(f['t']).astype(np.int64)
    abar = np.asarray(f['alphabar'], dtype=np.float64)
    N = x.shape[0]

    # ------------------------------------------------------------------
    # Stage 1: fit a per-timestep 1-D piecewise-linear map on x_t:
    #   eps_hat = c * relu(x - theta_t) + b_t
    # plus a static 3x3 local filter on x_t (linear part).
    # ------------------------------------------------------------------
    # per-timestep: regress y on [relu(x-theta), 1] over all pixels of that t
    # theta chosen per t by a small closed-form search over a grid (allowed:
    # statistics / closed-form fitting, no iterative optimizer).
    thetas = np.zeros(32, dtype=np.float64)
    cs = np.zeros(32, dtype=np.float64)
    bs = np.zeros(32, dtype=np.float64)
    for tt in range(32):
        m = (t == tt)
        if m.sum() < 2:
            continue
        xt = x[m].reshape(-1).astype(np.float64)
        yt = y[m].reshape(-1).astype(np.float64)
        best = None
        # grid of candidate thresholds (quantiles of xt) -> closed-form 2-param LS
        cand = np.quantile(xt, np.linspace(0.02, 0.98, 25))
        cand = np.unique(np.concatenate([cand, [0.0, 1.0, -1.0]]))
        for th in cand:
            u = np.maximum(xt - th, 0.0)
            A = np.stack([u, np.ones_like(u)], axis=1)
            G = A.T @ A
            G.flat[::3] += 1e-8
            try:
                sol = np.linalg.solve(G, A.T @ yt)
            except np.linalg.LinAlgError:
                continue
            res = float(np.mean((A @ sol - yt) ** 2))
            if best is None or res < best[0]:
                best = (res, th, sol[0], sol[1])
        if best is not None:
            thetas[tt], cs[tt], bs[tt] = best[1], best[2], best[3]

    # global (static) gain c: average of per-t gains weighted by count, but the
    # architecture can only apply a *static* gain to the relu(x-theta) branch.
    # Use the mean of per-t c's (timesteps are equally weighted in the metric).
    c_static = float(np.mean(cs))

    # static 3x3 local linear filter on x_t (residual after the 1-D map)
    # fit jointly: y ~ c*relu(x-theta_t) + sum_{dy,dx} K[dy,dx]*x[p+dy,dx] + b_t
    # Build design on a subsample of examples for speed, then refine.
    def build_design(idx):
        xs = x[idx]
        ts = t[idx]
        ys = y[idx]
        M = len(idx)
        # 3x3 neighborhood of x
        xp = np.pad(xs, ((0, 0), (0, 0), (1, 1), (1, 1)))
        s = xp.strides
        nb = np.lib.stride_tricks.as_strided(
            xp, shape=(M, 1, 3, 3, 16, 16),
            strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
        nb = nb.reshape(M, 9, 256)
        th = thetas[ts].reshape(M, 1, 1)
        u = np.maximum(xs - th[:, :, None, None], 0.0).reshape(M, 256)
        A = np.concatenate([u, nb.transpose(0, 2, 1).reshape(M, 256, 9)], axis=2)
        A = A.reshape(M * 256, 10)
        B = ys.reshape(M * 256)
        return A, B

    sub = np.arange(0, N, max(1, N // 400))
    A, B = build_design(sub)
    # columns: [u, x9(9)] ; fit with the *static* gain c applied to u:
    #   model = c*u + K.x + b_t  -> solve for [c, K(9), b_t(32)] jointly
    # do it as: for fixed c, residual = B - c*u ; fit K and per-t intercepts.
    # Use alternating closed-form (2 passes) -> still closed-form fitting.
    def fit_K(resid, idx, lam=1e-4):
        # per-t intercept + shared 3x3 filter
        K = np.zeros(9)
        for _ in range(3):
            # solve per-t intercepts given K
            r = resid - A[:, 1:] @ K
            tt = t[idx]
            bt = np.zeros(32)
            for tv in np.unique(tt):
                mm = (tt == tv)
                bt[tv] = r[mm].mean()
            r2 = r - bt[tt][:, None] if False else (r - np.take(bt, tt))
            # solve K from centered data
            G = A[:, 1:].T @ A[:, 1:]
            G.flat[::9] += lam * max(1.0, np.trace(G) / 9)
            rhs = A[:, 1:].T @ r2
            K = np.linalg.solve(G, rhs)
        return K, bt

    c = c_static
    K = np.zeros(9)
    bt = np.zeros(32)
    for _ in range(4):
        resid = B - c * A[:, 0]
        K, bt = fit_K(resid, sub)
        # update c given K, bt
        r = B - A[:, 1:] @ K - np.take(bt, t[sub])
        denom = float(np.sum(A[:, 0] ** 2)) + 1e-8
        c = float(np.sum(A[:, 0] * r) / denom)
    # final per-t intercepts
    resid = B - c * A[:, 0] - A[:, 1:] @ K
    tt = t[sub]
    bt = np.zeros(32)
    for tv in np.unique(tt):
        bt[tv] = resid[tt == tv].mean()

    # ------------------------------------------------------------------
    # Stage 2: realize the fitted map inside the fixed architecture.
    #   target: pred(p) = c * relu(x(p) - theta_t) + sum_{dy,dx} K[dy,dx] x(p+dy,dx) + b_t
    #
    # conv1 (2->32): channel 0 = x (center tap) ; channel 1 = t/31 (center tap)
    #   pre-act ch0 = x(p) - theta_t   -> needs theta_t = a*(t/31)+b0
    #   pre-act ch1 = t/31
    #   ReLU: relu(x-theta_t) and relu(t/31)=t/31 (>=0)
    # conv2 (32->64): build the linear combination
    #   out ch A = relu(x-theta)          (via +1 from conv1 ch0)
    #   out ch B = relu(-(x-theta))       (via -1 from conv1 ch0)  -> but this is <=0
    #   we need the *difference* relu(u)-relu(-u) = u.  Use conv2 to output
    #   both +relu(u) and -relu(-u) as separate channels, then conv3 combines.
    #   conv2 outputs: ch0 = +relu(u) (weight +1 on conv1 ch0), ch1 = -relu(-u)
    #        (weight -1 on conv1 ch0)  -> but conv2 output ch1 <= 0 -> ReLU kills.
    #   Instead: conv2 ch0 = +relu(u) (>=0), conv2 ch1 = +relu(-u) (>=0),
    #        and conv3 computes ch0 - ch1 = u.
    # conv3 (64->64): ch0 = conv2ch0 - conv2ch1 = u ; ch1..ch8 = local x taps
    #        (x taps come from conv1 ch0? no: conv1 ch0 = x-theta, not x.)
    #   -> we need raw x taps: add conv1 channels 2..10 = x at 8 neighbors
    #      (center handled by ch0).  conv1 has 32 channels: use 0 (center x),
    #      1 (t), 2..9 (8 neighbor x taps), rest unused.
    #   conv2: ch0 = +conv1ch0 (relu'd), ch1 = -conv1ch0 -> need relu(-u):
    #           use conv1 ch10 = -(x-theta) so relu gives relu(-u).
    #      conv2 ch0 = +1*conv1ch0, ch1 = +1*conv1ch10, ch2..ch9 = +1*conv1ch2..9
    #      (all conv1 outputs are >=0 after ReLU, so conv2 pre-acts >=0 -> survive)
    #   conv3: ch0 = conv2ch0 - conv2ch1 = u ; ch1..ch8 = conv2ch2..9 (x taps)
    #           (conv3 output must be >=0 for its ReLU: u = relu(x-theta) >=0 ✓,
    #            x taps can be negative ✗) -> use pair trick again for x taps:
    #           conv3 ch1..ch8 = x taps, but they may be negative.
    #      Solution: conv3 outputs ch1..ch8 = taps, and we add the "sign-safe"
    #      trick at the out layer instead: out layer sees relu(conv3out).
    #      => Instead, keep taps nonneg by construction: conv1 ch2..ch9 = relu(x_neighbor)
    #         loses negative info.
    #      Better: carry taps through as *pairs* (v, -v) through conv2/conv3 and
    #      combine at the out layer (out has 64 input channels, 3x3 kernel, so we
    #      can use the out layer's own 3x3 spatial taps to implement the local
    #      filter directly on a single channel!).
    # ------------------------------------------------------------------
    # SIMPLER EXACT CONSTRUCTION:
    #   The out layer is a 3x3 conv over conv3's output.  If conv3's output
    #   channel 0 = u (>=0) and channel 1 = x (raw, may be negative), then out
    #   cannot use negative values (ReLU at conv3 output).  So make conv3 output
    #   channel 1 = relu(x) and channel 2 = relu(-x); then relu(x)-relu(-x)=x
    #   is computed by the OUT layer as: out = c*u_center + K . (taps of x).
    #   Since out is a 3x3 conv over conv3 output, and conv3 output channel 1
    #   holds relu(x) at the same spatial location, the out 3x3 kernel can pick
    #   relu(x) at 9 spatial offsets.  But relu(x)-relu(-x) requires both signs
    #   at the same offset: use conv3 channels 1 (relu(x)) and 2 (relu(-x)),
    #   and out kernel weights w1[dy,dx] = K[dy,dx], w2[dy,dx] = -K[dy,dx].
    #   => out = c * u_center + sum K[dy,dx]*(relu(x)_off - relu(-x)_off) + bias
    #   where u_center = relu(x - theta_t) at the center offset.
    #   And b_t (per-timestep intercept) must come from the t channel:
    #   conv3 channel 3 = t/31 (>=0), out kernel weight on it gives b_t via
    #   center tap: out bias + w3_center * (t/31) + w4_center * (t/31)^2 ...
    #   -> we need b_t as a function of t: use conv1 to produce t/31 and (t/31)^2
    #      (t/31)^2 can be made by conv1 ch1 = t/31 (>=0) then conv2 ch? = conv1ch1 * ...
    #      no, multiplication not available.  Instead: fit b_t with a linear
    #      function of t/31 (1 dof) plus per-t residual handled by a *piecewise*
    #      trick is overkill.  Use: b_t = b0 + b1*(t/31) and additionally allow
    #      the threshold branch to absorb per-t offsets (theta_t is per-t, and
    #      c*relu(x-theta_t) already carries a per-t shift of the intercept).
    #      Actually with theta_t per-t and b_t = b0 + b1*(t/31), the fit is
    #      re-done with this constrained parameterization (closed form per t).
    # ------------------------------------------------------------------
    # Re-fit under the architectural constraint:
    #   pred = c * relu(x - theta_t) + sum K[dy,dx]*x[off] + b0 + b1*(t/31)
    # with theta_t = p0 + p1*(t/31)  (per-t threshold, affine in t)
    # Solve by grid search over (p0,p1) and closed-form LS for (c,K,b0,b1).
    # ------------------------------------------------------------------
    def build_A(idx, th):
        xs = x[idx]; ts = t[idx]
        M = len(idx)
        xp = np.pad(xs, ((0, 0), (0, 0), (1, 1), (1, 1)))
        s = xp.strides
        nb = np.lib.stride_tricks.as_strided(
            xp, shape=(M, 1, 3, 3, 16, 16),
            strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
        nb = nb.reshape(M, 9, 256)
        u = np.maximum(xs - th[:, None, None, None], 0.0).reshape(M, 256)
        tb = (ts / 31.0).astype(np.float64)
        A = np.empty((M * 256, 12), dtype=np.float64)
        A[:, 0] = u.reshape(-1)
        A[:, 1:10] = nb.transpose(0, 2, 1).reshape(M * 256, 9)
        A[:, 10] = 1.0
        A[:, 11] = np.repeat(tb, 256)
        return A

    best = None
    grid_p0 = np.linspace(-1.5, 1.5, 13)
    grid_p1 = np.linspace(-1.5, 1.5, 13)
    for p0 in grid_p0:
        for p1 in grid_p1:
            th = p0 + p1 * (t[sub] / 31.0)
            A = build_A(sub, th)
            sol = _fit_lstsq(A, B, lam=1e-6)
            res = float(np.mean((A @ sol - B) ** 2))
            if best is None or res < best[0]:
                best = (res, p0, p1, sol)
    res, p0, p1, sol = best
    c, K, b0, b1 = sol[0], sol[1:10], sol[10], sol[11]

    # ------------------------------------------------------------------
    # Stage 3: build the weights realizing the model exactly.
    #   conv1 (2->32):
    #     ch0 : w[0,0] = identity center ; w[0,1] = -p1 center ; bias = -p0
    #            -> pre-act = x - (p0 + p1*t/31) = x - theta_t
    #     ch1 : w[1,1] = 1 center ; bias 0            -> pre-act = t/31 (>=0)
    #     ch2 : w[2,0] = -1 center ; w[2,1] = p1 center ; bias = p0
    #            -> pre-act = -(x - theta_t) = theta_t - x
    #     ch3 : w[3,0] = 1 center ; bias 0             -> pre-act = x
    #     ch4 : w[4,0] = -1 center ; bias 0            -> pre-act = -x
    #     ch5 : w[5,1] = 1 center ; bias 0             -> pre-act = t/31 (>=0)
    #   conv2 (32->64): identity-ish copies (all conv1 outputs >=0 after ReLU)
    #     ch0 = conv1 ch0 (relu(x-theta))
    #     ch1 = conv1 ch2 (relu(theta-x))
    #     ch2 = conv1 ch3 (relu(x))
    #     ch3 = conv1 ch4 (relu(-x))
    #     ch4 = conv1 ch1 (t/31)
    #     ch5 = conv1 ch5 (t/31)
    #     rest 0
    #   conv3 (64->64):
    #     ch0 = conv2 ch0 - conv2 ch1  -> u = relu(x-theta) - relu(theta-x) = x-theta
    #           ... careful: we want relu(x-theta) itself for the c*relu branch,
    #           and relu(x)-relu(-x)=x for the taps.
    #     Use: ch0 = conv2 ch0  (relu(x-theta) >= 0)   -> out center tap * c
    #          ch1 = conv2 ch2  (relu(x) >= 0)
    #          ch2 = conv2 ch3  (relu(-x) >= 0)
    #          ch3 = conv2 ch4  (t/31)
    #          ch4 = conv2 ch5  (t/31)
    #   out (64->1): 3x3 kernel
    #          w[0,0] = c at center (on conv3 ch0)  -> c*relu(x-theta) center
    #          w[0,1] = K[dy,dx] (on conv3 ch1 = relu(x))
    #          w[0,2] = -K[dy,dx] (on conv3 ch2 = relu(-x))
    #          w[0,3] = b1 at center (on conv3 ch3 = t/31)
    #          w[0,4] = 0
    #          bias = b0
    #   NOTE: the out 3x3 kernel taps on conv3 ch1/ch2 give spatial offsets of
    #   relu(x) and relu(-x) -> their difference equals x at that offset.
    #   But careful: conv3 ch1 = conv2 ch2 = relu(conv1 ch3) = relu(x) at the
    #   SAME pixel (all conv2/conv3 kernels are identity centers) ✓.
    # ------------------------------------------------------------------
    c1w = np.zeros((32, 2, 3, 3), dtype=np.float32)
    c1b = np.zeros(32, dtype=np.float32)
    c1w[0, 0, 1, 1] = 1.0
    c1w[0, 1, 1, 1] = -p1
    c1b[0] = -p0
    c1w[1, 1, 1, 1] = 1.0
    c1w[2, 0, 1, 1] = -1.0
    c1w[2, 1, 1, 1] = p1
    c1b[2] = p0
    c1w[3, 0, 1, 1] = 1.0
    c1w[4, 0, 1, 1] = -1.0
    c1w[5, 1, 1, 1] = 1.0

    c2w = np.zeros((64, 32, 3, 3), dtype=np.float32)
    c2b = np.zeros(64, dtype=np.float32)
    for o, src in enumerate([0, 2, 3, 4, 1, 5]):
        c2w[o, src, 1, 1] = 1.0

    c3w = np.zeros((64, 64, 3, 3), dtype=np.float32)
    c3b = np.zeros(64, dtype=np.float32)
    c3w[0, 0, 1, 1] = 1.0     # relu(x - theta)
    c3w[1, 2, 1, 1] = 1.0     # relu(x)
    c3w[2, 3, 1, 1] = 1.0     # relu(-x)
    c3w[3, 4, 1, 1] = 1.0     # t/31
    c3w[4, 5, 1, 1] = 1.0     # t/31

    ow = np.zeros((1, 64, 3, 3), dtype=np.float32)
    ob = np.zeros(1, dtype=np.float32)
    ow[0, 0, 1, 1] = c
    for dy in range(3):
        for dx in range(3):
            ow[0, 1, dy, dx] = K[dy * 3 + dx]
            ow[0, 2, dy, dx] = -K[dy * 3 + dx]
    ow[0, 3, 1, 1] = b1
    ob[0] = b0

    w['conv1.weight'] = c1w
    w['conv1.bias'] = c1b
    w['conv2.weight'] = c2w
    w['conv2.bias'] = c2b
    w['conv3.weight'] = c3w
    w['conv3.bias'] = c3b
    w['out.weight'] = ow
    w['out.bias'] = ob

    # ------------------------------------------------------------------
    # Stage 4: verify realized forward pass vs the closed-form model, and
    # report diagnostics.  (No training happens after this.)
    # ------------------------------------------------------------------
    chk = np.arange(0, N, max(1, N // 300))
    pred_new = _fwd(x[chk], t[chk], w)
    mse_new = float(np.mean((pred_new - y[chk]) ** 2))
    pred_old = np.asarray(f['prediction'], dtype=np.float32)[chk]
    mse_old = float(np.mean((pred_old - y[chk]) ** 2))

    # safety: keep all params finite and bounded
    for k in w:
        w[k] = np.clip(np.nan_to_num(w[k], nan=0.0, posinf=0.0, neginf=0.0),
                       -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        'mse_before': mse_old,
        'mse_after': mse_new,
        'fit_resid_sub': res,
        'c_static': float(c),
        'p0': float(p0),
        'p1': float(p1),
        'b0': float(b0),
        'b1': float(b1),
        'K': [float(v) for v in K],
    }
    return {'weights': w, 'diagnostics': diagnostics}
