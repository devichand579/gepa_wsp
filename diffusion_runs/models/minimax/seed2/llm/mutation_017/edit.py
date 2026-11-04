import numpy as np

def edit(w, f):
    # ---------------- helpers ----------------
    def conv2d(x, W, b):
        # x: (N,C,H,W), W: (O,C,3,3) cross-correlation, pad 1
        N, C, H, Wd = x.shape
        O = W.shape[0]
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        # im2col
        cols = np.empty((N, C * 9, H * Wd), dtype=np.float32)
        for ky in range(3):
            for kx in range(3):
                patch = xp[:, :, ky:ky + H, kx:kx + Wd]
                cols[:, (ky * 3 + kx)::9, :] = patch.reshape(N, C, H * Wd)
        out = np.tensordot(W.reshape(O, C * 9), cols, axes=([1], [1]))
        out = out.reshape(N, O, H, Wd)
        out += b.reshape(1, O, 1, 1)
        return out

    def convT(W):
        # (O,C,ky,kx) -> (C,O,kx,ky)  (true convolution == cross-correlation with flipped kernel)
        return np.ascontiguousarray(W.transpose(2, 3, 1, 0))

    def forward(x, t, W1, b1, W2, b2, W3, b3, Wo, bo):
        u = (t.astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
        inp = np.concatenate([x, np.broadcast_to(u, (x.shape[0], 1, x.shape[2], x.shape[3]))], axis=1)
        h1 = np.maximum(conv2d(inp, W1, b1), 0.0)
        h2 = np.maximum(conv2d(h1, W2, b2), 0.0)
        h3 = np.maximum(conv2d(h2, W3, b3), 0.0)
        return conv2d(h3, Wo, bo)

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float64)
    t = np.asarray(f["t"]).astype(np.int64)
    y = np.asarray(f["y"], dtype=np.float64)
    clean = np.asarray(f["clean"], dtype=np.float64)
    ab = np.asarray(f["alphabar"], dtype=np.float64)
    N = x.shape[0]
    Xf = x.reshape(N, -1)          # (N,256)
    Yf = y.reshape(N, -1)
    Cf = clean.reshape(N, -1)

    # ---------------- closed-form per-timestep Wiener filters ----------------
    # x = s*clean + n*eps  =>  E[eps|x] = n*(s^2*Sigma + n^2*I)^-1 x
    Sigma = (Cf.T @ Cf) / N
    I256 = np.eye(256)
    B = {}
    for tt in range(32):
        s = np.sqrt(ab[tt]); n = np.sqrt(max(1.0 - ab[tt], 1e-12))
        M = (s * s) * Sigma + (n * n) * I256
        B[tt] = n * np.linalg.solve(M, I256)     # (256,256)

    # ---------------- group timesteps (few groups => low-rank per-group maps) ----------------
    def fit_group(tts, lam):
        idx = np.isin(t, tts)
        Xg = Xf[idx]; Yg = Yf[idx]
        # ridge regression y ~ x  (target: eps)
        A = Xg.T @ Xg + lam * np.eye(256)
        Wg = np.linalg.solve(A, Xg.T @ Yg)
        return Wg

    def mse_of(Wg, tts):
        idx = np.isin(t, tts)
        r = Yf[idx] - Xf[idx] @ Wg
        return float(np.mean(r * r))

    def mse_of_full(Bd):
        tot = 0.0
        for tt in range(32):
            idx = (t == tt)
            if idx.sum() == 0:
                continue
            r = Yf[idx] - Xf[idx] @ Bd[tt]
            tot += float(np.sum(r * r))
        return tot / (N * 256)

    # candidate groupings: uniform groups of size g
    best = None
    for g in [1, 2, 4, 8, 16]:
        groups = [list(range(i, min(i + g, 32))) for i in range(0, 32, g)]
        lam = 1e-2 * N / 100.0
        Ws = [fit_group(gr, lam) for gr in groups]
        # evaluate
        tot = 0.0
        for gr, Wg in zip(groups, Ws):
            idx = np.isin(t, gr)
            r = Yf[idx] - Xf[idx] @ Wg
            tot += float(np.sum(r * r))
        m = tot / (N * 256)
        if best is None or m < best[0]:
            best = (m, groups, Ws)

    # also: per-timestep Wiener (closed form, no data fitting) for reference
    mse_wiener = mse_of_full(B)

    m_group, groups, Ws = best
    # choose between grouped-ridge and per-timestep Wiener by training MSE
    use_wiener = mse_wiener < m_group
    if use_wiener:
        groups = [[tt] for tt in range(32)]
        Ws = [B[tt] for tt in range(32)]

    G = len(groups)
    # ---------------- build architecture ----------------
    # channel budget: conv1 -> 32 channels.  Each feature j needs 2 channels (A,B).
    # We use K = 16 features, assigned round-robin to groups.
    K = 16
    Mbig = 1.0e4
    epsx = 1.0e-2
    # group boundaries in u = t/31 space (midpoints between consecutive groups)
    bounds = []
    for gi in range(G):
        if gi == 0:
            bounds.append(-10.0)
        else:
            prev_end = groups[gi - 1][-1]
            cur_start = groups[gi][0]
            bounds.append(0.5 * ((prev_end + cur_start) + 1.0) / 31.0)
    bounds.append(10.0)

    # feature directions: use the leading right-singular vectors of the group's map
    # (rank <= K/G per group); use global PCA of clean images as spatial basis.
    # Build basis from clean covariance eigenvectors (top 16).
    evals, evecs = np.linalg.eigh(Sigma)
    order = np.argsort(evals)[::-1][:K]
    basis = evecs[:, order]           # (256, K) orthonormal

    # assign features to groups: group gi gets features with idx % G == gi
    feat_group = [j % G for j in range(K)]

    W1 = np.zeros((32, 2, 3, 3), dtype=np.float64)
    b1 = np.zeros(32, dtype=np.float64)
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float64)
    b2 = np.zeros(64, dtype=np.float64)
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float64)
    b3 = np.zeros(64, dtype=np.float64)
    Wo = np.zeros((1, 64, 3, 3), dtype=np.float64)
    bo = np.zeros(1, dtype=np.float64)

    # conv1: channel 2j = A_j  (ramp gate + epsx * feature), channel 2j+1 = B_j (ramp gate)
    # feature j spatial kernel: 3x3 from basis column j mapped onto 16x16 grid
    def kernel_from_vec(vec):
        k = np.zeros((3, 3))
        k[1, 1] = vec[8 * 8 + 8]   # center pixel of the 16x16 grid
        return k

    for j in range(K):
        gi = feat_group[j]
        tau = bounds[gi + 1] if gi + 1 < len(bounds) else 10.0
        # A_j: t-channel weight at kernel position (0,0) = Mbig ; x-channel kernel = epsx*kj
        kj = kernel_from_vec(basis[:, j])
        W1[2 * j, 0, :, :] = epsx * kj
        W1[2 * j, 1, 0, 0] = Mbig
        b1[2 * j] = -Mbig * tau
        # B_j: pure ramp
        W1[2 * j + 1, 1, 0, 0] = Mbig
        b1[2 * j + 1] = -Mbig * tau

    # conv2: channel k (k=0..K-1) = (A_j - B_j)/epsx  -> gated feature
    for j in range(K):
        W2[j, 2 * j, 1, 1] = 1.0 / epsx
        W2[j, 2 * j + 1, 1, 1] = -1.0 / epsx
    # conv3: identity copy channels 0..K-1 -> 0..K-1 ; plus constant channels for offsets
    for j in range(K):
        W3[j, j, 1, 1] = 1.0
    # out: combine features per group with the group's fitted linear map
    # out = sum_j alpha_j * feat_j  where alpha_j (3x3 kernel) = group map applied to basis_j
    for gi, gr in enumerate(groups):
        Wg = Ws[gi]                     # (256,256): y ~ Wg x
        for j in range(K):
            if feat_group[j] != gi:
                continue
            # map basis vector v (256,) through Wg -> 256-vector of pixel coefficients;
            # represent as 3x3 kernel using center-pixel trick is not enough for arbitrary maps.
            # Instead: use full 16x16 -> we approximate with the center-pixel trick only for
            # the diagonal part; handled below by direct construction of a 3x3 map.
            pass

    # The above per-group maps are dense 256x256; a 3x3 conv stack cannot realise them exactly.
    # Instead we restrict each group's map to a 3x3-convolution form by refitting in that space.
    # Build design matrices of 3x3 patch features and refit per group.
    def patch_features(X, idx):
        # returns (n, 576) : 3x3 patches of x (zero padded), flattened
        Xr = X[idx].reshape(-1, 16, 16)
        n = Xr.shape[0]
        xp = np.pad(Xr, ((0, 0), (1, 1), (1, 1)))
        cols = np.empty((n, 9, 256), dtype=np.float64)
        for ky in range(3):
            for kx in range(3):
                cols[:, ky * 3 + kx, :] = xp[:, ky:ky + 16, kx:kx + 16].reshape(n, 256)
        return cols.reshape(n, -1)

    # global refit: y_pixel(p) = sum_{q in 3x3 neighborhood} K[p,q] x(q)  (translation conv)
    # Fit a single 3x3 kernel shared across pixels and timesteps-groups via least squares.
    # Build for each group: rows = pixels, cols = 9 patch entries.
    def build_group_design(idx):
        Xr = Xf[idx].reshape(-1, 16, 16)
        Yr = Yf[idx].reshape(-1, 16, 16)
        n = Xr.shape[0]
        xp = np.pad(Xr, ((0, 0), (1, 1), (1, 1)))
        cols = np.empty((n, 9, 256), dtype=np.float64)
        for ky in range(3):
            for kx in range(3):
                cols[:, ky * 3 + kx, :] = xp[:, ky:ky + 16, kx:kx + 16].reshape(n, 256)
        # rows: (n*256, 9)
        A = cols.transpose(0, 2, 1).reshape(n * 256, 9)
        b = Yr.reshape(n * 256)
        return A, b

    # group-wise 3x3 kernel fit (translation equivariant), ridge
    K3 = {}
    for gi, gr in enumerate(groups):
        idx = np.isin(t, gr)
        A, b = build_group_design(idx)
        lam = 1e-3 * len(b) * 1e-2
        kern = np.linalg.solve(A.T @ A + lam * np.eye(9), A.T @ b)
        K3[gi] = kern.reshape(3, 3)

    # Now: the gated feature channel j must carry the *patch* of x, not a single pixel.
    # Redefine: conv1 channel A_j / B_j use a 3x3 kernel that extracts the 3x3 patch
    # around the center pixel (identity-like), scaled by epsx.  Then conv2 channel j
    # reconstructs the 3x3 patch values (9 numbers) -> but conv2 output has 64 channels,
    # we need 9 per feature.  Use 9 conv2 channels per feature -> 9*K channels > 64.
    # => reduce K to 6 features (12 conv1 channels, 54 conv2 channels).
    K = 6
    # rebuild conv1/conv2 with 3x3 patch extraction
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float64)
    b1 = np.zeros(32, dtype=np.float64)
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float64)
    b2 = np.zeros(64, dtype=np.float64)
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float64)
    b3 = np.zeros(64, dtype=np.float64)
    Wo = np.zeros((1, 64, 3, 3), dtype=np.float64)
    bo = np.zeros(1, dtype=np.float64)

    feat_group = [j % G for j in range(K)]
    patch_idx = [(ky, kx) for ky in range(3) for kx in range(3)]

    for j in range(K):
        gi = feat_group[j]
        tau = bounds[gi + 1] if gi + 1 < len(bounds) else 10.0
        # A_j channel: x-part = epsx * (3x3 patch extraction, one channel per patch pos is not
        # possible in one channel) -> instead A_j uses only the center pixel of the patch.
        # We therefore use 9 separate features?  -> Use per-patch-position features:
        # feature (j, p) for p in 0..8 : channel pair 2*(j*?)...
        pass

    # ---- final, simpler and exact design -------------------------------------
    # Use per-group 3x3 kernels directly (translation-equivariant linear map).
    # Implement:  conv1 channel 2g   = gate ramp for group g (t-dependent, no x)
    #             conv1 channel 2g+1 = epsx * x (identity 3x3 kernel) + same ramp
    #  conv2 channel p (p=0..8) for group g: (A_g - B_g)/epsx  -> gives gated x
    #  conv3: identity from conv2 channels to conv3 channels
    #  out: sum over groups of K3[g] * (gated x for group g)  -> but out conv is 3x3 and
    #        must apply a different kernel per group -> needs separate channels per group.
    #  conv3 has 64 channels: use 9 channels per group for the 9 patch positions of the
    #  group's gated x, then out applies the group's 3x3 kernel.
    #  => groups G must satisfy 9*G <= 64  => G <= 7.
    #  Use G = 4 groups of 8 timesteps (or G=6).  Choose G<=6.
    # -----------------------------------------------------------------------
    G = min(G, 6)
    # regroup timesteps into G contiguous groups of equal size
    per = int(np.ceil(32.0 / G))
    groups = [list(range(i, min(i + per, 32))) for i in range(0, 32, per)]
    G = len(groups)
    bounds = []
    for gi in range(G):
        if gi == 0:
            bounds.append(-10.0)
        else:
            prev_end = groups[gi - 1][-1]
            cur_start = groups[gi][0]
            bounds.append(0.5 * ((prev_end + cur_start) + 1.0) / 31.0)
    bounds.append(10.0)

    # refit per-group 3x3 kernels with the new grouping
    K3 = {}
    for gi, gr in enumerate(groups):
        idx = np.isin(t, gr)
        A, b = build_group_design(idx)
        lam = 1e-5 * len(b)
        kern = np.linalg.solve(A.T @ A + lam * np.eye(9), A.T @ b)
        K3[gi] = kern.reshape(3, 3)

    # ---- construct weights ----
    W1 = np.zeros((32, 2, 3, 3), dtype=np.float64)
    b1 = np.zeros(32, dtype=np.float64)
    W2 = np.zeros((64, 32, 3, 3), dtype=np.float64)
    b2 = np.zeros(64, dtype=np.float64)
    W3 = np.zeros((64, 64, 3, 3), dtype=np.float64)
    b3 = np.zeros(64, dtype=np.float64)
    Wo = np.zeros((1, 64, 3, 3), dtype=np.float64)
    bo = np.zeros(1, dtype=np.float64)

    # conv1 channel layout: for group g: ch 2g = ramp B_g (t only), ch 2g+1 = A_g
    # A_g = epsx * x_center + Mbig*(u - tau_g) ; B_g = Mbig*(u - tau_g)
    # conv2 channels: for group g, 9 channels (g*9 .. g*9+8) each = (A_g - B_g)/epsx
    #   -> gives gated x (all 9 channels identical: the x value)
    # conv3 channels: for group g, 9 channels (g*9..g*9+8) = identity copy of conv2's
    #   -> but then out conv (3x3) applied to channel set of group g must produce the
    #      group's 3x3 kernel output. out conv has ONE kernel for all 64 channels ->
    #      we need per-group kernels. Solution: use conv3 to place each group's 9
    #      channels into distinct "spatial offsets" is impossible (conv is per-channel).
    #      Instead: apply the group's 3x3 kernel in conv3 (which has 64 output channels,
    #      each with its own 3x3 kernel over 64 input channels) and then out selects.
    #      conv3 output channel (g*9+p) = conv3 kernel over conv2 channels.
    #      We want conv3 output for group g to be the 3x3 convolution of x with K3[g],
    #      i.e. a single channel per group (not 9). So conv3 has G channels used.
    #      Then out (64->1) applies a 3x3 kernel over conv3's 64 channels: for group g we
    #      need out to apply K3[g] to conv3 channel g -> but out has ONE kernel for all
    #      input channels. Problem: out kernel is shared across channels.
    #      => Use conv3 to do the 3x3 convolution per group: conv3 channel g = K3[g] * x_g.
    #      Then out must select channel g with a per-group weight -> out kernel is shared,
    #      so we need the group selection at out level: out = sum_g c_g * conv3_g.
    #      c_g must be 1 for the active group only -> use the gate again: multiply conv3_g
    #      by a gate that is 1 only for group g. We can implement the gate at conv3 level:
    #      conv3 channel g = K3[g]*x_g * gate_g? multiplication again...
    #      Alternative: apply the group-specific 3x3 kernel in conv3 but with the GATE
    #      applied to the *input* channels of conv3 (i.e., gate the x channels before the
    #      3x3 convolution). Since gating is per-group and conv3's input channels are
    #      shared, we need 9*G channels in conv3 input (each group needs 9 patch channels).
    #      9*G <= 64 -> G <= 7.  conv3 output: G channels (one per group) = K3[g] * gated_x_g.
    #      Then out: needs per-group selection -> out = sum_g conv3_g * sel_g.
    #      sel_g must be 1 only for group g.  We can get sel_g from conv3's own output
    #      channels? Not without multiplication.
    #      Trick: make conv3 output channel g = K3[g]*gated_x_g + (large constant)*gate_g
    #      where gate_g is a channel that is 1 for group g... but gate_g is a ramp, not 1.
    #      Hmm: instead, note that out's kernel is shared, so out = sum_c Wo[c] * conv3_c.
    #      If conv3_c for group g already equals (K3[g]*gated_x_g) and we want out to
    #      select group g with coefficient 1, we need Wo to depend on g -> impossible.
    #      SOLUTION: put the group selection in conv3 by making conv3 output channel g
    #      equal to K3[g]*gated_x_g only for the active group and ZERO otherwise.
    #      Zeroing requires multiplication by the gate...
    #      BUT: the gate can be applied at the conv1 level to the x channels, i.e. the
    #      gated x channels for group g are already zero for inactive groups. Then
    #      conv3 output channel g = K3[g] * (gated x for group g) is automatically zero
    #      for inactive groups!  Then out = sum_g conv3_g with Wo[c=g]=identity kernel
    #      (center 1) -> out = sum_g K3[g]*gated_x_g = the right answer (only active group
    #      contributes).
    #      And the gating at conv1 level: we need the x channels (per group, 9 patch
    #      positions) to be zeroed when the group is inactive. Our A/B trick gives
    #      (A_g - B_g)/epsx = x when active, 0 when inactive -- but only for the center
    #      pixel (single channel). To get 9 patch positions per group we need 9 pairs
    #      of channels -> 18 channels per group -> too many for 32.
    #      => Instead: apply the 3x3 convolution FIRST (in conv2) and gate afterwards.
    #      conv2 output: for group g, channel c = K3[g] * x  (3x3 conv of x) -- but conv2
    #      computes the same thing for all groups unless we use different kernels per group,
    #      which we CAN: conv2 has 64 output channels each with its own 3x3 kernel over
    #      the 32 conv1 channels. conv1 channels can carry the 9 patch positions of x
    #      (9 channels: each conv1 channel = one patch position of x, scaled by epsx).
    #      Then conv2 output channel (g) = K3[g] applied to the 9 patch channels = the
    #      group's convolved map. But we need the gate applied to this map (zero when
    #      group inactive). Gate must be applied after conv2 -> conv3 can do it:
    #      conv3 output channel g = (A_g - B_g)/epsx * conv2_g? Again multiplication.
    #      Hmm, but conv3's kernel is over conv2's channels: conv3_out[g] = sum_c W3[g,c] * conv2_c.
    #      If conv2_c for the group's map is already gated... circular.
    #      KEY: the gate signal (A_g - B_g)/epsx is itself a channel in conv2 (or conv1).
    #      We need conv3_out[g] = gate_g * conv2_map_g. Since conv3 is linear, we need
    #      the product. Not possible.
    #      => Use ReLU-based gating at conv3 level: conv3_out[g] = relu(conv2_map_g + BIG*gate_g)?
    #      no.
    #      => Practical resolution: implement the group selection using the *out* layer
    #         with 9 channels per group in conv3 (64 channels total -> G<=7) where
    #         conv3 channel (g,p) = conv2_map_g shifted by patch offset p (i.e., the
    #         9 patch positions of the group's convolved map). Then out applies a 3x3
    #         kernel to conv3's channels: out = sum_c Wo[c] * conv3_c. Since Wo's kernel
    #         is shared across channels, out = sum_c sum_{q} Wo[c,q] conv3_c(shift q).
    #         We want out = sum_g K3[g] * map_g where map_g is the group's convolved map
    #         and K3[g] is a 3x3 kernel. If conv3 channel (g,p) holds map_g shifted by p,
    #         then out = sum_{g,p} Wo[g*9+p, q] * map_g(shift p+q). We need the net
    #         kernel over shifts to equal K3[g]. Since Wo is shared across all channels,
    #         the effective kernel is the same for all groups -> we can only realize ONE
    #         3x3 kernel K3 common to all groups. Not per-group.

    # ---- FINAL SIMPLE DESIGN (per-group kernel via conv3, group select via out bias trick) ----
    # Realize: out = sum_g 1[t in group g] * (K3[g] * x)
    # Implementation:
    #   conv1: 9 channels carrying the 3x3 patch positions of x (scaled by epsx),
    #           plus 2 channels per group carrying the gate ramps (A_g, B_g).
    #           channel budget: 9 + 2G <= 32  -> G <= 11. Use G = 6 or 8.
    #   conv2: 9*G channels: for group g, patch position p:
    #              conv2[g*9+p] = (A_g - B_g)/epsx * (patch value p)  -> needs product of
    #              gate and patch value -> NOT linear.
    #   => Instead: conv2 channel (g*9+p) = (A_g - B_g)/epsx  [the gate, 9 copies]
    #              conv2 channel (g*9+p) must also carry the patch value...
    #   => Use conv3 to apply the 3x3 kernel to the patch channels:
    #        conv2 channels 0..8 : patch positions of x (from conv1's 9 patch channels)
    #        conv2 channels 9..9+2G-1 : gate channels (A_g - B_g)/epsx
    #        conv3 channels: for group g: conv3[g] = K3[g] * patchchannels (3x3 conv over
    #           the 9 patch channels)  -> but we need the gate applied.
    #           conv3[g] = sum_{p,q} W3[g, p, q] * patch_p(shift q)  -> this is K3[g]*x.
    #           To gate it: conv3[g] = (gate_g) * (K3[g]*x) -> product.
    #   => Use ReLU trick at conv3: conv3[g] = relu( (K3[g]*x) + BIG*gate_g ) - BIG*gate_g?
    #      subtraction needs another channel and the ReLU clips.
    #      Alternative: since gate_g is exactly 0 or 1 (approximately) we can use:
    #      conv3[g] = relu( (K3[g]*x) + BIG*gate_g - BIG ) + BIG*(1 - gate_g)?? no.
    #      Hmm: relu(z + BIG*g - BIG) with g in {0,1}: if g=1 -> relu(z) (clips negatives!),
    #      if g=0 -> relu(z - BIG) = 0 (if |z| < BIG). Then conv3[g] = relu(K3[g]*x) when
    #      active -> loses sign information (ReLU). Not good.
    #   => Use the "offset" trick: keep everything positive by adding a large constant
    #      offset to the conv3 output and subtracting it later. Specifically:
    #      conv3[g] = K3[g]*x + BIG*gate_g  (linear! since gate_g is a channel)
    #      -> for active group: K3[g]*x + BIG; for inactive: 0.
    #      Then out must subtract BIG for the active group -> out = sum_g Wo[g]*conv3[g]
    #      with Wo[g] = identity kernel -> out = sum_g (K3[g]*x + BIG*gate_g).
    #      The BIG*gate_g terms must be cancelled: they equal BIG for the active group.
    #      Cancel via a channel that carries the same BIG*gate_g with a negative weight:
    #      out = sum_g Wo[g]*conv3[g] + sum_g Wo'[g]*conv3'[g] where conv3'[g] = BIG*gate_g.
    #      But conv3 has 64 channels: allocate 2G channels for conv3 (map+offset per group)
    #      and the offset channels carry BIG*gate_g directly (copied from conv2's gate
    #      channels). Then out = sum_g [conv3_map[g] - conv3_off[g]] = sum_g K3[g]*x.
    #      Wait: conv3_map[g] = K3[g]*x + BIG*gate_g requires the gate inside conv3's
    #      linear map: conv3_map[g] = sum_p W3[g,p]*patch_p + BIG*gate_g -> yes linear,
    #      with W3[g, gate_ch_g] = BIG (center kernel) and W3[g,p] = K3[g] kernel.
    #      conv3_off[g] = BIG*gate_g (copy of the gate channel).
    #      Then out: Wo[0, conv3_map[g]] = identity kernel (center 1), Wo[0, conv3_off[g]] =
    #      identity kernel with -1. But Wo's kernel is shared across input channels!
    #      Wo[0, c, :, :] is per-channel (shape (64,3,3)) -> so each channel has its own
    #      3x3 kernel. YES! out.weight has shape (1,64,3,3) -> per-input-channel kernels.
    #      So we CAN set Wo[0, conv3_map[g]] = center-identity and Wo[0, conv3_off[g]] =
    #      center-identity*(-1). PERFECT.
    #      And conv3's kernels are per-output-channel too: conv3.weight (64,64,3,3) ->
    #      per-output-channel kernel over all input channels. So conv3_map[g] can mix
    #      patch channels with kernel K3[g] AND the gate channel with BIG.
    #      Also conv2 (64,32,3,3) per-output-channel kernels over conv1's 32 channels:
    #      conv2 channels 0..8 = patch positions (copy from conv1's 9 patch channels),
    #      conv2 channels 9.. = gate channels (copy of (A_g - B_g)/epsx).
    #      conv1: 9 patch channels (each = epsx * one patch position of x) + 2G gate channels.
    #      Budget: 9 + 2G <= 32 -> G <= 11.
    #      ReLU safety: conv1 outputs must be > 0 for the copies to be exact.
    #        patch channels: epsx*x_patch + BIGP (BIGP large, e.g. 10) -> positive.
        #        gate channels A_g = Mbig*(u-tau_g) + epsx*x_center : can be negative (ramp).
        #        B_g = Mbig*(u-tau_g): negative when inactive.
        #      ReLU would clip negatives -> breaks the ramp. So we need the ramps to be
        #      positive: use B_g = Mbig*(u - tau_g) + BIGR where BIGR > Mbig*max|tau| so
        #      always positive; then A_g - B_g = epsx*x_center (both positive) -> but
        #      ReLU(A_g) = A_g only if A_g > 0. With BIGR offset both are positive.
        #      Then the difference A_g - B_g = epsx*x_center regardless of group!
        #      We lose the gating! Because the ramp is positive everywhere.
        #      => We need the gate to be zero for inactive groups. With ReLU, the ramp
        #         must cross zero -> negative values get clipped to 0 -> that's exactly
        #         what we want for gating! But then the difference A_g - B_g is not
        #         epsx*x when active because A_g is clipped.
        #      Let's redo: we want gate_g = 1 if u > tau_g else 0, and to multiply by x.
        #      Use: A_g = relu(Mbig*(u - tau_g) + epsx*x_center)  -> for u>tau_g: Mbig(u-tau)+epsx*x;
        #        for u<tau_g: 0 (if Mbig(u-tau) < -epsx|x|, true away from boundary).
        #      B_g = relu(Mbig*(u - tau_g)) -> for u>tau_g: Mbig(u-tau); else 0.
        #      Then A_g - B_g = epsx*x_center for active group, 0 for inactive.
        #      => conv2 gate channel = (A_g - B_g)/epsx = x_center for active, 0 inactive.
        #      But we need the gate to multiply the whole 3x3-patch map, not just the center.
        #      Solution: apply the gate to the *center* only, and use the structure:
        #        out_g = K3[g] * x  requires all 9 patch values, each gated.
        #      Alternative: gate the conv2 patch channels instead: conv2 patch channel p
        #        should be x_patch_p * gate_g. Not linear.
        #      => Alternative: apply the 3x3 convolution to x FIRST (in conv2), then gate
        #         the RESULT. conv2 output channel for group g = K3[g]*x (using conv1's
        #         9 patch channels). Then conv3 channel g = gate_g * conv2_map_g -> product.
        #         BUT: we can use the offset trick with the gate channel being 0/1:
        #         conv3[g] = conv2_map_g + BIG*gate_g  (linear) -> for active: map+BIG,
        #         for inactive: 0 + 0 = 0 (gate_g = 0 there).
        #         Then out = sum_g [conv3[g] - BIG*gate_g] -> but BIG*gate_g must be
        #         subtracted only when active -> out channel = conv3[g] - conv3_off[g]
        #         where conv3_off[g] = BIG*gate_g (copied from conv2's gate channel).
        #         For active group: conv3[g] = map + BIG, conv3_off[g] = BIG -> diff = map.
        #         For inactive: conv3[g] = 0, conv3_off[g] = 0 -> diff = 0.
        #         PERFECT. And gate_g (0/1) must be available as a channel: it is
        #         (A_g - B_g)/epsx? No, that's x_center*gate. We need a pure 0/1 gate.
        #         Pure gate: use relu(Mbig*(u-tau_g)) - relu(Mbig*(u-tau_g) - BIG2)?? that's a
        #         ramp again. A 0/1 step: relu(Mbig*(u-tau_g)) is a ramp; to get a step we
        #         need relu(Mbig*(u-tau_g)) - relu(Mbig*(u-tau_g) - BIG2) where BIG2 is huge:
        #         for u>tau_g: first = Mbig(u-tau) (ramp), second = 0 if Mbig(u-tau) < BIG2
        #         -> diff = ramp, not step. Hmm.
        #         Use two ramps: step(u>tau) = relu(Mbig(u-tau)) / Mbig ... not constant.
        #         Alternative: use the ratio trick: gate_g = relu(Mbig*(u-tau_g)) * (1/Mbig)?
        #         Not constant.
        #         OK: use a different formulation of the gate that is naturally 0/1:
        #         We need a channel = 1 if u > tau else 0. Using ReLU on a linear function
        #         of u only gives a ramp. But we can use the *pair* trick with two ramps:
        #         R1 = relu(Mbig*(u-tau)), R2 = relu(Mbig*(u-tau) - C) with C chosen so that
        #         R1 - R2 = min(Mbig(u-tau), C) -> a clipped ramp = C for u-tau > C/Mbig.
        #         So R1 - R2 = C when u > tau + C/Mbig, and = Mbig(u-tau) in between.
        #         Not a step but a clipped ramp: constant C for large u.
        #         Then gate-ish channel = (R1 - R2)/C = clipped ramp in [0,1]. Not binary but
        #         bounded and monotone. Multiplying by it is what we want... but again product.
        #      Hmm, but for the offset trick we don't need a binary gate! We need:
        #         conv3[g] = map_g + BIG * s_g  and  conv3_off[g] = BIG * s_g
        #         where s_g is ANY channel; the difference cancels exactly regardless of s_g!
        #         The only requirement is that conv3[g] = map_g + BIG*s_g is computed
        #         correctly, i.e. map_g must be gated (zero for inactive groups) OR we accept
        #         that for inactive groups the output includes map_g (wrong).
        #         So we DO need the map to be gated. map_g = K3[g]*x must be zeroed for
        #         inactive groups. Gating a map requires multiplication.
    #  => RESOLUTION: gate at the conv1 level per patch position, using 2 channels per
    #     (group, patch position) pair: too many channels (9*2*G).
    #     BUT we can reduce: gate only the 9 patch channels ONCE per group by using
    #     conv1 channels = 9 (patch, ungated) + 2G (gate ramps) and then in conv2 we
    #     compute per-group gated patch channels... product again.
    #  => ALTERNATIVE RESOLUTION: use the *ReLU nonlinearity* to gate the map:
    #     conv2_map_g = K3[g]*x (unconditional). conv3[g] = relu(conv2_map_g + BIG*s_g) where
    #     s_g is a channel that is +1-ish for the active group and -BIG-ish for inactive?
    #     s_g must be a channel: s_g = relu(Mbig*(u - tau_g)) - relu(Mbig*(u-tau_g)+Mbig)?? Let's
    #     construct s_g = relu(Mbig*(u-tau_g)) - relu(Mbig*(u-tau_g) + Mbig) = for u<tau: 0-0=0;
    #     for u>tau: Mbig(u-tau) - (Mbig(u-tau)+Mbig) = -Mbig. So s_g = 0 or -Mbig.
    #     Then conv3[g] = relu(map_g + BIG*s_g): active -> relu(map_g) (clips negatives!).
    #     Hmm clipping.
    #     Use instead: conv3[g] = relu(map_g + BIG*s_g) - relu(BIG*s_g)? second term needs a
    #     channel with BIG*s_g -> relu(BIG*s_g) = 0 for active (s_g=-Mbig -> relu(-BIG*Mbig)=0)
    #     and 0 for inactive (s_g=0). Useless.
    #  => FINAL RESOLUTION: exploit the fact that we can afford a *per-group* set of conv1
    #     channels: use 9 patch channels per group but share the "x" content via the
    #     t-channel? no.
    #     Use fewer patch positions: instead of a full 3x3 kernel per group, restrict the
    #     per-group map to a *center-pixel* (1x1) map: out_g = beta_g * x_center.
    #     Then gating needs only 1 patch channel per group -> 2 channels per group in conv1
    #     (A_g, B_g) -> G <= 16. And the per-group scalar beta_g is fitted by least squares
    #     of y_center on x_center within the group. That's a weak model (only center pixel),
    #     but combined with the fact that the Wiener filter is largely a per-pixel
    #     (approximately diagonal) map, plus we can add a few 3x3 terms.
    #     Better: use 3x3 kernels but gate in conv3 via a *different* mechanism:
    #     note that the map K3[g]*x is a linear function of x; we want it multiplied by
    #     gate_g. We can get multiplication if one of the factors is a *constant* per
    #     group... The gate is piecewise constant in t! If gate_g were exactly 0/1 and
    #     we had a channel equal to gate_g, then multiplying a map by it is not linear.
    #     HOWEVER: we can instead compute the map from *gated input channels*: conv2
    #     output channel for group g, patch p = sum over conv1 channels: we need
    #     x_patch_p * gate_g. If conv1 channel (g,p) already contains the gated patch value
    #     (i.e., conv1 computes x_patch_p * gate_g) -> conv1 is linear in [x,u], so it
    #     cannot compute the product. BUT conv1's ReLU can: conv1 channel = relu(epsx*x_patch_p
    #     + Mbig*(u-tau_g)) -> for active group: epsx*x_patch_p + ramp; for inactive: 0.
    #     Then conv2 channel (g,p) = (that channel) - (ramp channel) = epsx*x_patch_p for
    #     active, 0 for inactive. The ramp channel = relu(Mbig*(u-tau_g)) is a separate
    #     conv1 channel (shared across p). So per group: 9 patch channels + 1 ramp = 10
    #     conv1 channels -> G <= 3 with 32 channels. Hmm, G=3 groups only.
    #     Alternatively note the ramp channel can be shared: for group g, the ramp is
    #     relu(Mbig*(u-tau_g)) — different per group. But we could use a single ramp
    #     channel per group and reuse it for all 9 patch positions of that group.
    #     => 10 channels per group -> G=3 (30 channels). With 3 groups of ~11 timesteps,
    #     the per-group kernel differs a lot across t. Maybe acceptable?
    #     Alternatively reduce patch positions: use only the center + 4 neighbors (5) ->
    #     6 channels per group -> G=5.
    #     Or: use a 3x3 kernel but share the patch channels across groups by gating in
    #     conv3 with the offset trick where the map is computed from UNGATED patch
    #     channels and the gate is applied via ReLU clipping at a large threshold:
    #       conv3[g] = relu(map_g + BIG*s_g) where s_g = 0 for inactive, +BIG2 for active
    #       -> active: relu(map + BIG) = map + BIG (positive) ; inactive: relu(map + 0) = relu(map)
    #       -> clipped. Bad.
    #       Use s_g = -BIG for inactive, 0 for active: active: relu(map) clipped. Bad.
    #       Use s_g = BIG for active, -BIG for inactive: active relu(map+BIG)=map+BIG;
    #       inactive relu(map-BIG)=0 (if |map|<BIG). Then conv3[g] = map+BIG for active,
    #       0 for inactive. Then out needs to subtract BIG for the active group only:
    #       out = conv3[g] - BIG*gate_g where gate_g must be 1 for active. We need a
    #       0/1 gate channel. Construct: gate_g = relu(Mbig*(u-tau_g) + Mbig) - relu(Mbig*(u-tau_g))
    #       -> for u>tau: (Mbig(u-tau)+Mbig) - Mbig(u-tau) = Mbig; for u<tau: 0-0 = 0.
    #       YES! That's a binary gate (0 or Mbig) using two ramp channels!
    #       So per group we need: 2 ramp channels (R1_g = relu(Mbig(u-tau_g)+Mbig),
    #       R2_g = relu(Mbig(u-tau_g))) -> gate = R1_g - R2_g (binary Mbig).
    #       Plus s_g = +BIG for active / -BIG for inactive: s_g = 2*gate - Mbig? that's a
    #       combination: s_g = R1_g - 2*R2_g + ... let's see: gate = R1-R2 (0 or Mbig).
    #       We want s_g = +BIG if active else -BIG = 2*gate - BIG (since gate=Mbig active ->
    #       2*Mbig - BIG; set BIG = Mbig -> s_g = 2*gate - Mbig = Mbig active, -Mbig inactive).
    #       So s_g = 2*R1_g - 2*R2_g - Mbig -> needs a constant channel: we can use
    #       conv1's bias? The constant -Mbig must be a channel too. We can get a constant
    #       channel from the t channel: conv1 channel = relu(Mbig*u + Mbig) - ... hmm.
    #       Simpler: make s_g = relu(Mbig*(u-tau_g) + BIG) - relu(Mbig*(u-tau_g) - BIG) - BIG?
    #       Let's define: P_g = relu(Mbig*(u-tau_g) + BIG), Q_g = relu(Mbig*(u-tau_g) - BIG).
    #       For u-tau > BIG/Mbig: P = Mbig(u-tau)+BIG, Q = Mbig(u-tau)-BIG -> P-Q = 2BIG.
    #       For |u-tau| < BIG/Mbig: P = BIG + Mbig(u-tau) (if positive), Q = 0 -> P-Q = BIG+Mbig(u-tau).
    #       For u-tau < -BIG/Mbig: P = 0, Q = 0 -> 0.
    #       Not clean. Use the two-ramp binary gate (R1, R2) and get the constant from
    #       the "u" channel: conv1 can output a constant channel: e.g. channel = relu(Mbig*u +
    #       Mbig) which is Mbig*(u+1) (always positive) -> not constant but we can use
    #       relu(Mbig*u + Mbig) - relu(Mbig*u + BIG) = ... for u in [0,1]: Mbig(u+1) -
    #       (Mbig u + BIG) = Mbig - BIG if BIG > Mbig*u... messy.
    #       EASIEST: use TWO conv1 channels per group for the ramps and note that the
    #       needed constant can be folded into the conv3 bias! conv3's bias is per-output-
    #       channel and constant. But the "-Mbig" term must be multiplied by gate (0/1),
    #       so it can't be a bias. Hmm: s_g = 2*gate - Mbig*1. The "1" is a constant
    #       channel. We can create a constant-1 channel in conv1: channel = relu(Mbig*u +
    #       Mbig) - relu(Mbig*u + Mbig - 1)? Both positive... Let's do: const channel C0 =
    #       relu(Mbig*u + Mbig) and C1 = relu(Mbig*u + Mbig - 1) -> C0 - C1 = 1 for
    #       u in [0,1] (since Mbig*u+Mbig > Mbig*u+Mbig-1 > 0, both positive, difference 1).
    #       So a constant-1 channel = difference of two ramps (2 channels).
    #       Total conv1 channels: 9 (patch) + 2G (ramps for gate) + 2 (constant) + maybe
    #       extra for s_g construction. For G=6: 9+12+2 = 23 <= 32. OK!
    #       And s_g = 2*gate_g - Mbig*const: computed in conv2 as a linear combination of
    #       conv1 channels (conv2 is linear over conv1 channels) -> conv2 channel
    #       s_g = 2*R1_g - 2*R2_g - Mbig*(C0 - C1). Then conv3[g] = relu(map_g + BIG*s_g)
    #       -> but conv3 has a ReLU after it: conv3_out[g] = relu(conv2_out[g]) where
    #       conv2_out[g] = map_g + BIG*s_g. For active: map_g + BIG (positive if BIG large) ->
    #       relu passes. For inactive: map_g - BIG -> negative -> relu gives 0.
    #       Then out = sum_g [conv3_out[g] - BIG*gate_g] where gate_g channel must be
    #       available in conv3's input (conv2 channel) -> conv3_off[g] = BIG*gate_g.
    #       out = conv3_map[g] - conv3_off[g] -> for active: map+BIG - BIG = map; inactive: 0-0=0.
    #       But conv3_out[g] = relu(...) is a ReLU, so conv3_off[g] = relu(BIG*gate_g) = BIG*gate_g
    #       (nonnegative) fine.
    #       However the ReLU after conv3 clips conv3_off? conv3_off = BIG*gate_g >= 0 fine.
    #       And conv3_map[g] = map_g + BIG (active) >= 0 fine, 0 inactive.
    #       Then out: Wo[0, conv3_map[g]] = center-identity(+1); Wo[0, conv3_off[g]] = center-identity(-1).
    #       out = sum_g map_g (only active group contributes) = K3[g]*x.
    #       Also need ReLU safety in conv2: conv2 outputs must be positive for the copies
    #       (patch channels, gate channels) to be exact. Add large positive offsets and
    #       cancel them at the next layer.
    #       This is getting long but let's just implement it carefully with offsets.

    # ---- implement the final design ----
    # constants
    Mbig = 1.0e4      # ramp scale for t gating
    epsx = 1.0e-2     # x scale in gated patch channels
    BIG = 1.0e3       # offset scale for map activation
    OFF = 1.0e3       # positive offset to keep conv2 outputs positive

    # channel plan for conv1 (32 channels):
    #  0..8   : patch channels p=0..8  -> relu(epsx*x_patch_p + OFF1)  (always positive)
    #  9,10   : constant-1 pair: C0 = relu(Mbig*u + Mbig), C1 = relu(Mbig*u + Mbig - 1)
    #  11..   : per group g: R1_g = relu(Mbig*(u-tau_g) + Mbig), R2_g = relu(Mbig*(u-tau_g))
    #  total = 11 + 2G <= 32 -> G <= 10
    # conv2 (64 channels):
    #  0..8   : patch values (copied from conv1 patch channels, minus OFF1)
    #  9,10   : constant-1 (C0 - C1)
    #  11..11+2G-1 : gate_g = R1_g - R2_g   (0 or Mbig)
    #  next G channels: s_g = 2*gate_g - Mbig*const1  (Mbig if active, -Mbig if inactive)
    #  next G channels: map_g = K3[g] * patch (3x3 conv over the 9 patch channels)
    #  next G channels: off_g = BIG * gate_g  (for cancelling)
    #  total needed: 9 + 2 + 2G + G + G + G = 11 + 5G <= 64 -> G <= 10
    # conv3 (64 channels): copy channels 0..(needed) from conv2 (identity), plus
    #   conv3_map[g] = conv2_map[g] + BIG*s_g  (linear: kernel identity on map channel +
    #   BIG on s_g channel)
    #   conv3_off[g] = BIG * gate_g (copy of conv2 gate channel scaled)
    # out: Wo[0, conv3_map[g]] = +1 center ; Wo[0, conv3_off[g]] = -1 center
    #   plus cancel the OFF offsets introduced in conv2/conv3.
    # offsets: conv2 patch channels = conv1 patch - OFF1  -> conv3 copies them, and
    #   conv3_map[g] = K3[g]*(patch - OFF1) + ... -> the constant part K3[g]*OFF1*ones
    #   is a constant map; conv3_map[g] then contains a constant offset -> we must
    #   cancel it. Simplest: DON'T offset the patch channels; instead ensure positivity
    #   differently: patch channel = relu(epsx*x_patch_p + OFF1). Instead of subtracting
    #   OFF1 later, note that the map K3[g]*patch includes a constant component
    #   OFF1 * sum(K3[g]) which is spatially constant. We can cancel it by adding a
    #   compensating constant channel in conv3 (from the const-1 channel) and subtracting
    #   in out. Let's do it explicitly:
    #   map_g = K3[g] * (patch - OFF1*1)  -> we implement conv2_map[g] = K3[g]*patch
    #   and then subtract OFF1*(sum over p,q of K3[g,p,q]) * const1 in conv3_map[g].
    #   conv3_map[g] = conv2_map[g] + BIG*s_g - c_g * const1  where c_g = OFF1*sum(K3[g]).
    #   Then conv3_map[g] = K3[g]*x + BIG*s_g (active) -> good.
    #   Then out: out = sum_g [conv3_map[g] - conv3_off[g]] with conv3_off[g] = BIG*gate_g.
    #   For active group: (K3[g]*x + BIG) - BIG = K3[g]*x. For inactive: 0 - 0 = 0.
    #   But careful: conv3 applies ReLU to its output: conv3_map[g] for active group =
    #   K3[g]*x + BIG -> positive if BIG > |K3[g]*x| -> OK. For inactive: K3[g]*x - BIG ->
    #   negative -> ReLU -> 0. Good (that's the gating!).
    #   conv3_off[g] = BIG*gate_g >= 0 -> passes ReLU.
    #   Also conv3_map[g] must be >= 0 for the active group: need BIG > max|K3[g]*x|.
    #   Choose BIG = 1e3 (K3 values are small, x up to ~4, so |K3*x| < 1). Good.
    #   conv2 outputs must be positive for exact copies:
    #     - patch channels: conv2_patch[p] = conv1_patch[p] - OFF1 -> could be negative!
    #       Instead: keep conv2 patch channels = conv1 patch channels (positive, includes
    #       OFF1) and handle the constant in conv3 as described (c_g). Then conv2 patch
    #       channels are positive -> conv3 copies exact.
    #     - const1 channel: C0 - C1 = 1 > 0 fine.
    #     - gate channels: R1_g - R2_g = 0 or Mbig -> 0 is fine (ReLU passes 0).
    #     - s_g channels: 2*gate - Mbig*const1 = ±Mbig -> can be negative -> conv2 ReLU
    #       would clip! Problem.
    #       Fix: build s_g differently to be non-negative: we want s_g = +BIG_ACT for
    #       active and -BIG_ACT for inactive, but conv2's ReLU clips negatives.
    #       Alternative: use two channels: s_pos_g = BIG_ACT*gate_g (>=0) and use the
    #       "inactive" case via a different mechanism: conv3_map[g] = map_g + BIG*s_pos_g
    #       -> for inactive: map_g (small, could be negative -> ReLU clips to 0 partially!).
    #       Hmm, we want inactive -> exactly 0. If map_g is small and we subtract a large
    #       constant: conv3_map[g] = map_g + BIG*s_pos_g - BIG*const1 -> inactive: map_g -
    #       BIG < 0 -> ReLU -> 0. active: map_g + BIG - BIG = map_g -> but map_g may be
    #       negative -> ReLU clips! We need the final value to be exactly map_g (signed).
    #       So we need the offset trick at the out level: conv3_map[g] = map_g + BIG
    #       (active) and out subtracts BIG. But "BIG" must be added only when active.
    #       => conv3_map[g] = map_g + BIG*s_pos_g + BIG*const1*(1 - gate)??
    #       Let's define: conv3_map[g] = map_g + BIG*s_pos_g + BIG*(const1 - gate_g) where
    #       const1 = 1 channel, gate_g = 0/1 channel (Mbig scale -> use gate_g/Mbig).
    #       Let's use gate_g in units of 1 (i.e., conv2 gate channel = R1_g - R2_g scaled
    #       by 1/Mbig -> but R1-R2 = Mbig exactly, so divide by Mbig -> 1).
    #       Then: active (gate=1): conv3_map[g] = map_g + BIG*1 + BIG*(1-1) = map_g + BIG.
    #       inactive (gate=0): conv3_map[g] = map_g + 0 + BIG*1 = map_g + BIG -> also
    #       positive! Not zero. Hmm, that doesn't gate.
    #       We want: active -> map_g + BIG (positive), inactive -> negative (so ReLU -> 0).
    #       inactive: map_g + BIG*(gate-1) = map_g - BIG < 0 -> ReLU -> 0.
    #       active: map_g + BIG*(gate-1) = map_g -> may be negative -> clipped.
    #       So we need active: map_g + BIG, inactive: map_g - BIG. I.e. conv3_map[g] =
    #       map_g + BIG*(2*gate - 1). Requires the signed combination 2*gate - 1 which
    #       is negative for inactive -> clipped by conv2's ReLU.
    #       Workaround: put the "2*gate - 1" construction in conv3 instead of conv2!
    #       conv3 is linear over conv2's channels and has a bias. conv3_map[g] =
    #       sum_c W3[g,c]*conv2_c + b3[g]. We need conv2 to provide gate_g (0/1) and
    #       map_g and const1. Then conv3_map[g] = map_g + BIG*(2*gate_g - 1) can be
    #       computed in conv3 as: W3[g, map_ch] = identity, W3[g, gate_ch] = 2*BIG,
    #       W3[g, const1_ch] = -BIG, and bias b3[g] = 0. All conv2 channels are
    #       non-negative (map_g? map_g = K3[g]*patch can be negative! conv2's ReLU
    #       would clip it.)
    #       Argh: map_g must be non-negative in conv2 to survive conv2's ReLU.
    #       Solution: compute the map in conv3 (not conv2): conv3_map[g] = K3[g]*patch
    #       (from conv2's positive patch channels) + BIG*(2*gate_g - 1) — but the
    #       2*gate-1 term is signed, and conv3's output goes through ReLU, which is
    #       exactly what we want (clip negatives for inactive group). But the SIGNED
    #       term must be computed inside conv3 (linear) — yes! conv3's pre-activation
    #       is linear in conv2's channels: map_g (from patch channels) + 2*BIG*gate_g
    #       - BIG*const1. All fine inside conv3 since it's linear. The ReLU after
    #       conv3 then clips: active -> map_g + BIG (positive), inactive -> map_g - BIG
    #       (negative -> 0).
    #       Then out must subtract BIG for the active group: out = sum_g [conv3_map[g] -
    #       BIG*gate_g]. conv3_off[g] = BIG*gate_g (a conv3 channel: copy of gate with
    #       BIG scale; its pre-activation = BIG*gate_g >= 0 -> passes ReLU).
    #       out: Wo[0, conv3_map[g]] = +1 (center), Wo[0, conv3_off[g]] = -1 (center).
    #       Active: (map_g + BIG) - BIG = map_g. Inactive: 0 - 0 = 0.
    #       Also the constant OFF1 in the patch channels: conv3_map[g] includes
    #       K3[g]*(patch) where patch = epsx*x_patch + OFF1 -> constant part
    #       OFF1*sum(K3[g])*1 -> cancel with -c_g*const1 in conv3_map[g].
    #       Also the epsx scaling: patch channels hold epsx*x_patch + OFF1 -> we want
    #       map = K3[g]*x -> so conv3_map[g] = K3[g]*(patch - OFF1)/epsx -> scale by
    #       1/epsx. Fine (linear).
    #       Then out = sum over groups of K3[g]*x for the active group.
    #       Also we must handle the ReLU in conv2 for the patch channels: conv1 patch
    #       channels = relu(epsx*x_patch + OFF1) -> positive (OFF1 large) -> conv2 copies
    #       them (identity kernel) -> conv2 ReLU passes (positive). Good.
    #       conv1 constant channels: C0 = relu(Mbig*u + Mbig) (positive), C1 = relu(Mbig*u +
    #       Mbig - 1) (positive) -> conv2 const1 = C0 - C1 = 1 -> positive -> ok.
    #       conv1 ramp channels R1_g = relu(Mbig*(u-tau_g) + Mbig), R2_g = relu(Mbig*(u-tau_g))
    #       -> both >= 0 -> conv2 gate_g = (R1_g - R2_g)/Mbig = 1 or 0 -> >= 0 ok.
    #       Note: R1_g - R2_g = Mbig when u > tau_g, else 0. Check: u>tau: R1 = Mbig(u-tau)+Mbig,
    #       R2 = Mbig(u-tau) -> diff = Mbig. u<tau: R1 = relu(Mbig(u-tau)+Mbig): if u-tau > -1
    #       then R1 = Mbig(u-tau)+Mbig > 0, R2 = 0 -> diff = Mbig(u-tau)+Mbig ≠ Mbig. Hmm!
    #       So the difference isn't binary unless Mbig(u-tau)+Mbig <= 0, i.e., u <= tau - 1.
    #       Since u ∈ [0,1] and tau ∈ (0,1), u - tau > -1 always -> R1 > 0 always. Bad.
    #       Fix: use R1_g = relu(Mbig*(u-tau_g) + Mbig) with Mbig replaced by a much
    #       larger scale for the offset: define R1_g = relu(Mbig*(u - tau_g) + BIGOFF) with
    #       BIGOFF = Mbig*1.5? Then diff = Mbig(u-tau)+BIGOFF for u>tau... still not binary.
    #       We want a step function. Use: R1_g = relu(Mbig*(u - tau_g) + Mbig) and
    #       R2_g = relu(Mbig*(u - tau_g) + Mbig) - ... no.
    #       Correct construction of a step from two ramps:
    #         step(u>tau) = relu(Mbig*(u-tau) + Mbig) - relu(Mbig*(u-tau) + Mbig - Mbig)
    #       = relu(a + Mbig) - relu(a) where a = Mbig(u-tau).
    #       For a > 0: relu(a+Mbig) - relu(a) = (a+Mbig) - a = Mbig. ✓
    #       For a < 0 but a + Mbig > 0 (i.e. -Mbig < a < 0): (a+Mbig) - 0 = a+Mbig ≠ Mbig. ✗
    #       So the step is exact only if a < -Mbig for inactive, i.e. Mbig(u-tau) < -Mbig
    #       -> u < tau - 1. Impossible for u∈[0,1], tau∈(0,1).
    #       => Use a much larger ramp scale Mbig and a smaller offset: step via
    #          R1 = relu(Mbig*(u-tau) + Mbig*delta), R2 = relu(Mbig*(u-tau) + Mbig*delta - Mbig)
    #          Hmm same issue.
    #       The fundamental issue: relu of a linear function is a ramp (0 then linear),
    #       and the difference of two ramps with different offsets is a clipped ramp,
    #       which equals the constant only when the ramp has saturated past the offset.
    #       To get a step, use: step = relu(a + c) - relu(a + c - K) where K is large:
    #         for a > -c: relu(a+c) = a+c; relu(a+c-K) = 0 if a+c < K -> diff = a+c
    #            (not constant). ✗
    #       Use: step = relu(a + c) - relu(a) with c = 0 gives a (ramp). ✗
    #       A step function of a scalar cannot be built from 2 ReLUs exactly? Actually
    #       it can: relu(a+c) - relu(a+c-K) - ... hmm. Consider relu(a+c) - relu(a+c) = 0.
    #       Consider: relu(a+c) - relu(a+c-K) - (c - (c-K)) = clipped ramp - K... Let's
    #       think: we want f(a) = 1 for a>0, 0 for a<0. ReLU nets with 1 hidden layer
    #       represent piecewise-linear functions with breakpoints at the kink points.
    #       relu(a) is 0 for a<0, a for a>0. relu(a) - relu(a-1) = 0 (a<0), a (0<a<1),
    #       1 (a>1). So a clipped ramp. To get a step at 0: relu(a) - relu(a) ... no.
    #       Use: relu(a) - relu(a) + ... Actually: relu(a) - relu(a - 0) = 0. Hmm.
    #       A step at 0 = relu(a) - relu(a)*slope... The step function is NOT piecewise
    #       linear with a single breakpoint? It IS piecewise linear: 0 for a<0, 1 for a>0,
    #       with a jump at 0 (discontinuous). ReLU networks produce continuous functions!
    #       So an exact step is impossible. But a CLIPPED RAMP is fine: we can use a
    #       clipped ramp as the "gate" — but then the gate value varies with t within
    #       the group, which would scale the map incorrectly (we want a constant per group).
    #       Alternative: make the ramp saturate over the whole group's t-range:
    #       gate_g(a) = relu(a + c) - relu(a + c - K) = min(max(a+c,0), K). If we choose
    #       c and K so that within the group's u-range the value is constant K, and
    #       outside it's 0, we get a near-step (with a ramp transition region of width K/Mbig).
    #       Choose: for group g with u-range [lo_g, hi_g], we want the gate = K for u in
    #       [lo_g, hi_g] and 0 for u < lo_g (and for u > hi_g we want 0 too, but the ramp
    #       only increases...). Hmm, we need a band-pass, not a step.
    #       Band-pass = step(u>lo) - step(u>hi): each step is a clipped ramp. With
    #       clipped ramps: S_lo = relu(Mbig(u-lo)+K) - relu(Mbig(u-lo)) hmm.
    #       Let's define clipped ramp: CR(a; K) = relu(a + K) - relu(a) = min(max(a,0),K).
    #       Then step-like(u > lo) ≈ CR(Mbig*(u-lo) + K/2 ... ). Let's use:
    #       S(u; lo) = CR(Mbig*(u-lo) + K, K)?? Let's compute CR(a)=min(max(a,0),K).
    #       We want S(u;lo) = 1 for u>lo, 0 for u<lo - eps. Take a = Mbig*(u-lo).
    #       CR(a + K) = min(max(a+K,0),K) = K for a>=0, = a+K for -K<a<0, = 0 for a<-K.
    #       So CR(a+K) = K for u>lo, and ramps down to 0 for u < lo - K/Mbig.
    #       With Mbig = 1e4 and K = 1e4 (i.e., K/Mbig = 1 in u units) the ramp region
    #       spans u ∈ [lo-1, lo] -> too wide. Instead choose K small relative to Mbig:
    #       we need K/Mbig << spacing between groups, but also K must be >> the
    #       variation of Mbig*(u-lo) within a group... contradictory: within a group
    #       u varies by ~1/G, so a varies by Mbig/G. For the clipped ramp to be
    #       constant (=K) over the group we need a >= 0 for all u in group -> lo <= min u
    #       in group, and a + K >= K -> fine, and for u below lo we need a + K <= 0 ->
    #       u <= lo - K/Mbig. So the transition region is [lo - K/Mbig, lo]. We need
    #       this to be outside all group ranges -> K/Mbig < spacing/2. But within the
    #       group, a = Mbig*(u-lo) ranges over [0, Mbig/G] which must be <= K for
    #       saturation?? No: CR(a+K) = K requires a+K >= K i.e. a >= 0. ✓ (a>=0 in group).
    #       So within the group, CR(a+K) = K exactly (constant!) as long as a >= 0.
    #       And for u < lo: CR = min(max(a+K,0),K) = a+K if a > -K, which is nonzero
    #       (ramp) for u in (lo - K/Mbig, lo). With K/Mbig = 1e-3 (i.e., K=10, Mbig=1e4)
    #       the ramp region is tiny (u width 1e-3 = 0.03 timesteps). But we also need
    #       a >= 0 for all u in the group: lo = min u in group. Then for u slightly below
    #       lo (previous group's timesteps) the gate is in the ramp region → wrong.
    #       Since groups are contiguous integer timesteps, we can set lo_g = (first t of
    #       group - 0.5)/31 and hi... but the ramp region [lo - K/Mbig, lo] would then
    #       cover u values between the previous timestep and lo. With K/Mbig = 1e-3 and
    #       timestep spacing 1/31 = 0.032, the ramp region (width 0.001) is much smaller
    #       than the gap to the previous timestep (0.032) -> so at the previous
    #       timestep's u, a = Mbig*(u_prev - lo) = -Mbig*0.032 = -320, and a + K = -310 < 0
    #       -> CR = 0. ✓
    #       So: gate_g = CR(Mbig*(u - lo_g) + K) with K = 10, Mbig = 1e4, lo_g =
    #       (first_t_g - 0.5)/31. Then gate_g = K for u >= lo_g (all timesteps in group g
    #       and later ones) and 0 for u <= lo_g - K/Mbig.
    #       But for LATER groups, gate_g is also K (since u > lo_g) -> not exclusive!
    #       We need exclusivity: group g's gate should be 1 only for u in group's range.
    #       Use band-pass: gate_g = CR(Mbig*(u-lo_g)+K) - CR(Mbig*(u-hi_g)+K) where
    #       hi_g = (last_t_g + 0.5)/31. Then for u in [lo_g, hi_g]: first = K, second = 0
    #       (since u < hi_g - K/Mbig) -> gate = K. For u > hi_g: first = K, second = K ->
    #       gate = 0. ✓ For u < lo_g: both 0 -> 0 ✓.
    #       So each gate needs 4 ramp channels (2 per CR). With G groups: 4G channels
    #       in conv1 for gates + 9 patch + 2 const = 11 + 4G <= 32 -> G <= 5.
    #       conv2 channels: 9 patch + 2 const + G gates + G maps + G offs = 11 + 3G <= 64 ✓.
    #       conv3: G maps + G offs = 2G <= 64 ✓.
    #       G=5 groups of ~6-7 timesteps. Hmm, or G=4 (8 timesteps each).
    #       Alternatively reduce patch channels: the map only needs the 9 patch positions
    #       (9 channels). Could we use fewer, e.g., a 3x3 kernel needs 9. Keep 9.
    #       Alternatively, note CR(a+K) - CR(a) can be computed with 2 channels; and
    #       CR(Mbig*(u-lo)+K) - CR(Mbig*(u-hi)+K) needs 4 channels per group. Yes.
    #       Hmm, alternatively use a single CR per group with the "monotone ramp" trick:
    #       since groups are ordered, we can use cumulative gates: gate_g = S_g - S_{g+1}
    #       where S_g = CR(Mbig*(u-lo_g)+K) (monotone step-like). Then S_g needs 2 channels
    #       each, but consecutive S's share... each S_g = CR(Mbig(u-lo_g)+K) - CR(Mbig(u-lo_g))
    #       = relu(Mbig(u-lo_g)+K) - relu(Mbig(u-lo_g)). The second term relu(Mbig(u-lo_g))
    #       is a different channel per g. So still 2 per group. Total 2G for S's, then
    #       gate_g = S_g - S_{g+1} computed in conv2 (linear) -> conv2 needs S_g channels.
    #       conv1 channels: 9 + 2 + 2G <= 32 -> G <= 10 ✓.
    #       conv2 channels: 9 patch + 2 const + G (S_g) + G (map) + G (off) = 11+3G <= 64 ✓
    #       conv3: 2G ✓.
    #       With G=8 groups of 4 timesteps, or G=10 (3-4 timesteps).
    #       Note: S_g for the last group must be defined with lo beyond 1 -> S_last = K
    #       for all u (constant) — handle: define S_{G} = constant K (use the const channel
    #       scaled). Then gate_{G-1} = S_{G-1} - K.
    #       Actually simpler: define S_g for g=0..G-1 with lo_g, and S_G = const K.
    #       gate_g = S_g - S_{g+1} (for g=0..G-1).
    #       conv2 computes gate_g = S_g - S_{g+1} (linear) -> non-negative ✓ (since S is
    #       non-increasing? S_g is non-decreasing in g... wait S_g = CR(Mbig(u-lo_g)+K):
    #       for a given u, as g increases lo_g increases, so Mbig(u-lo_g) decreases, so
    #       S_g decreases with g. So gate_g = S_g - S_{g+1} >= 0 ✓.
    #       And gate_g = K exactly for u in group g's range ✓ (S_g = K, S_{g+1} = 0).
    #       Wait, need S_{g+1} = 0 for u in group g's range: S_{g+1} = CR(Mbig(u-lo_{g+1})+K)
    #       with lo_{g+1} > u (since group g+1 starts later) -> Mbig(u-lo_{g+1}) < 0 and
    #       we need it <= -K -> u <= lo_{g+1} - K/Mbig. With lo_{g+1} = (first_t_{g+1}-0.5)/31
    #       and u <= (last_t_g)/31, the gap is (first_t_{g+1} - 0.5 - last_t_g)/31 =
    #       (0.5)/31 = 0.016 in u units -> Mbig*0.016 = 160 > K=10 ✓.
    #       And S_g = K for u in group g: need Mbig(u - lo_g) >= 0 -> u >= lo_g ✓ (u >= (first_t_g)/31 > lo_g) and Mbig(u-lo_g)+K >= K → fine, CR = min(max(a+K,0),K) = K requires a+K >= K → a >= 0 ✓.
    #       Also for u > lo_g + K/Mbig, CR(a+K) = K ✓ (since a+K >= K).
    #       Great: S_g = K for u >= lo_g, and ramps to 0 over [lo_g - K/Mbig, lo_g].
    #       So S_g(u) = K for u >= lo_g, 0 for u <= lo_g - K/Mbig.
    #       gate_g = S_g - S_{g+1}: for u in group g: K - 0 = K ✓. For u > group g's
    #       range but before group g+1: K - 0 = K → WRONG (gate_g should be 0 there).
    #       Hmm: for u between last_t_g and first_t_{g+1} (i.e., in the gap), S_g = K and
    #       S_{g+1} = 0 → gate_g = K. But u only takes integer-timestep values, and the
    #       gap contains no integer t (groups are contiguous covering all t). So for
    #       integer t, u is always within some group's range → gate_g = K only for t in
    #       group g ✓ (since S_{g+1} = 0 for u <= lo_{g+1} - K/Mbig and u in group g
    #       satisfies that).
    #       Wait, but for t in group g+1: S_g = K (since u > lo_g), S_{g+1} = K → gate_g = 0 ✓.
    #       For t in group g: S_g = K, S_{g+1} = 0 → gate_g = K ✓.
    #       For t in group g-1: S_g = 0 → gate_g = 0 ✓.
    #       PERFECT. So gate_g = K for t in group g, 0 otherwise.
    #       Now conv2 gate channel = (S_g - S_{g+1})/K → 1 or 0 ✓ (non-negative).
    #       conv3_map[g] = map_g + BIG*(2*gate_g - 1) computed linearly in conv3 ✓.
    #       conv3_off[g] = BIG*gate_g ✓ (non-negative).
    #       out: sum_g [conv3_map[g] - conv3_off[g]] = map_g for the active group ✓.
    #       Also need to cancel the OFF1 constant in the patch channels and the epsx scale.
    #       Let's define conv1 patch channel p: pre-act = epsx * x_patch_p + OFF1,
    #       post-ReLU = epsx*x_patch_p + OFF1 (positive). conv2 copies it. conv3_map[g]
    #       = (1/epsx) * K3[g] applied to (patch - OFF1) + BIG*(2*gate_g - 1)
    #       = K3[g]*x + BIG*(2*gate_g - 1) ✓ (constant part cancelled via const channel).
    #       The constant cancellation: (1/epsx)*K3[g]*(patch) includes (1/epsx)*OFF1*sum(K3)
    #       constant -> subtract via const1 channel: conv3_map[g] gets
    #       W3[g, const1_ch] = -(OFF1/epsx)*sum(K3[g]) and W3[g, map_ch_p] = K3[g,p,q]/epsx
    #       Wait, the map is computed in conv3 from the patch channels: conv3_map[g] =
    #       sum_p W3[g, patch_ch_p] * patch_p (3x3 conv) + ... So W3[g, patch_ch_p, :, :] =
    #       K3[g,p,:,:]/epsx. And the constant: sum_p (K3[g,p]/epsx)*OFF1*1 = (OFF1/epsx)*sum_p
    #       K3[g,p] -> subtract with W3[g, const1_ch] = -(OFF1/epsx)*sum(K3[g]) (center kernel).
    #       Then conv3_map[g] = K3[g]*x + BIG*(2*gate_g - 1) ✓.
    #       ReLU after conv3: active -> K3[g]*x + BIG > 0 (need BIG > |K3[g]*x|);
    #       inactive -> K3[g]*x - BIG < 0 -> 0 ✓.
    #       conv3_off[g] = BIG*gate_g: pre-act = BIG*gate_g (>=0) -> passes ✓.
    #       out: Wo[0, conv3_map[g]] = identity center +1; Wo[0, conv3_off[g]] = -1 center.
    #       out = sum_g [conv3_map[g] - conv3_off[g]] = K3[g]*x for the active group ✓.
    #       Also: out.bias = 0.
    #       Also conv2's ReLU: conv2 outputs must be >= 0: patch channels (epsx*x+OFF1 >
    #       0 ✓), const1 = 1 ✓, S_g channels (>=0 ✓), map channels — wait, do we need
    #       map channels in conv2? No: the map is computed in conv3 from the patch
    #       channels. So conv2 channels: 9 patch + 2 const + (G+1) S channels + G gate
    #       channels = 12 + 2G <= 64 ✓ (G<=26).
    #       conv3 channels: G maps + G offs = 2G <= 64 ✓.
    #       conv1 channels: 9 patch + 2 const + 2(G+1) ramp channels = 13 + 2G <= 32
    #       -> G <= 9. Use G=8 (groups of 4) or G=6.
    #       Also: conv2 must copy the patch channels exactly: conv2 patch channel p has
    #       kernel = identity at conv1's patch channel p (center 1) ✓ and bias 0.
    #       conv2 S channels: copy from conv1's ramp channels (center 1) ✓.
    #       conv2 gate channel g: = (S_g - S_{g+1})/K: kernel on conv1's ramp channels:
    #       S_g = relu(Mbig(u-lo_g)+K) - relu(Mbig(u-lo_g)) → conv1 ramp channels:
    #         rampA_g = relu(Mbig*(u - lo_g) + K)   (pre-act: Mbig*u - Mbig*lo_g + K)
    #         rampB_g = relu(Mbig*(u - lo_g))       (pre-act: Mbig*u - Mbig*lo_g)
    #       conv2 gate_g = (rampA_g - rampB_g)/K ✓ (linear in conv1 channels).
    #       Note: conv1's ramp pre-activations use the t channel (index 1) with kernel
    #       position (0,0) = Mbig and bias = -Mbig*lo_g (+K for rampA). Since the t
    #       channel is broadcast spatially, the conv output is constant spatially ✓.
    #       IMPORTANT: conv1's t-channel weight at kernel position (0,0) means the conv
    #       output at pixel (i,j) = sum over kernel positions of W[.,1,ky,kx]*u(i+ky-1,j+kx-1)
    #       = Mbig*u (since u is constant) ✓.
    #       But careful: the x-channel kernel for the patch channels: patch position p
    #       (ky,kx) → conv1 channel p has W[.,0,ky,kx] = epsx*delta. Then conv1 output
    #       at pixel (i,j) = epsx*x(i+ky-1, j+kx-1) + OFF1 ✓ (cross-correlation with
    #       zero padding at borders ✓).
    #       Then conv3_map[g] = sum_p sum_{q} W3[g,patch_ch_p,q] * patch_p(shift q)
    #       = (1/epsx) * sum_p sum_q K3[g,p,q] * x_patch_p(shift q) = K3[g]*x ✓
    #       (with the same cross-correlation convention, and zero padding at borders ✓).
    #       One subtlety: the patch channels at the border have zero padding, so the
    #       effective receptive field is zero-padded ✓ consistent with the fit (which used
    #       zero-padded patches ✓).
    #       Also the constant OFF1 in patch channels: at borders, the patch value is
    #       still epsx*x + OFF1 (OFF1 is added everywhere) ✓ so the constant cancellation
    #       works uniformly ✓.

    # ---- choose G and fit per-group 3x3 kernels ----
    def eval_group_kernels(G, lam_rel):
        per = int(np.ceil(32.0 / G))
        grps = [list(range(i, min(i + per, 32))) for i in range(0, 32, per)]
        kers = {}
        for gi, gr in enumerate(grps):
            idx = np.isin(t, gr)
            A, b = build_group_design(idx)
            lam = lam_rel * len(b)
            kers[gi] = np.linalg.solve(A.T @ A + lam * np.eye(9), A.T @ b).reshape(3, 3)
        # evaluate
        tot = 0.0
        for gi, gr in enumerate(grps):
            idx = np.isin(t, gr)
            A, b = build_group_design(idx)
            r = b - A @ kers[gi].reshape(9)
            tot += float(np.sum(r * r))
        return tot / (N * 256), grps, kers

    bestG = None
    for G in [2, 3, 4, 5, 6, 8]:
        for lam_rel in [1e-6, 1e-5, 1e-4, 1e-3]:
            m, grps, kers = eval_group_kernels(G, lam_rel)
            if bestG is None or m < bestG[0]:
                bestG = (m, G, grps, kers)
    m_best, G, groups, K3 = bestG

    # ---- build the network ----
    Mbig = 1.0e4
    K = 10.0            # clip width for step-like ramps
    epsx = 1.0e-2
    OFF1 = 1.0e2
    BIG = 1.0e3

    nch1 = 13 + 2 * (G + 1)
    if nch1 > 32:
        G = min(G, 9)
        per = int(np.ceil(32.0 / G))
        groups = [list(range(i, min(i + per, 32))) for i in range(0, 32, per)]
        G = len(groups)
        m_best, G, groups, K3 = eval_group_kernels(G, 1e-5)
        nch1 = 13 + 2 * (G + 1)

    W1 = np.zeros((32, 2, 3, 3))
    b1 = np.zeros(32)
    W2 = np.zeros((64, 32, 3, 3))
    b2 = np.zeros(64)
    W3 = np.zeros((64, 64, 3, 3))
    b3 = np.zeros(64)
    Wo = np.zeros((1, 64, 3, 3))
    bo = np.zeros(1)

    # conv1 layout
    # 0..8   patch channels
    # 9,10   const channels (C0, C1)
    # 11+2g, 12+2g : rampA_g, rampB_g  for g=0..G  (G+1 pairs)
    idx_patch = list(range(9))
    idx_C0, idx_C1 = 9, 10
    idx_rampA = [11 + 2 * g for g in range(G + 1)]
    idx_rampB = [12 + 2 * g for g in range(G + 1)]

    for p in range(9):
        ky, kx = divmod(p, 3)
        W1[p, 0, ky, kx] = epsx
        b1[p] = OFF1
    # const channels: C0 = relu(Mbig*u + Mbig) ; C1 = relu(Mbig*u + Mbig - 1)
    W1[idx_C0, 1, 0, 0] = Mbig
    b1[idx_C0] = Mbig
    W1[idx_C1, 1, 0, 0] = Mbig
    b1[idx_C1] = Mbig - 1.0
    # ramp channels
    for g in range(G + 1):
        if g < G:
            lo = (groups[g][0] - 0.5) / 31.0
        else:
            lo = 1.5   # beyond u=1 -> S_G = 0? we want S_G = 0 for all u in [0,1]
        # S_G should be 0 always (so gate_{G-1} = S_{G-1} - 0 = K for t in last group)
        # set lo_G = 1.5 -> Mbig*(u-1.5) + K < 0 for u<=1 ✓ (Mbig*(-0.5) = -5000 < -K)
        W1[idx_rampA[g], 1, 0, 0] = Mbig
        b1[idx_rampA[g]] = -Mbig * lo + K
        W1[idx_rampB[g], 1, 0, 0] = Mbig
        b1[idx_rampB[g]] = -Mbig * lo

    # conv2 layout
    # 0..8   patch copies
    # 9      const1 = C0 - C1
    # 10..10+G  S_g copies (rampA - rampB)/K  -> S_g/K in {0,1}
    # 11+G..10+2G gate_g = (S_g - S_{g+1})/K
    idx2_patch = list(range(9))
    idx2_const = 9
    idx2_S = [10 + g for g in range(G + 1)]
    idx2_gate = [11 + G + g for g in range(G)]

    for p in range(9):
        W2[idx2_patch[p], p, 1, 1] = 1.0
    W2[idx2_const, idx_C0, 1, 1] = 1.0
    W2[idx2_const, idx_C1, 1, 1] = -1.0
    for g in range(G + 1):
        W2[idx2_S[g], idx_rampA[g], 1, 1] = 1.0 / K
        W2[idx2_S[g], idx_rampB[g], 1, 1] = -1.0 / K
    for g in range(G):
        W2[idx2_gate[g], idx_rampA[g], 1, 1] = 1.0 / K
        W2[idx2_gate[g], idx_rampB[g], 1, 1] = -1.0 / K
        W2[idx2_gate[g], idx_rampA[g + 1], 1, 1] = -1.0 / K
        W2[idx2_gate[g], idx_rampB[g + 1], 1, 1] = 1.0 / K

    # conv3 layout
    # 0..G-1   map channels
    # G..2G-1  off channels
    idx3_map = list(range(G))
    idx3_off = [G + g for g in range(G)]
    for g in range(G):
        kern = K3[g] / epsx
        for p in range(9):
            ky, kx = divmod(p, 3)
            W3[idx3_map[g], idx2_patch[p], ky, kx] = kern[p]
        # constant cancellation
        W3[idx3_map[g], idx2_const, 1, 1] = -(OFF1 / epsx) * float(np.sum(K3[g]))
        # gate term: BIG*(2*gate - 1)
        W3[idx3_map[g], idx2_gate[g], 1, 1] = 2.0 * BIG
        W3[idx3_map[g], idx2_const, 1, 1] += -BIG
        # off channel: BIG * gate
        W3[idx3_off[g], idx2_gate[g], 1, 1] = BIG
    # out
    for g in range(G):
        Wo[0, idx3_map[g], 1, 1] = 1.0
        Wo[0, idx3_off[g], 1, 1] = -1.0

    # ---- verify numerically on training data ----
    W1f = convT(W1); W2f = convT(W2); W3f = convT(W3); Wof = convT(Wo)
    pred = forward(x.astype(np.float32), t, W1f, b1, W2f, b2, W3f, b3, Wof, bo)
    mse_new = float(np.mean((pred.reshape(N, -1) - Yf) ** 2))
    mse_old = 1.0022

    # ---- assemble output ----
    out = {}
    for k in w:
        out[k] = np.asarray(w[k], dtype=np.float32).copy()
    out["conv1.weight"] = W1.astype(np.float32)
    out["conv1.bias"] = b1.astype(np.float32)
    out["conv2.weight"] = W2.astype(np.float32)
    out["conv2.bias"] = b2.astype(np.float32)
    out["conv3.weight"] = W3.astype(np.float32)
    out["conv3.bias"] = b3.astype(np.float32)
    out["out.weight"] = Wo.astype(np.float32)
    out["out.bias"] = bo.astype(np.float32)
    for k in out:
        out[k] = np.clip(np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0), -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        "mse_new_train": round(mse_new, 5),
        "mse_old_train": mse_old,
        "mse_group3x3_fit": round(m_best, 5),
        "mse_wiener_full": round(mse_wiener, 5),
        "num_groups": int(G),
        "group_sizes": [len(gr) for gr in groups],
        "max_abs_weight": float(max(np.abs(out[k]).max() for k in out)),
    }
    return {"weights": out, "diagnostics": diagnostics}
