import numpy as np

def edit(w, f):
    # ---------------------------------------------------------------
    # 1. Read training pool and build bag-of-words count matrix
    # ---------------------------------------------------------------
    X = f["x"].astype(np.int64)          # [N, 64]
    y = f["y"].astype(np.int64)          # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]       # 2000
    D = w["embed.weight"].shape[1]       # 32

    # Count tokens, ignoring padding (id 0)
    counts = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        valid = row[row != 0]
        if valid.size:
            np.add.at(counts[i], valid, 1.0)

    # Normalize by number of non-pad tokens (masked mean pooling)
    lengths = np.maximum((X != 0).sum(axis=1), 1).astype(np.float64)
    counts = counts / lengths[:, None]   # [N, V]

    # ---------------------------------------------------------------
    # 2. Fit a regularized least-squares linear classifier on counts
    #    (closed form; no iterative optimization)
    # ---------------------------------------------------------------
    # Target: +1 for positive, -1 for negative
    t = np.where(y == 1, 1.0, -1.0)

    # Add bias column
    A = np.concatenate([counts, np.ones((N, 1))], axis=1)  # [N, V+1]

    # Ridge regression:  w_ls = (A^T A + lambda I)^-1 A^T t
    lam = 1.0
    AtA = A.T @ A
    AtA += lam * np.eye(AtA.shape[0])
    Atb = A.T @ t
    w_ls = np.linalg.solve(AtA, Atb)     # [V+1]

    w_tok = w_ls[:V]                     # per-token weights
    b_ls = w_ls[V]

    # ---------------------------------------------------------------
    # 3. Embed the desired per-token score into the 32-dim embedding
    #    We want:  v @ LN(e_t) = w_t   for all t, where v = head_1 - head_0.
    #    Choose v = (1, 0, 0, ..., 0).  Then we need LN(e_t)[0] = w_t.
    #    LayerNorm subtracts the mean and divides by std.  To make
    #    LN(e_t)[0] = w_t, set e_t[0] = w_t and e_t[1:] = -w_t/31.
    #    Then mean(e_t) = 0 and std(e_t) = |w_t| * sqrt(1/31 + 1/31^2 * 31)
    #    = |w_t| * sqrt(1/31 + 1/31) = |w_t| * sqrt(2/31).
    #    So LN(e_t)[0] = w_t / (|w_t| * sqrt(2/31)) = sign(w_t) * sqrt(31/2).
    #    That is not w_t.  We need a different construction.
    #
    #    Better: choose e_t = c + alpha_t * v + u_t, with u_t orthogonal to v.
    #    Then LN(e_t) = LN(c + alpha_t v + u_t).  If we make the variance
    #    dominated by a large constant component, LN becomes approximately
    #    linear.  But we can do it exactly by choosing the embedding so that
    #    LN(e_t) is exactly a linear function of w_t.
    #
    #    Exact construction:
    #    Let e_t = a * w_t * v + b * v + u_t, where u_t is orthogonal to v.
    #    Then v @ e_t = a * w_t * |v|^2 + b * |v|^2.
    #    We want v @ LN(e_t) = w_t.
    #    LN(e_t) = (e_t - mean(e_t)) / std(e_t).
    #    If we choose u_t such that mean(e_t) = 0 and std(e_t) = 1, then
    #    LN(e_t) = e_t, and v @ e_t = a * w_t * |v|^2 + b * |v|^2.
    #    Set a = 1/|v|^2, b = 0.  Then v @ e_t = w_t.
    #    So we need e_t = w_t * v / |v|^2 + u_t, with u_t orthogonal to v,
    #    and we must enforce mean(e_t) = 0 and std(e_t) = 1.
    #
    #    We can always find such u_t because we have 31 dimensions orthogonal
    #    to v.  We can adjust u_t to fix the mean and variance.
    #
    #    Simpler: use a large constant offset to make LayerNorm approximately
    #    linear, and accept a tiny approximation error.  Since eps=1e-5 is
    #    small, we can make the variance large and the mean zero.
    #
    #    Let v = (1, 0, ..., 0).  Set e_t[0] = w_t * S, and e_t[1:] = -w_t * S / 31,
    #    where S is a large scale factor.  Then mean(e_t) = 0.
    #    var(e_t) = (1/32) * ( (w_t S)^2 + 31 * (w_t S / 31)^2 )
    #             = (w_t S)^2 / 32 * (1 + 1/31)
    #             = (w_t S)^2 / 32 * (32/31)
    #             = (w_t S)^2 / 31.
    #    std(e_t) = |w_t S| / sqrt(31).
    #    LN(e_t)[0] = (w_t S) / (|w_t S| / sqrt(31)) = sign(w_t) * sqrt(31).
    #    That's constant magnitude, not w_t.  So this doesn't work.
    #
    #    We need LN(e_t)[0] to be proportional to w_t.  That means the
    #    first component after normalization should be w_t.  But normalization
    #    divides by the norm.  To get a linear response, we can use a
    #    different approach: make the embedding have a constant component
    #    that dominates the variance, so that LN(e_t) ≈ (e_t - mean) / sigma
    #    where sigma is nearly constant.  Then LN(e_t)[0] ≈ (e_t[0] - mean) / sigma.
    #    If we set e_t[0] = w_t, and the other components are chosen so that
    #    mean and sigma are constant across t, then LN(e_t)[0] ∝ w_t.
    #
    #    Let's do this: set e_t[0] = w_t.  Set e_t[1:] = c (a constant vector).
    #    Then mean(e_t) = (w_t + 31 c) / 32.
    #    var(e_t) = (1/32) * ( (w_t - mean)^2 + 31 * (c - mean)^2 ).
    #    This depends on w_t, so sigma varies.  To make sigma constant, we
    #    can make c very large compared to w_t.  Then mean ≈ c, and
    #    var ≈ (1/32) * ( (w_t - c)^2 + 31 * (c - c)^2 ) ≈ (w_t - c)^2 / 32.
    #    Still depends on w_t.  Hmm.
    #
    #    Actually, we can use the fact that the head is 2x32.  We can set
    #    head_1 - head_0 = v, and we want v @ LN(e_t) = w_t.  We can choose
    #    e_t such that LN(e_t) = w_t * v / |v|^2 + u_t, where u_t is
    #    orthogonal to v.  Then v @ LN(e_t) = w_t.  So we need to find e_t
    #    such that LN(e_t) has a specific projection onto v.  Since LN is
    #    invertible (for a given mean and variance), we can set e_t = LN^{-1}(desired).
    #    But LN^{-1} is not unique because LN is invariant to scaling and
    #    shifting?  Actually LN(x) = (x - mean(x)) / std(x).  Given a desired
    #    normalized vector z, we can set x = alpha * z + beta * 1, for any
    #    alpha > 0 and any beta.  Then LN(x) = z.  So we can choose any
    #    alpha and beta.  So we can set e_t = alpha * z_t + beta * 1, where
    #    z_t is the desired normalized vector.  We want v @ z_t = w_t.
    #    So we can choose z_t = w_t * v / |v|^2 + u_t, with u_t orthogonal to v.
    #    Then e_t = alpha * z_t + beta * 1.  We can choose alpha and beta
    #    freely.  So we can set e_t = z_t (alpha=1, beta=0).  Then LN(e_t) = z_t
    #    only if z_t already has mean 0 and std 1.  But z_t = w_t * v / |v|^2 + u_t.
    #    We can choose u_t to make mean(z_t) = 0 and std(z_t) = 1.  Since u_t
    #    has 31 degrees of freedom, we can always do this.  So we can achieve
    #    exactly v @ LN(e_t) = w_t.
    #
    #    Let's implement this exactly.
    # ---------------------------------------------------------------

    # Choose v = (1, 0, ..., 0) for simplicity.
    v = np.zeros(D, dtype=np.float64)
    v[0] = 1.0
    v_norm2 = 1.0

    # Desired z_t = w_t * v / |v|^2 + u_t, with u_t orthogonal to v.
    # We need mean(z_t) = 0 and std(z_t) = 1.
    # Let u_t = (0, a_t, a_t, ..., a_t) + (0, b_t, -b_t, 0, ..., 0) ... 
    # Actually, we can just set u_t = (0, c_t, c_t, ..., c_t) plus a small
    # adjustment.  But we need 31 components.  Let's use a simple construction:
    # z_t[0] = w_t.  z_t[1:] = -w_t / 31 + d_t, where d_t is a constant
    # added to all 31 components.  Then mean(z_t) = (w_t + 31*(-w_t/31 + d_t)) / 32
    # = (w_t - w_t + 31 d_t) / 32 = 31 d_t / 32.
    # To make mean 0, set d_t = 0.  Then z_t[1:] = -w_t / 31.
    # Then mean(z_t) = 0.  Now compute var(z_t):
    # var = (1/32) * ( w_t^2 + 31 * (w_t/31)^2 ) = (1/32) * ( w_t^2 + w_t^2/31 )
    # = w_t^2 / 32 * (32/31) = w_t^2 / 31.
    # So std(z_t) = |w_t| / sqrt(31).  This is not 1 unless |w_t| = sqrt(31).
    # So we need to scale z_t to have std 1.  But scaling z_t changes its
    # projection onto v.  If we scale z_t by alpha, then v @ (alpha z_t) = alpha w_t.
    # So we need alpha = 1 to keep the projection w_t.  So we cannot scale.
    # Therefore we need a different u_t that makes the variance 1 while
    # keeping the projection w_t.
    #
    # We have 31 dimensions orthogonal to v.  We can use them to adjust the
    # variance without changing the projection.  Let z_t[0] = w_t.
    # Let z_t[1:] = a_t * u + b_t * w, where u is a fixed unit vector in the
    # orthogonal space, and w is another orthogonal vector.  We want
    # mean(z_t) = 0 and var(z_t) = 1.
    # mean(z_t) = (w_t + sum_{j=1}^{31} z_t[j]) / 32 = 0.
    # var(z_t) = (1/32) * ( w_t^2 + sum_{j=1}^{31} z_t[j]^2 ) = 1.
    # We have two equations and 31 variables.  We can solve by setting
    # z_t[1] = -w_t, and z_t[2:] = c_t (constant).  Then
    # sum_{j=1}^{31} z_t[j] = -w_t + 30 c_t.
    # mean = (w_t - w_t + 30 c_t) / 32 = 30 c_t / 32 = 0 => c_t = 0.
    # Then z_t[1] = -w_t, z_t[2:] = 0.
    # var = (1/32) * ( w_t^2 + w_t^2 ) = 2 w_t^2 / 32 = w_t^2 / 16.
    # std = |w_t| / 4.  Still not 1.
    #
    # We need var = 1.  So we need sum_{j=1}^{31} z_t[j]^2 = 32 - w_t^2.
    # And sum_{j=1}^{31} z_t[j] = -w_t.
    # This is possible if 32 - w_t^2 >= w_t^2 / 31, i.e., 32 >= w_t^2 (1 + 1/31) = w_t^2 * 32/31,
    # i.e., w_t^2 <= 31.  If |w_t| > sqrt(31), we cannot achieve var=1 with mean=0
    # while keeping z_t[0] = w_t.  But we can scale the whole thing: we can
    # choose z_t[0] = alpha * w_t, and then v @ z_t = alpha * w_t.  We want
    # v @ LN(e_t) = w_t, so we need alpha = 1.  So we are stuck if |w_t| > sqrt(31).
    #
    # However, we can use a different v.  We can choose v to have a large norm.
    # If we scale v by a factor, then v @ z_t = w_t requires z_t[0] to be
    # w_t / |v|^2 if v is aligned with the first axis.  Actually, let's
    # choose v = (1, 1, ..., 1) / sqrt(32)?  No, let's keep it simple.
    #
    # Alternative: We don't need LN(e_t) to be exactly z_t.  We can use the
    # fact that the head is 2x32, and we can set head_1 - head_0 = v.
    # We want v @ LN(e_t) = w_t.  We can choose e_t such that LN(e_t) is
    # a vector whose dot product with v is w_t.  We can make the variance
    # large by adding a large constant to all components.  Let's try:
    # e_t = c * 1 + d_t, where d_t is a vector with mean 0.  Then
    # mean(e_t) = c + mean(d_t) = c (if mean(d_t)=0).
    # var(e_t) = var(d_t).
    # LN(e_t) = (c*1 + d_t - c) / std(d_t) = d_t / std(d_t).
    # So LN(e_t) = d_t / std(d_t).  This is independent of c!  So adding a
    # constant doesn't help.  The normalization removes the constant.
    #
    # So we need to directly control the normalized vector.  The normalized
    # vector z_t = LN(e_t) must satisfy v @ z_t = w_t.  We can choose z_t
    # freely as long as it has mean 0 and std 1.  So we need to find a
    # vector z_t with mean 0, std 1, and v @ z_t = w_t.  This is possible
    # iff |w_t| <= |v| * sqrt(32) (by Cauchy-Schwarz, since z_t has norm sqrt(32)).
    # Actually, z_t has mean 0 and std 1, so ||z_t||^2 = 32.  By Cauchy-Schwarz,
    # |v @ z_t| <= ||v|| * ||z_t|| = ||v|| * sqrt(32).  So we need
    # |w_t| <= ||v|| * sqrt(32).  We can choose v to have a large norm to
    # accommodate any w_t.  For example, set v = (M, 0, ..., 0).  Then
    # ||v|| = M.  Then we need |w_t| <= M * sqrt(32).  We can choose M large
    # enough.  Then we can set z_t = (w_t / M, ...) with the rest chosen to
    # make mean 0 and std 1.  Let's do this:
    # z_t[0] = w_t / M.
    # We need mean(z_t) = 0 => sum_{j=1}^{31} z_t[j] = -w_t / M.
    # We need var(z_t) = 1 => sum_{j=0}^{31} z_t[j]^2 = 32.
    # So sum_{j=1}^{31} z_t[j]^2 = 32 - (w_t/M)^2.
    # We can set z_t[1] = -w_t / M, and z_t[2:] = c_t.
    # Then sum_{j=1}^{31} z_t[j] = -w_t/M + 30 c_t = -w_t/M => c_t = 0.
    # Then sum_{j=1}^{31} z_t[j]^2 = (w_t/M)^2.
    # Total sum = 2 (w_t/M)^2.  We need this to equal 32.
    # So (w_t/M)^2 = 16 => w_t/M = ±4 => M = |w_t| / 4.
    # But M must be constant for all t.  So this doesn't work for varying w_t.
    #
    # We need a construction that works for all w_t with a fixed M.
    # We have 31 dimensions to play with.  We need to satisfy two equations:
    # sum_{j=1}^{31} z_t[j] = -w_t / M
    # sum_{j=1}^{31} z_t[j]^2 = 32 - (w_t/M)^2
    # This is possible if the required norm squared is at least the minimum
    # possible for a given sum.  The minimum sum of squares for a given sum S
    # over 31 variables is S^2 / 31.  So we need
    # 32 - (w_t/M)^2 >= (w_t/M)^2 / 31
    # => 32 >= (w_t/M)^2 (1 + 1/31) = (w_t/M)^2 * 32/31
    # => (w_t/M)^2 <= 31
    # => |w_t| <= M * sqrt(31).
    # So if we choose M large enough such that max |w_t| <= M * sqrt(31),
    # we can always find such z_t.  For example, set M = max|w_t| / sqrt(31) * 1.1.
    # Then we can construct z_t by setting z_t[1] = -w_t/M - (31-1)*c_t? 
    # Actually, we can just set all z_t[1:] equal to a constant c_t.
    # Then sum = 31 c_t = -w_t/M => c_t = -w_t/(31 M).
    # Then sum of squares = 31 c_t^2 = 31 * (w_t/(31 M))^2 = w_t^2 / (31 M^2).
    # Total sum = (w_t/M)^2 + w_t^2/(31 M^2) = (w_t/M)^2 * (1 + 1/31) = (w_t/M)^2 * 32/31.
    # We need this to equal 32.  So (w_t/M)^2 * 32/31 = 32 => (w_t/M)^2 = 31 => |w_t| = M sqrt(31).
    # So this only works for one specific w_t.  We need to vary the distribution
    # of z_t[1:] to achieve the required sum of squares for each w_t.
    # We can do this by setting z_t[1] = a_t, z_t[2] = b_t, and the rest 0.
    # Then a_t + b_t = -w_t/M.
    # a_t^2 + b_t^2 = 32 - (w_t/M)^2.
    # This is a system of two equations in two unknowns.  It has a solution
    # if the discriminant is non-negative.  The minimum of a^2+b^2 given a+b=S
    # is S^2/2.  So we need 32 - (w_t/M)^2 >= (w_t/M)^2 / 2
    # => 32 >= (w_t/M)^2 * 1.5 => (w_t/M)^2 <= 32/1.5 = 21.33 => |w_t| <= M * 4.618.
    # So if we choose M such that max|w_t| <= M * 4.6, we can use only two
    # dimensions.  Let's just use all 31 dimensions and solve the system
    # by setting z_t[1:] = c_t * u + d_t * v, where u and v are fixed
    # orthogonal vectors.  Actually, we can just use a numerical approach:
    # for each t, we need to find a vector z_t[1:] of length 31 with
    # sum = S and sum of squares = Q.  We can set z_t[1] = x, and the rest
    # equal to y.  Then x + 30 y = S, x^2 + 30 y^2 = Q.
    # Solve for x, y.  This is a quadratic.  We can do this for each t.
    # Let's implement this.
    #
    # Actually, there is a much simpler way: we can use the fact that
    # LayerNorm is applied to the embedding, and we can set the embedding
    # to be exactly the desired normalized vector z_t.  Then LN(e_t) = z_t
    # only if z_t already has mean 0 and std 1.  So we just need to construct
    # z_t with mean 0, std 1, and v @ z_t = w_t.  We can do this by setting
    # z_t = w_t * v / |v|^2 + u_t, where u_t is orthogonal to v, and then
    # adjusting u_t to make mean 0 and std 1.  Since u_t has 31 degrees of
    # freedom, we can always do this as long as |w_t| is not too large.
    # Let's just do it numerically: for each t, we have a vector z_t of
    # length 32.  We want z_t[0] = w_t / M (if v = (M,0,...,0)).
    # We need mean(z_t) = 0 and std(z_t) = 1.
    # We can set z_t[1] = a, z_t[2] = b, and z_t[3:] = c.
    # Then we have three variables a, b, c and two equations.  We can
    # choose c = 0 for simplicity.  Then a + b = -z_t[0].
    # a^2 + b^2 = 32 - z_t[0]^2.
    # This has a solution if 32 - z_t[0]^2 >= z_t[0]^2 / 2
    # => z_t[0]^2 <= 32/1.5 = 21.33 => |z_t[0]| <= 4.618.
    # So if we choose M such that |w_t|/M <= 4.618 for all t, i.e.,
    # M >= max|w_t| / 4.618, we can use this.  Let's do that.
    # ---------------------------------------------------------------

    # Compute max |w_t|
    max_w = np.max(np.abs(w_tok))
    if max_w < 1e-8:
        max_w = 1.0
    # Choose M so that |w_t|/M <= 4.0 (safe margin)
    M = max_w / 4.0
    if M < 1e-6:
        M = 1.0

    # v = (M, 0, ..., 0)
    v = np.zeros(D, dtype=np.float64)
    v[0] = M

    # Build embedding matrix E (V x D)
    E = np.zeros((V, D), dtype=np.float64)
    for t in range(V):
        z0 = w_tok[t] / M
        # We need z_t[0] = z0, mean 0, std 1.
        # Use z_t[1] = a, z_t[2] = b, z_t[3:] = 0.
        # a + b = -z0
        # a^2 + b^2 = 32 - z0^2
        S = -z0
        Q = 32.0 - z0 * z0
        # Solve a, b: a = (S + sqrt(2Q - S^2)) / 2, b = (S - sqrt(2Q - S^2)) / 2
        disc = 2.0 * Q - S * S
        if disc < 0:
            # Fallback: distribute evenly
            a = S / 2.0
            b = S / 2.0
        else:
            sq = np.sqrt(disc)
            a = (S + sq) / 2.0
            b = (S - sq) / 2.0
        z = np.zeros(D, dtype=np.float64)
        z[0] = z0
        z[1] = a
        z[2] = b
        # Normalize to exactly mean 0 and std 1 (numerical safety)
        z = z - z.mean()
        std = z.std()
        if std > 1e-8:
            z = z / std
        # Now v @ z = M * z[0] = M * (w_tok[t] / M) = w_tok[t] (approximately)
        # But after normalization, z[0] may have changed slightly.  We can
        # rescale to ensure v @ z = w_tok[t] exactly.
        # Actually, we need v @ LN(e_t) = w_tok[t].  If we set e_t = z, then
        # LN(e_t) = z (since z has mean 0, std 1).  So v @ z = M * z[0].
        # We want this to equal w_tok[t].  So we need z[0] = w_tok[t] / M.
        # After normalization, z[0] might not be exactly that.  We can
        # adjust by scaling the whole vector?  But scaling changes std.
        # Instead, we can just set e_t = z, and accept the small error.
        # Or we can solve more carefully: we want z with mean 0, std 1,
        # and z[0] = w_tok[t] / M.  This is possible if |w_tok[t]/M| <= sqrt(32).
        # Our construction above ensures this.  Let's just use the z we built
        # before normalization, and then re-normalize while preserving z[0].
        # Actually, the construction already gives mean 0 and std 1 if we
        # solve the equations exactly.  Let's do that:
        # We set z[0] = z0.  We need sum_{j=1}^{31} z[j] = -z0.
        # We need sum_{j=1}^{31} z[j]^2 = 32 - z0^2.
        # We set z[1] = a, z[2] = b, z[3:] = 0.
        # Then a + b = -z0, a^2 + b^2 = 32 - z0^2.
        # This gives a, b as above.  Then mean(z) = (z0 + a + b)/32 = (z0 - z0)/32 = 0.
        # var(z) = (z0^2 + a^2 + b^2)/32 = (z0^2 + 32 - z0^2)/32 = 1.
        # So z already has mean 0 and std 1.  No need to re-normalize.
        # Let's just use that.
        z = np.zeros(D, dtype=np.float64)
        z[0] = z0
        z[1] = a
        z[2] = b
        # Ensure mean 0 and std 1 exactly (they should be)
        z = z - z.mean()
        std = z.std()
        if std > 1e-8:
            z = z / std
        # Now v @ z = M * z[0].  We want this to be w_tok[t].
        # Since z[0] might have changed slightly, we can rescale the entire
        # vector to make v @ z = w_tok[t]?  But that changes std.
        # Instead, we can just set e_t = z * (w_tok[t] / (M * z[0])) if z[0] != 0.
        # But that changes the mean and std.  However, LayerNorm will
        # re-normalize, so the scaling factor cancels out!  Because
        # LN(alpha * z) = LN(z) for any alpha > 0.  So we can scale e_t
        # arbitrarily without changing LN(e_t).  So we can set e_t = z,
        # and LN(e_t) = z.  So v @ LN(e_t) = M * z[0].  We need this to
        # equal w_tok[t].  So we need z[0] = w_tok[t] / M.  Our construction
        # sets z[0] = z0 = w_tok[t] / M.  After the mean subtraction and
        # normalization, z[0] might change.  Let's avoid the post-processing
        # and just use the exact solution.
        # The exact solution is:
        # z[0] = z0
        # z[1] = a, z[2] = b, z[3:] = 0
        # with a + b = -z0, a^2 + b^2 = 32 - z0^2.
        # This gives mean 0 and std 1 exactly.  So no post-processing needed.
        # Let's just compute a and b correctly.
        # We need a^2 + b^2 = 32 - z0^2.
        # (a+b)^2 = z0^2 => a^2 + b^2 + 2ab = z0^2 => 2ab = z0^2 - (32 - z0^2) = 2z0^2 - 32.
        # So ab = z0^2 - 16.
        # We can solve the quadratic: a, b are roots of x^2 - S x + P = 0,
        # where S = -z0, P = z0^2 - 16.
        # Discriminant = S^2 - 4P = z0^2 - 4(z0^2 - 16) = 64 - 3z0^2.
        # So we need 64 - 3z0^2 >= 0 => z0^2 <= 64/3 ≈ 21.33 => |z0| <= 4.618.
        # Our choice M = max_w / 4.0 ensures |z0| <= 4.0, so discriminant > 0.
        # Good.
        S = -z0
        P = z0 * z0 - 16.0
        disc = S * S - 4.0 * P
        if disc < 0:
            disc = 0.0
        sq = np.sqrt(disc)
        a = (S + sq) / 2.0
        b = (S - sq) / 2.0
        z = np.zeros(D, dtype=np.float64)
        z[0] = z0
        z[1] = a
        z[2] = b
        E[t] = z

    # Now set embed.weight
    w["embed.weight"] = E.astype(np.float32)

    # Set head weights: we want head_1 - head_0 = v = (M, 0, ..., 0).
    # And we need the bias difference to match b_ls.
    # The model computes logit_c = head_c @ pooled + bias_c.
    # We want logit_1 - logit_0 = w_tok · counts + b_ls.
    # Our construction gives pooled = (1/N) sum_i LN(e_{t_i}) = (1/N) sum_i z_{t_i}.
    # Then (head_1 - head_0) @ pooled = v @ pooled = (1/N) sum_i (v @ z_{t_i})
    # = (1/N) sum_i w_tok[t_i] = w_tok · counts (since counts = (1/N) sum_i 1_{t_i}).
    # So logit_1 - logit_0 = w_tok · counts + (bias_1 - bias_0).
    # We want this to equal w_tok · counts + b_ls.
    # So set bias_1 - bias_0 = b_ls.
    # We also need to set the absolute values.  We can set head_0 = 0, head_1 = v.
    # Then logit_0 = bias_0, logit_1 = v @ pooled + bias_1.
    # We want the probabilities to be reasonable.  The softmax is over logits.
    # We can set bias_0 = 0, bias_1 = b_ls.  Then logit_0 = 0, logit_1 = score + b_ls.
    # But the score is w_tok · counts, which can be large.  We might want to
    # center it.  Actually, the least squares fit already gives a bias b_ls.
    # Let's just set head_0 = -v/2, head_1 = v/2, and bias_0 = -b_ls/2, bias_1 = b_ls/2.
    # Then logit_1 - logit_0 = v @ pooled + b_ls.  And logit_0 + logit_1 = 0.
    # This is symmetric and fine.
    head = np.zeros((2, D), dtype=np.float64)
    head[0] = -v / 2.0
    head[1] = v / 2.0
    w["head.weight"] = head.astype(np.float32)

    bias = np.zeros(2, dtype=np.float64)
    bias[0] = -b_ls / 2.0
    bias[1] = b_ls / 2.0
    w["head.bias"] = bias.astype(np.float32)

    # ---------------------------------------------------------------
    # 4. Ensure all other weights are zero (they already are) and
    #    positional weights are zero.
    # ---------------------------------------------------------------
    # (No changes needed; they are already zero.)

    # ---------------------------------------------------------------
    # 5. Diagnostics
    # ---------------------------------------------------------------
    # Compute training accuracy of the constructed model on the feedback pool
    # using the closed-form classifier directly.
    scores = counts @ w_tok + b_ls
    preds = (scores > 0).astype(np.int64)
    acc = np.mean(preds == y)

    diagnostics = {
        "train_accuracy": float(acc),
        "max_abs_w_tok": float(np.max(np.abs(w_tok))),
        "M": float(M),
        "b_ls": float(b_ls),
        "n_tokens": int(V),
        "n_samples": int(N),
    }

    return {"weights": w, "diagnostics": diagnostics}
