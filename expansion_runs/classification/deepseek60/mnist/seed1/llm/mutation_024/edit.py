import numpy as np

def edit(w, f):
    # Extract current fc2 parameters
    W2 = w["fc2.weight"].copy()  # (10, 128)
    b2 = w["fc2.bias"].copy()    # (10,)

    # Full-pool hidden features and labels
    H = f["features"].astype(np.float32)  # (N, 128)
    y = f["y"].astype(np.int64)           # (N,)
    N = H.shape[0]
    C = 10

    # Current logits and probabilities
    L = f["logits"].astype(np.float32)    # (N, 10)
    P = f["probs"].astype(np.float32)     # (N, 10)

    # One-hot targets
    Y = np.zeros((N, C), dtype=np.float32)
    Y[np.arange(N), y] = 1.0

    # Gradient of cross-entropy w.r.t. W2 and b2
    # dL/dW2 = (P - Y)^T H / N   -> shape (10, 128)
    # dL/db2 = (P - Y).sum(0) / N -> shape (10,)
    G = P - Y
    grad_W = (G.T @ H) / N
    grad_b = G.mean(axis=0)

    # Build the Hessian for the softmax layer.
    # For each sample i, the Hessian w.r.t. the logits is
    # diag(P_i) - P_i P_i^T.
    # The full Hessian w.r.t. W2 (flattened as (10*128,)) is
    # (1/N) * sum_i (H_i H_i^T) ⊗ (diag(P_i) - P_i P_i^T)
    # where ⊗ is Kronecker product.  This is 1280 x 1280.
    # We will build it efficiently and add a ridge term.

    # Precompute H_i H_i^T for all i? That would be N x 128 x 128 = 2000*16384 = 32M floats, okay but we can do it in blocks.
    # Instead, build the full Hessian by accumulating over samples.
    D = 128
    K = C * D  # 1280
    Hmat = np.zeros((K, K), dtype=np.float32)

    # To save memory, we can compute the Hessian using the fact that
    # for each sample, the block for classes c,d is (P_c δ_cd - P_c P_d) * h h^T.
    # We can accumulate the 128x128 matrices for each (c,d) pair.
    # There are 10*10 = 100 blocks, each 128x128. Total 100*16384 = 1.6M floats, fine.
    # We'll accumulate M_cd = sum_i (P_ic δ_cd - P_ic P_id) h_i h_i^T.
    # Then Hessian block (c,d) = M_cd / N.

    # Precompute h_i h_i^T? We can do it by looping over samples, but N=2000 is fine.
    # Actually we can compute M_cd = H^T diag(w_i) H where w_i = P_ic δ_cd - P_ic P_id.
    # For each c,d, we can compute weights vector w = P[:,c] * ( (c==d) - P[:,d] ), then M_cd = H.T @ (w[:,None] * H).
    # That's 100 matrix multiplications of (128x2000) @ (2000x128) = 128x128. 100 * 128*128*2000 = 3.2e9 flops, a bit heavy but okay in numpy? Maybe too slow.
    # Better: compute the full Hessian using the Kronecker structure.
    # Let S = H.T @ H / N (128x128). But the weights depend on P, so it's not separable.
    # We can compute the Hessian as:
    # For each class c, the diagonal block is H^T diag(P_c (1-P_c)) H / N.
    # The off-diagonal block (c,d) is - H^T diag(P_c P_d) H / N.
    # We can compute these 100 blocks. 100 * (128x2000 @ 2000x128) is 100 * 128*128*2000 = 3.2e9 multiply-adds. In numpy, this might take a few seconds. Acceptable.

    # To be efficient, we can compute the weighted Gram matrices.
    # We'll compute for each c, the matrix H_c = H * sqrt(P_c (1-P_c))? Not exactly because off-diagonals need P_c P_d.
    # Let's just compute the 100 blocks using matrix multiplication.
    # We can precompute H (N,128). For each c,d, w = P[:,c] * ((c==d) - P[:,d]). Then M_cd = H.T @ (w[:,None] * H).
    # This is 100 * (128x2000 @ 2000x128) = 100 * 128*128*2000 = 3.2e9. Might be okay but let's optimize.
    # We can compute all blocks at once by forming a (N, 10) weight matrix for each d? Actually for a fixed d, we need for all c: w_c = P[:,c] * ((c==d) - P[:,d]).
    # So for each d, we can compute M_cd for all c. That's 10 * 10 = 100 blocks. Still same.
    # We can compute the full Hessian by building a (N, 10, 128) tensor? That's 2000*10*128 = 2.56M floats, fine.
    # Let's do it directly: for each sample i, compute the 10x10 matrix A_i = diag(P_i) - P_i P_i^T.
    # Then the Hessian is sum_i A_i ⊗ (h_i h_i^T) / N.
    # We can accumulate the 1280x1280 matrix by looping over samples. 2000 iterations, each adding a 10x10 Kronecker with 128x128. That's 2000 * 10*10*128*128 = 2000 * 1.6M = 3.2e9 operations. Similar.
    # Let's just use the block approach with matrix multiplications. It's simpler.

    # Precompute H (N,128)
    # We'll compute M_cd for c,d in 0..9.
    # To speed up, we can compute for each c, the matrix H_c = H * P[:,c][:,None] (N,128). Then M_cd = H_c.T @ H_d for c!=d? Not exactly.
    # M_cd = sum_i (P_ic δ_cd - P_ic P_id) h_i h_i^T = δ_cd sum_i P_ic h_i h_i^T - sum_i P_ic P_id h_i h_i^T.
    # Let A_c = H.T @ (P[:,c][:,None] * H)  (128x128). This is sum_i P_ic h_i h_i^T.
    # Let B_cd = H.T @ (P[:,c][:,None] * P[:,d][:,None] * H) = sum_i P_ic P_id h_i h_i^T.
    # Then M_cd = δ_cd A_c - B_cd.
    # We can compute A_c for each c (10 matrices). And B_cd for each c,d (100 matrices). But B_cd = B_dc, so 55 matrices.
    # Computing A_c: 10 * (128x2000 @ 2000x128) = 10 * 128*128*2000 = 327M ops.
    # Computing B_cd: 55 * 128*128*2000 = 1.8B ops. Total ~2.1B ops. In numpy, this might take a few seconds but okay.
    # Alternatively, we can compute the full Hessian by using the fact that B_cd = H.T @ diag(P_c P_d) H. We can compute all B_cd by forming a matrix of weighted H? 
    # Let's just do the loop over c,d. 100 iterations of 128x2000 @ 2000x128. That's 100 * 128*128*2000 = 3.2e9. Might be 10-20 seconds. Acceptable for a single edit? Probably okay.
    # But we can optimize: For each d, we can compute B_cd for all c at once by forming a (N, 10) matrix of weights? Actually B_cd = H.T @ (P[:,c] * P[:,d])[:,None] * H. For fixed d, we can compute for all c: we need sum_i P_ic P_id h_i h_i^T. This is like for each d, compute the weighted Gram matrix with weights P_ic P_id. We can do: for each d, let W_d = P * P[:,d][:,None] (N,10). Then for each c, B_cd = H.T @ (W_d[:,c][:,None] * H). That's still 10*10.
    # Let's just use the sample loop but vectorized over classes? 
    # Actually, we can compute the Hessian using the following:
    # Let H (N,128). Let P (N,10). 
    # The Hessian is (1/N) * sum_i (h_i h_i^T) ⊗ (diag(P_i) - P_i P_i^T).
    # We can compute the 1280x1280 matrix by:
    # For each i, compute A_i = diag(P_i) - P_i P_i^T (10x10). Then add np.kron(A_i, h_i[:,None] @ h_i[None,:]) / N.
    # 2000 iterations of np.kron(10x10, 128x128) = 1280x1280. That's 2000 * 1.6M = 3.2B operations. Might be slow in Python loop.
    # Better to vectorize: We can compute the full Hessian as:
    # Hessian = (1/N) * (H^T H) ⊗ (diag(mean(P)) - mean(P P^T))? No, because P varies per sample.
    # We can use the fact that the Hessian is a sum of Kronecker products. We can compute it by reshaping.
    # Let's just use the block approach with matrix multiplications. It's clean and fast enough.

    # Compute A_c = H.T @ (P[:,c][:,None] * H) for c=0..9
    A = np.zeros((C, D, D), dtype=np.float32)
    for c in range(C):
        w_c = P[:, c]
        A[c] = H.T @ (w_c[:, None] * H)

    # Compute B_cd = H.T @ (P[:,c][:,None] * P[:,d][:,None] * H)
    # We can compute B for all c,d by using the fact that B_cd = H.T @ diag(P_c P_d) H.
    # We can compute for each c, the matrix H_c = H * P[:,c][:,None] (N,128). Then B_cd = H_c.T @ H_d.
    # Let's compute H_c for all c: shape (C, N, D). That's 10*2000*128 = 2.56M floats, fine.
    H_c = np.zeros((C, N, D), dtype=np.float32)
    for c in range(C):
        H_c[c] = H * P[:, c][:, None]
    # Now B_cd = H_c[c].T @ H_c[d] for each c,d. We can compute B = np.einsum('cnd,dne->cde', H_c, H_c) ? Actually H_c[c] is (N,D). B_cd = H_c[c].T @ H_c[d] -> (D,D). So B = np.einsum('cnd,dne->cde', H_c, H_c). That's 10*10*128*128*2000 = 3.2e9 ops. np.einsum might be slow. We can do it with a loop over c,d. 100 iterations of (128x2000 @ 2000x128) is 100 * 128*128*2000 = 3.2e9. Let's just do the loop, it's simple.

    B = np.zeros((C, C, D, D), dtype=np.float32)
    for c in range(C):
        for d in range(C):
            B[c, d] = H_c[c].T @ H_c[d]

    # Now build the full Hessian matrix (K, K) where K = C*D.
    # The block for (c,d) is (δ_cd A_c - B_cd) / N.
    Hess = np.zeros((K, K), dtype=np.float32)
    for c in range(C):
        for d in range(C):
            block = (A[c] if c == d else 0.0) - B[c, d]
            Hess[c*D:(c+1)*D, d*D:(d+1)*D] = block / N

    # Add ridge regularisation
    lam = 1e-3  # small ridge
    Hess += lam * np.eye(K, dtype=np.float32)

    # Gradient vector (flattened)
    grad = np.concatenate([grad_W.flatten(), grad_b.flatten()])  # length K? Wait, grad_W is (10,128) and grad_b is (10,). The parameter vector is W2 (10,128) and b2 (10). Total 1280 + 10 = 1290. But our Hessian above is only for W2 (1280x1280). We also need the bias part. The bias is just another set of parameters. We can treat the bias as an extra feature of 1. So we can augment H with a column of ones. Let's do that to include bias in the same framework.

    # Let's redo with augmented features: H_aug = [H, 1] (N, 129). Then W_aug = [W2, b2] (10, 129). Total parameters = 10*129 = 1290.
    # The Hessian is (1290, 1290). We can compute similarly.
    # Let's restart with augmented features to keep it simple.

    # Augmented features
    H_aug = np.concatenate([H, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N, 129)
    D_aug = H_aug.shape[1]  # 129
    K_aug = C * D_aug  # 1290

    # Current parameters flattened
    W_aug = np.concatenate([W2, b2[:, None]], axis=1)  # (10, 129)
    # Current logits L = H_aug @ W_aug.T
    # We already have L from f["logits"], but let's verify: L = H_aug @ W_aug.T
    # Actually f["logits"] is given. We can use it.

    # Gradient w.r.t. W_aug: (P - Y)^T H_aug / N -> (10, 129)
    grad_aug = (G.T @ H_aug) / N  # (10, 129)

    # Hessian blocks for augmented features
    # A_c = H_aug.T @ (P[:,c][:,None] * H_aug) -> (129, 129)
    A_aug = np.zeros((C, D_aug, D_aug), dtype=np.float32)
    for c in range(C):
        w_c = P[:, c]
        A_aug[c] = H_aug.T @ (w_c[:, None] * H_aug)

    # B_cd = H_aug.T @ (P[:,c][:,None] * P[:,d][:,None] * H_aug)
    # We can compute H_c_aug = H_aug * P[:,c][:,None] (N, 129)
    H_c_aug = np.zeros((C, N, D_aug), dtype=np.float32)
    for c in range(C):
        H_c_aug[c] = H_aug * P[:, c][:, None]

    B_aug = np.zeros((C, C, D_aug, D_aug), dtype=np.float32)
    for c in range(C):
        for d in range(C):
            B_aug[c, d] = H_c_aug[c].T @ H_c_aug[d]

    # Build full Hessian (K_aug, K_aug)
    Hess_aug = np.zeros((K_aug, K_aug), dtype=np.float32)
    for c in range(C):
        for d in range(C):
            block = (A_aug[c] if c == d else 0.0) - B_aug[c, d]
            Hess_aug[c*D_aug:(c+1)*D_aug, d*D_aug:(d+1)*D_aug] = block / N

    # Add ridge
    lam = 1e-3
    Hess_aug += lam * np.eye(K_aug, dtype=np.float32)

    # Gradient vector
    grad_vec = grad_aug.flatten()  # length K_aug

    # Solve for Newton step: delta = - Hess^{-1} grad
    # We can use np.linalg.solve. The matrix is 1290x1290, which is fine.
    try:
        delta = np.linalg.solve(Hess_aug, -grad_vec)
    except np.linalg.LinAlgError:
        # fallback to pseudo-inverse
        delta = -np.linalg.pinv(Hess_aug) @ grad_vec

    # Reshape delta to (10, 129)
    delta_W = delta.reshape(C, D_aug)

    # Apply a step size. We can choose a small step to be safe.
    # Since this is a Newton step, step=1 is theoretically optimal for the quadratic model.
    # But to avoid overshooting, we can use a smaller step. Let's use step=0.5.
    step = 0.5
    W_aug_new = W_aug + step * delta_W

    # Extract new W2 and b2
    W2_new = W_aug_new[:, :D].astype(np.float32)
    b2_new = W_aug_new[:, D].astype(np.float32)

    # Clip to ensure finite and within bounds
    W2_new = np.clip(W2_new, -1000.0, 1000.0)
    b2_new = np.clip(b2_new, -1000.0, 1000.0)

    # Update weights
    w["fc2.weight"] = W2_new
    w["fc2.bias"] = b2_new

    # Diagnostics
    # Compute new logits on full pool to report CE and accuracy
    L_new = H_aug @ W_aug_new.T
    # Softmax
    L_new_max = L_new.max(axis=1, keepdims=True)
    exp_L = np.exp(L_new - L_new_max)
    P_new = exp_L / exp_L.sum(axis=1, keepdims=True)
    acc = np.mean(np.argmax(L_new, axis=1) == y)
    ce = -np.mean(np.log(P_new[np.arange(N), y] + 1e-12))

    diagnostics = {
        "full_pool_accuracy": float(acc),
        "full_pool_cross_entropy": float(ce),
        "step": step,
        "lambda": lam,
        "delta_norm": float(np.linalg.norm(delta)),
        "fc2_weight_norm": float(np.linalg.norm(W2_new)),
        "fc2_bias_norm": float(np.linalg.norm(b2_new)),
    }

    return {"weights": w, "diagnostics": diagnostics}
