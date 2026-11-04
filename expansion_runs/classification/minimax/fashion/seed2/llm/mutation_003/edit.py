def edit(w, f):
    import numpy as np

    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float32)          # (2000,1,28,28)
    y = np.asarray(f["y"]).astype(np.int64).ravel()    # (2000,)
    N = x.shape[0]
    X = x.reshape(N, 28, 28)

    # ---------------- helpers ----------------
    def im2col3(P, H, W):                 # P:(N,H,W) -> (N,H,W,9) patches, pad=1
        Pp = np.zeros((N, H + 2, W + 2), dtype=np.float32)
        Pp[:, 1:H+1, 1:W+1] = P
        out = np.empty((N, H, W, 9), dtype=np.float32)
        for kr in range(3):
            for kc in range(3):
                out[:, :, :, kr*3+kc] = Pp[:, kr:kr+H, kc:kc+W]
        return out

    def maxpool2(A):                     # A:(N,C,H,W) -> (N,C,H//2,W//2)
        N, C, H, W = A.shape
        return A[:, :, 0:H:2, 0:W:2].reshape(N, C, H//2, 2, W//2, 2).max(axis=(3, 5))

    def pca_filters(P, k, seed):         # P:(M,d) -> (k,d) orthonormal rows
        mu = P.mean(axis=0)
        Pc = P - mu
        C = (Pc.T @ Pc) / max(1, Pc.shape[0] - 1)
        C = 0.5 * (C + C.T)
        ev, evec = np.linalg.eigh(C)
        order = np.argsort(ev)[::-1]
        evec = evec[:, order]
        s = np.sqrt(np.maximum(ev[order], 0.0))
        B = evec[:, :k].T                     # (k,d)
        B = B / np.maximum(s[:k], 1e-6)[:, None]
        # fix sign deterministically
        for i in range(B.shape[0]):
            j = int(np.argmax(np.abs(B[i])))
            if B[i, j] < 0:
                B[i] = -B[i]
        return B.astype(np.float32), mu.astype(np.float32)

    # ---------------- conv1: PCA of 3x3 image patches ----------------
    pat1 = im2col3(X, 28, 28).reshape(-1, 9)          # (N*784, 9)
    B1, mu1 = pca_filters(pat1, 9, 0)                # 9 meaningful filters
    rng = np.random.RandomState(0)
    extra = rng.randn(7, 9).astype(np.float32)
    extra /= np.maximum(np.linalg.norm(extra, axis=1, keepdims=True), 1e-6)
    W1 = np.concatenate([B1, extra], axis=0)          # (16,9)
    b1 = (-mu1 @ W1.T).astype(np.float32)             # center patches

    # forward conv1 + relu + maxpool
    P1 = im2col3(X, 28, 28)                          # (N,28,28,9)
    c1 = P1 @ W1.T + b1                              # (N,28,28,16)
    c1 = np.maximum(c1, 0.0)
    c1 = c1.reshape(N, 28, 28, 16).transpose(0, 3, 1, 2)   # (N,16,28,28)
    p1 = maxpool2(c1)                                     # (N,16,14,14)

    # ---------------- conv2: PCA of 144-d patches of pooled conv1 ----------------
    # build (N,14,14,144) patch tensor from p1
    Pt = np.zeros((N, 16, 14 + 2, 14 + 2), dtype=np.float32)
    Pt[:, :, 1:15, 1:15] = p1
    pat2 = np.empty((N, 14, 14, 144), dtype=np.float32)
    for ic in range(16):
        for kr in range(3):
            for kc in range(3):
                pat2[:, :, :, ic*9 + kr*3 + kc] = Pt[:, ic, kr:kr+14, kc:kc+14]
    flat2 = pat2.reshape(-1, 144)
    B2, mu2 = pca_filters(flat2, 32, 1)
    W2 = B2.reshape(32, 16, 3, 3).astype(np.float32)
    b2 = (-mu2 @ B2.T).astype(np.float32)

    # forward conv2 + relu + maxpool -> features
    c2 = np.einsum('nhwijk,oikl->nohwjl', pat2, W2.reshape(32, 16, 9)) + b2[None, :, None, None]
    c2 = np.maximum(c2, 0.0)
    c2 = c2.transpose(0, 1, 2, 4, 3)                  # (N,32,14,14,3,3)
    c2 = c2.reshape(N, 32, 14, 14, 9)
    c2 = np.einsum('nchwij,kij->nchw', c2, np.eye(9, dtype=np.float32).reshape(3, 3, 9))
    # simpler: direct 3x3 conv over (N,16,14,14)
    c2b = np.zeros((N, 32, 14, 14), dtype=np.float32)
    for o in range(32):
        acc = np.zeros((N, 14, 14), dtype=np.float32)
        for ic in range(16):
            for kr in range(3):
                for kc in range(3):
                    acc += c2[:, ic, kr:kr+14, kc:kc+14] * W2[o, ic, kr, kc]
        c2b[:, o] = acc
    c2b = np.maximum(c2b + b2[None, :, None, None], 0.0)
    p2 = maxpool2(c2b)                               # (N,32,7,7)
    F = p2.reshape(N, 1568).astype(np.float64)       # features (channel,row,col)

    # ---------------- closed-form ridge head on features ----------------
    Y1 = np.zeros((N, 10), dtype=np.float64)
    Y1[np.arange(N), y] = 1.0
    fbar = F.mean(axis=0)
    Fm = F - fbar
    ybar = Y1.mean(axis=0)
    T = Y1 - ybar
    G = Fm.T @ Fm
    ridge = 1e-2 * np.trace(G) / max(1, G.shape[0]) + 1e-8
    A = G + ridge * np.eye(G.shape[0])
    Whead = np.linalg.solve(A, Fm.T @ T)             # (1568,10)
    bhead = ybar - fbar @ Whead

    # ---------------- write back ----------------
    w["conv1.weight"][...] = W1.reshape(16, 1, 3, 3).astype(np.float32)
    w["conv1.bias"][...] = b1.astype(np.float32)
    w["conv2.weight"][...] = W2.astype(np.float32)
    w["conv2.bias"][...] = b2.astype(np.float32)
    w["head.weight"][...] = Whead.T.astype(np.float32)
    w["head.bias"][...] = bhead.astype(np.float32)

    # ---------------- diagnostics ----------------
    logits = F @ Whead + bhead
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    nz = float((F > 0).mean())
    per_class = [float((pred[y == c] == c).mean()) for c in range(10)]

    diag = {
        "train_acc_ridge_head": acc,
        "feature_zero_frac": nz,
        "feature_mean": float(F.mean()),
        "feature_std": float(F.std()),
        "per_class_acc": per_class,
        "ridge": float(ridge),
        "conv1_w_std": float(W1.std()),
        "conv2_w_std": float(W2.std()),
        "head_w_std": float(Whead.std()),
        "head_w_absmax": float(np.abs(Whead).max()),
        "logit_std": float(logits.std()),
    }
    return {"weights": w, "diagnostics": diag}
