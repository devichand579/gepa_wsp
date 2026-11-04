import numpy as np

def edit(w, f):
    # ---------------- data ----------------
    x = np.asarray(f["x"], dtype=np.float64)      # (N,1,28,28)
    y = np.asarray(f["y"], dtype=np.int64)
    N = x.shape[0]
    X = x[:, 0]                                    # (N,28,28)
    Y1 = np.zeros((N, 10)); Y1[np.arange(N), y] = 1.0

    c1w = np.asarray(w["conv1.weight"], dtype=np.float64)   # (16,1,3,3)
    c1b = np.asarray(w["conv1.bias"], dtype=np.float64)
    c2w = np.asarray(w["conv2.weight"], dtype=np.float64)   # (32,16,3,3)
    c2b = np.asarray(w["conv2.bias"], dtype=np.float64)
    hw  = np.asarray(w["head.weight"], dtype=np.float64)    # (10,1568)
    hb  = np.asarray(w["head.bias"], dtype=np.float64)

    # ---------------- numpy forward (verify vs supplied) ----------------
    def pad1(A):
        return np.pad(A, ((0,0),(1,1),(1,1),(0,0)), mode="constant")

    def conv3x3(A, Wt, b):
        # A: (N,C,H,W) ; Wt: (O,C,3,3) ; b: (O,)
        Ap = pad1(A)
        N, C, H, Wd = Ap.shape
        O = Wt.shape[0]
        cols = np.empty((N, C, 9, H, Wd), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                cols[:, :, i*3+j] = Ap[:, :, i:i+H, j:j+Wd]
        out = np.tensordot(cols, Wt, axes=([1,2],[1,2])).reshape(N, O, H, Wd)
        return out + b[None, :, None, None]

    def relu(A):
        return np.maximum(A, 0.0)

    def maxpool2(A):
        N, C, H, Wd = A.shape
        A = A[:, :, :H//2*2, :Wd//2*2]
        A = A.reshape(N, C, H//2, 2, Wd//2, 2).transpose(0,1,2,4,3,5).reshape(N, C, H//2, Wd//2, 4)
        return A.max(axis=-1)

    def forward(c1w_, c1b_, c2w_, c2b_, hw_, hb_):
        h1 = relu(conv3x3(X[:, None], c1w_, c1b_))
        p1 = maxpool2(h1)                                   # (N,16,14,14)
        h2 = relu(conv3x3(p1, c2w_, c2b_))
        p2 = maxpool2(h2)                                   # (N,32,7,7)
        feat = p2.reshape(N, -1)
        logits = feat @ hw_.T + hb_[None, :]
        return p1, p2, feat, logits

    p1, p2, feat_cur, logits_cur = forward(c1w, c1b, c2w, c2b, hw, hb)
    sup_feat = np.asarray(f["features"], dtype=np.float64)
    sup_log  = np.asarray(f["logits"], dtype=np.float64)
    err_feat = float(np.max(np.abs(feat_cur - sup_feat)))
    err_log  = float(np.max(np.abs(logits_cur - sup_log)))
    acc_cur  = float((logits_cur.argmax(1) == y).mean())

    # ---------------- stage 1: local 16-channel discriminant ----------------
    # class means of the conv1 pooled responses (class-moment construction)
    P1 = p1.reshape(N, 16, -1)                              # (N,16,196)
    mu_c = np.zeros((10, 16))
    for c in range(10):
        mu_c[c] = P1[y == c].mean(axis=(0, 2))              # (16,)
    # pooled within-class covariance of the 16-dim local vectors
    S = np.zeros((16, 16))
    for c in range(10):
        Z = P1[y == c] - mu_c[c][None, :, None]             # (Nc,16,196)
        S += np.einsum('nci,ncj->ij', Z, Z)
    S /= float(N - 10)
    tr = float(np.trace(S)) / 16.0
    lam = 0.35 * max(tr, 1e-8)
    Sc = S + lam * np.eye(16)
    V = np.linalg.solve(Sc, mu_c.T).T                       # (10,16) local discriminants
    V = V - V.mean(axis=0, keepdims=True)                   # remove common component
    # scale so that each class map has comparable dynamic range
    g = p1.reshape(N, 16, -1).std(axis=(0, 2)) + 1e-8
    Vs = V / g[None, :]

    # ---------------- stage 2: new conv2 = point-wise local discriminant ----------------
    # channel 0..9  : class-c local discriminant (delta spatial kernel)
    # channel 10..25: point-wise copies of conv1 channels 0..15 (sharper evidence)
    # channel 26..31: point-wise difference pairs of conv1 channels
    c2w_new = np.zeros((32, 16, 3, 3), dtype=np.float64)
    c2b_new = np.zeros(32, dtype=np.float64)
    for c in range(10):
        c2w_new[c, :, 1, 1] = Vs[c]
    for k in range(16):
        c2w_new[10 + k, k, 1, 1] = 1.0
    for k in range(6):
        c2w_new[26 + k, k, 1, 1] = 1.0
        c2w_new[26 + k, 10 + k, 1, 1] = -1.0

    # ---------------- stage 3: new features + ridge head (normal equations) ----------------
    def feats_from(c2w_, c2b_):
        h2 = relu(conv3x3(p1, c2w_, c2b_))
        return maxpool2(h2).reshape(N, -1)

    F = feats_from(c2w_new, c2b_new)
    F = np.concatenate([F, np.ones((N, 1))], axis=1)        # bias column
    G = F.T @ F
    ridge = 3.0 * float(np.trace(G)) / G.shape[0]
    G = G + ridge * np.eye(G.shape[0])
    G[-1, -1] -= ridge                                     # do not penalise bias
    B = np.linalg.solve(G, F.T @ Y1)                        # (1569,10)
    W_new = B[:-1].T.astype(np.float32)                     # (10,1568)
    b_new = B[-1].astype(np.float32)                        # (10,)

    # ---------------- stage 4: sanity check on the training pool ----------------
    logits_new = F @ B
    acc_new = float((logits_new.argmax(1) == y).mean())
    ce_new = float(-np.mean(np.log(np.clip(logits_new - logits_new.max(1, keepdims=True), -60, 0).sum(1)
                                  + logits_new.max(1))))
    # per-class accuracy of the new head
    pc = []
    for c in range(10):
        m = (y == c)
        pc.append(float((logits_new[m].argmax(1) == c).mean()))

    # ---------------- stage 5: write back ----------------
    w["conv2.weight"] = c2w_new.astype(np.float32)
    w["conv2.bias"] = c2b_new.astype(np.float32)
    w["head.weight"] = np.clip(W_new, -1000.0, 1000.0)
    w["head.bias"] = np.clip(b_new, -1000.0, 1000.0)

    # per-class accuracy of the OLD head for comparison
    pc_old = []
    for c in range(10):
        m = (y == c)
        pc_old.append(float((logits_cur[m].argmax(1) == c).mean()))

    diag = {
        "forward_check_max_abs_feat_err": err_feat,
        "forward_check_max_abs_logit_err": err_log,
        "pool_acc_old": acc_cur,
        "pool_acc_new": acc_new,
        "pool_ce_new": ce_new,
        "per_class_acc_old": [round(v, 4) for v in pc_old],
        "per_class_acc_new": [round(v, 4) for v in pc],
        "ridge_lambda_scale": ridge,
        "cov_shrink_lambda": lam,
        "local_discriminant_row_norms": [round(float(np.linalg.norm(Vs[c])), 4) for c in range(10)],
        "new_feat_mean": round(float(F[:, :-1].mean()), 5),
        "new_feat_std": round(float(F[:, :-1].std()), 5),
        "head_w_max_abs": round(float(np.abs(W_new).max()), 4),
    }
    return {"weights": w, "diagnostics": diag}
