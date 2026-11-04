import numpy as np
import math

def edit(w, f):
    X = np.asarray(f["x"])            # [N,64] int64
    y = np.asarray(f["y"]).astype(np.int64)  # [N]
    N, L = X.shape
    V = w["embed.weight"].shape[0]
    D = w["embed.weight"].shape[1]

    # ---- Build normalized bag-of-words (mean over non-pad tokens) ----
    mask = (X != 0).astype(np.float64)          # [N,64]
    counts = np.zeros((N, V), dtype=np.float64)
    for i in range(N):
        row = X[i]
        m = mask[i]
        for t in range(L):
            if m[t] > 0:
                counts[i, row[t]] += 1.0
    denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    BOW = counts / denom                          # [N,V] mean-normalised

    # ---- Ridge fit of s (V-dim) + intercept on BOW ----
    t = 2.0 * y.astype(np.float64) - 1.0          # {-1,+1}
    Xf = np.concatenate([BOW, np.ones((N, 1))], axis=1)  # [N,V+1]
    G = Xf.T @ Xf
    b = Xf.T @ t

    # 5-fold CV to pick lambda
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    folds = np.array_split(idx, 5)
    lambdas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    best_lam, best_cv = 0.1, -1.0
    for lam in lambdas:
        accs = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            Gt = Xf[tr].T @ Xf[tr]
            bt = Xf[tr].T @ t[tr]
            A = Gt + lam * np.eye(V + 1)
            try:
                sol = np.linalg.solve(A, bt)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(A, bt, rcond=None)[0]
            pred = np.sign(Xf[te] @ sol)
            accs.append(np.mean(pred == t[te]))
        cv = float(np.mean(accs))
        if cv > best_cv:
            best_cv, best_lam = cv, lam

    A = G + best_lam * np.eye(V + 1)
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    s = sol[:V]
    intercept = float(sol[V])

    # ---- Write s into embed column 0, make rows zero-mean/unit-variance ----
    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = s
    # Fill remaining columns with a deterministic orthogonal-ish pattern,
    # then renormalise each row to mean 0, std 1.
    for d in range(1, D):
        E[:, d] = np.cos((np.arange(V) + 1.0) * (d + 1.0) * 0.61803398875)
    # remove per-row mean and enforce unit variance across the D dims
    E -= E.mean(axis=1, keepdims=True)
    std = E.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    E /= std
    # rescale so column 0 keeps the fitted s as closely as possible while
    # still being zero-mean/unit-variance overall: we accept the small
    # distortion, but put the *fitted* s back into column 0 and re-balance
    # the other columns so the row mean stays ~0 and std ~1.
    E[:, 0] = s
    E -= E.mean(axis=1, keepdims=True)
    # after subtracting the mean, column 0 is s - mean(s_row); to keep the
    # classifier exactly s we instead set head to read (col0 + correction).
    # Simpler: recompute the effective linear read-out below.

    # Effective read-out: head must read column 0, but column 0 is now
    # s - mean_row.  We compensate by adding the row-mean contribution into
    # the head bias is not possible per-token, so instead we keep column 0
    # exactly s and let the *other* columns absorb the zero-mean constraint.
    E = np.zeros((V, D), dtype=np.float64)
    E[:, 0] = s
    for d in range(1, D):
        E[:, d] = np.cos((np.arange(V) + 1.0) * (d + 1.0) * 0.61803398875)
    # orthogonalise columns 1..D-1 against column 0 and each other
    for d in range(1, D):
        v = E[:, d].copy()
        v -= E[:, 0] * (np.dot(v, E[:, 0]) / max(np.dot(E[:, 0], E[:, 0]), 1e-12))
        for e in range(1, d):
            v -= E[:, e] * (np.dot(v, E[:, e]) / max(np.dot(E[:, e], E[:, e]), 1e-12))
        E[:, d] = v
    # now scale columns 1..D-1 so each row has mean 0 and std 1
    # (column 0 is fixed to s; we allow the row stats to be approximate)
    for d in range(1, D):
        col = E[:, d]
        sd = col.std()
        if sd > 1e-8:
            E[:, d] = col / sd
    # final per-row centering using only columns 1..D-1 is impossible without
    # touching column 0, so we accept a tiny per-row mean and rely on the
    # final LayerNorm; to keep LayerNorm ≈ identity we instead scale column 0
    # down slightly and add its mean back via the head bias is not per-token.
    # Pragmatic choice: keep column 0 = s and set the other columns so the
    # row mean is small; LayerNorm will then be close to identity.

    w["embed.weight"] = E.astype(np.float32)

    # ---- Head reads column 0 ----
    H = np.zeros((2, D), dtype=np.float64)
    H[0, 0] = -1.0
    H[1, 0] = 1.0
    w["head.weight"] = H.astype(np.float32)
    w["head.bias"] = np.array([-intercept, intercept], dtype=np.float32)

    # ---- Zero out everything else (blocks, pos) ----
    for name in ["pos.weight",
                 "block0.qkv.weight", "block0.qkv.bias",
                 "block0.proj.weight", "block0.proj.bias",
                 "block0.ff1.weight", "block0.ff1.bias",
                 "block0.ff2.weight", "block0.ff2.bias",
                 "block1.qkv.weight", "block1.qkv.bias",
                 "block1.proj.weight", "block1.proj.bias",
                 "block1.ff1.weight", "block1.ff1.bias",
                 "block1.ff2.weight", "block1.ff2.bias"]:
        if name in w:
            w[name] = np.zeros_like(w[name], dtype=np.float32)

    for name in ["block0.norm1.weight", "block0.norm1.bias",
                 "block0.norm2.weight", "block0.norm2.bias",
                 "block1.norm1.weight", "block1.norm1.bias",
                 "block1.norm2.weight", "block1.norm2.bias",
                 "norm.weight", "norm.bias"]:
        if name in w:
            if name.endswith("weight"):
                w[name] = np.ones_like(w[name], dtype=np.float32)
            else:
                w[name] = np.zeros_like(w[name], dtype=np.float32)

    # ---- Diagnostics ----
    # quick train accuracy on the pool
    logits = BOW @ s + intercept
    pred = (logits > 0).astype(np.int64)
    train_acc = float(np.mean(pred == y))

    diag = {
        "method": "BOW ridge + LayerNorm-identity embedding",
        "best_lambda": float(best_lam),
        "cv_accuracy": float(best_cv),
        "train_accuracy": train_acc,
        "s_norm": float(np.linalg.norm(s)),
        "intercept": float(intercept),
        "embed_row_mean_maxabs": float(np.max(np.abs(E.mean(axis=1)))),
        "embed_row_std_maxdev": float(np.max(np.abs(E.std(axis=1) - 1.0))),
    }
    return {"weights": w, "diagnostics": diag}
