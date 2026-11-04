import numpy as np
import math

def edit(w, f):
    # ---------------- data ----------------
    X = np.asarray(f["x"], dtype=np.float64).reshape(-1, 784)
    y = np.asarray(f["y"]).astype(np.int64).ravel()
    N = X.shape[0]
    K = 10
    H = 128

    mu = X.mean(axis=0)
    Xc = X - mu

    # ---------------- PCA-128 subspace (closed form eigendecomposition) ----------------
    Cov = (Xc.T @ Xc) / (N - 1)
    Cov = 0.5 * (Cov + Cov.T)
    evals, evecs = np.linalg.eigh(Cov)          # ascending
    idx = np.argsort(evals)[::-1][:H]
    A = evecs[:, idx].T.copy()                   # (H, 784), orthonormal rows
    lam = np.maximum(evals[idx], 1e-12)

    Z = Xc @ A.T                                  # (N, H) PCA scores
    Zm = Z.mean(axis=0)
    Zc = Z - Zm

    # ---------------- shrinkage LDA in PCA space ----------------
    cls = np.zeros((K, H))
    cnt = np.zeros(K)
    Sw = np.zeros((H, H))
    for c in range(K):
        m = (y == c)
        cnt[c] = m.sum()
        if cnt[c] == 0:
            cnt[c] = 1
            continue
        Zc_c = Zc[m]
        mu_c = Zc_c.mean(axis=0)
        cls[c] = mu_c
        D = Zc_c - mu_c
        Sw += D.T @ D
    Sw /= max(N - K, 1)
    Sw = 0.5 * (Sw + Sw.T)

    tr = np.trace(Sw) / H
    # shrink small-variance directions more (ridge) + mild diagonal shrinkage
    Sw_reg = 0.95 * Sw + 0.05 * np.diag(np.diag(Sw)) + (0.05 * tr + 1e-9) * np.eye(H)
    P = np.linalg.inv(Sw_reg)

    prior = cnt / cnt.sum()
    grand = (cls * cnt[:, None]).sum(axis=0) / cnt.sum()
    Wc = (cls - grand[None, :]) @ P.T          # (K, H) discriminant directions
    b_eff = -0.5 * np.sum((cls @ P) * cls, axis=1) + np.log(prior + 1e-12)

    # ---------------- build fc1 / fc2 so that ReLU stays linear ----------------
    # pre-activation = Z + c ; choose c so every training point is strictly positive
    minZ = Z.min(axis=0)
    margin = 3.0
    c_bias = -minZ + margin
    # extra headroom for held-out points: scale with the spread of the score
    c_bias = c_bias + 0.25 * np.abs(Z).max(axis=0)

    W1 = A.copy()                                # (128, 784)
    b1 = c_bias.astype(np.float32)               # (128,)
    W2 = Wc.astype(np.float32)                   # (10, 128)
    b2 = (b_eff - Wc @ c_bias).astype(np.float32)  # (10,)

    # ---------------- verify reconstruction & diagnostics ----------------
    Hh = X @ W1.T + b1[None, :]
    act = (Hh > 0).mean()
    logits = Hh @ W2.T + b2[None, :]
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    mx = logits.max(axis=1)
    lse = mx + np.log(np.exp(logits - mx[:, None]).sum(axis=1))
    ce = float((-lse[np.arange(N), y]).mean())

    # effective linear map in pixel space (for rank / conditioning inspection)
    Meff = W2 @ W1
    sv = np.linalg.svd(Meff, compute_uv=False)
    # margin of ReLU linear regime on the batch
    Hb = np.asarray(f["batch.x"], dtype=np.float64).reshape(-1, 784) @ W1.T + b1[None, :]
    batch_margin = float(Hb.min())

    # class-wise training accuracy
    cls_acc = {}
    for c in range(K):
        m = (y == c)
        cls_acc[str(c)] = float((pred[m] == c).mean()) if m.any() else 0.0

    # ---------------- write back ----------------
    out = dict(w)
    out["fc1.weight"] = np.ascontiguousarray(W1, dtype=np.float32)
    out["fc1.bias"] = np.ascontiguousarray(b1, dtype=np.float32)
    out["fc2.weight"] = np.ascontiguousarray(W2, dtype=np.float32)
    out["fc2.bias"] = np.ascontiguousarray(b2, dtype=np.float32)

    for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
        arr = out[k]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out[k] = np.clip(arr, -1000.0, 1000.0).astype(np.float32)

    diag = {
        "method": "PCA128 subspace + shrinkage LDA readout, ReLU held in linear regime",
        "train_acc_reconstructed": acc,
        "train_ce_reconstructed": ce,
        "train_acc_per_class": cls_acc,
        "relu_positive_fraction_train": float(act),
        "min_preactivation_train": float(Hh.min()),
        "min_preactivation_batch": batch_margin,
        "fc1_bias_min": float(b1.min()),
        "fc1_bias_max": float(b1.max()),
        "fc2_bias": [float(v) for v in out["fc2.bias"]],
        "eff_rank_Meff": float((sv > 1e-8 * sv.max()).sum()),
        "sv_Meff_top5": [float(v) for v in sv[:5]],
        "sv_Meff_min": float(sv[-1]),
        "pca_evals_top5": [float(v) for v in lam[:5]],
        "pca_evals_min": float(lam[-1]),
        "lda_sw_trace_per_dim": float(tr),
        "fc1_w_absmax": float(np.abs(out["fc1.weight"]).max()),
        "fc2_w_absmax": float(np.abs(out["fc2.weight"]).max()),
    }
    return {"weights": out, "diagnostics": diag}
