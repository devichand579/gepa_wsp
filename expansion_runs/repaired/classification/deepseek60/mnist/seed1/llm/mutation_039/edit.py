import numpy as np
import math

def edit(w, f):
    # Extract data
    x = f["x"].reshape(2000, -1).astype(np.float64)  # (2000, 784)
    y = f["y"].astype(np.int64)
    N = x.shape[0]
    C = 10
    D = x.shape[1]
    
    # Class means
    M = np.zeros((C, D))
    for c in range(C):
        M[c] = x[y == c].mean(axis=0)
    mu = x.mean(axis=0)
    
    # Candidate directions
    w_list = []
    
    # 1. Current fc1 rows
    W_cur = w["fc1.weight"].astype(np.float64)
    for i in range(W_cur.shape[0]):
        ww = W_cur[i].copy()
        nrm = np.linalg.norm(ww)
        if nrm > 1e-8:
            w_list.append(ww / nrm)
    
    # 2. Class means (centered)
    for c in range(C):
        ww = M[c] - mu
        nrm = np.linalg.norm(ww)
        if nrm > 1e-8:
            w_list.append(ww / nrm)
    
    # 3. Pairwise class mean differences
    for i in range(C):
        for j in range(i+1, C):
            ww = M[i] - M[j]
            nrm = np.linalg.norm(ww)
            if nrm > 1e-8:
                w_list.append(ww / nrm)
    
    # 4. LDA directions
    Sw = np.zeros((D, D))
    for c in range(C):
        Xc = x[y == c] - M[c]
        Sw += Xc.T @ Xc
    Sw_reg = Sw + 1e-3 * np.eye(D)
    Sw_inv = np.linalg.inv(Sw_reg)
    Mc = M - mu
    A = Mc @ Sw_inv @ Mc.T
    eigvals, eigvecs = np.linalg.eigh(A)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    for k in range(min(9, C-1)):
        v = eigvecs[:, k]
        ww = Mc.T @ v
        nrm = np.linalg.norm(ww)
        if nrm > 1e-8:
            w_list.append(ww / nrm)
    
    # 5. PCA directions
    X_centered = x - mu
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    n_pca = min(128, Vt.shape[0])
    for i in range(n_pca):
        ww = Vt[i].copy()
        nrm = np.linalg.norm(ww)
        if nrm > 1e-8:
            w_list.append(ww / nrm)
    
    # 6. Ridge regression directions
    Y_onehot = np.zeros((N, C))
    Y_onehot[np.arange(N), y] = 1.0
    XtX = x.T @ x
    lambda_ridge = 1e-2
    W_ridge = np.linalg.solve(XtX + lambda_ridge * np.eye(D), x.T @ Y_onehot)
    for c in range(C):
        ww = W_ridge[:, c].copy()
        nrm = np.linalg.norm(ww)
        if nrm > 1e-8:
            w_list.append(ww / nrm)
    
    # 7. Random projections
    np.random.seed(42)
    for _ in range(200):
        ww = np.random.randn(D)
        nrm = np.linalg.norm(ww)
        if nrm > 1e-8:
            w_list.append(ww / nrm)
    
    # Generate (w, b) candidates and compute ANOVA F-value
    candidates = []  # (F, w, b)
    percentiles = [0, 10, 25, 50, 75, 90]
    for ww in w_list:
        p = x @ ww
        biases = [0.0, -p.mean()]
        for q in percentiles:
            biases.append(-np.percentile(p, q))
        # unique biases
        biases = list(set([round(b, 6) for b in biases]))
        for b in biases:
            h = np.maximum(0.0, p + b)
            # ANOVA F-value
            overall_mean = h.mean()
            between = 0.0
            within = 0.0
            for c in range(C):
                hc = h[y == c]
                mc = hc.mean()
                between += len(hc) * (mc - overall_mean) ** 2
                within += ((hc - mc) ** 2).sum()
            F = (between / (C - 1)) / (within / (N - C) + 1e-12)
            candidates.append((F, ww, b))
    
    # Select top 128 by F-value
    candidates.sort(key=lambda t: t[0], reverse=True)
    selected = candidates[:128]
    
    W1 = np.zeros((128, D))
    b1 = np.zeros(128)
    for i, (F, ww, b) in enumerate(selected):
        W1[i] = ww
        b1[i] = b
    
    # Compute hidden features
    H = np.maximum(0.0, x @ W1.T + b1)  # (2000, 128)
    
    # Standardize features
    mean_h = H.mean(axis=0)
    std_h = H.std(axis=0) + 1e-6
    H_std = (H - mean_h) / std_h
    
    # Fit fc2 via ridge regression
    A_aug = np.hstack([H_std, np.ones((N, 1))])
    lambda_fc2 = 1e-2
    B = np.linalg.solve(A_aug.T @ A_aug + lambda_fc2 * np.eye(129), A_aug.T @ Y_onehot)
    
    W_std = B[:128].T  # (10, 128)
    b_std = B[128]     # (10,)
    
    # Convert back to original feature space
    W2 = W_std / std_h[None, :]
    b2 = b_std - (W_std / std_h[None, :]) @ mean_h
    
    # Update weights
    w["fc1.weight"] = W1.astype(np.float32)
    w["fc1.bias"] = b1.astype(np.float32)
    w["fc2.weight"] = W2.astype(np.float32)
    w["fc2.bias"] = b2.astype(np.float32)
    
    # Clip to safe range
    for k in w:
        w[k] = np.clip(w[k], -1000.0, 1000.0).astype(np.float32)
    
    # Diagnostics
    logits = H @ W2.T + b2
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    acc = (preds == y).mean()
    ce = -np.log(probs[np.arange(N), y] + 1e-12).mean()
    
    diag = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "num_candidates": len(candidates),
        "selected_F_mean": float(np.mean([c[0] for c in selected])),
        "fc1_weight_mean": float(w["fc1.weight"].mean()),
        "fc1_weight_std": float(w["fc1.weight"].std()),
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std": float(w["fc2.weight"].std()),
        "fc2_bias_mean": float(w["fc2.bias"].mean()),
        "fc2_bias_std": float(w["fc2.bias"].std()),
    }
    
    return {"weights": w, "diagnostics": diag}
