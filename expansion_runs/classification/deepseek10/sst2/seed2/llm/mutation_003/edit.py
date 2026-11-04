import numpy as np
import math

def edit(w, f):
    # Extract training data
    x = f["x"]  # (4000, 64) int64
    y = f["y"]  # (4000,) int64
    N = x.shape[0]
    V = 2000  # vocab size from embed.weight shape
    D = 32    # embedding dim

    # Build mask for non-pad tokens
    mask = x != 0  # (N, 64)
    lengths = mask.sum(axis=1)  # (N,)
    lengths = np.maximum(lengths, 1)  # avoid division by zero

    # Build document-term matrix (counts)
    X = np.zeros((N, V), dtype=np.float32)
    for i in range(N):
        tokens = x[i][mask[i]]
        if len(tokens) > 0:
            counts = np.bincount(tokens, minlength=V)
            X[i, :len(counts)] = counts[:V]

    # TF-IDF transformation
    # Term frequency: 1 + log(count) for count > 0
    tf = np.zeros_like(X)
    nz = X > 0
    tf[nz] = 1.0 + np.log(X[nz])
    # Document frequency
    df = (X > 0).sum(axis=0)  # (V,)
    idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
    X_tfidf = tf * idf  # (N, V)
    # L2 normalize each document
    norms = np.linalg.norm(X_tfidf, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X_tfidf = X_tfidf / norms

    # Center the data for PCA
    X_mean = X_tfidf.mean(axis=0, keepdims=True)
    X_centered = X_tfidf - X_mean

    # SVD to get top D principal components
    # X_centered is (N, V). We want right singular vectors (V, D)
    # Use economy SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Vt shape (min(N,V), V). Take top D rows.
    W = Vt[:D, :].T  # (V, D)

    # Standardize each token embedding to zero mean and unit variance
    W_mean = W.mean(axis=1, keepdims=True)
    W_std = W.std(axis=1, keepdims=True)
    W_std = np.maximum(W_std, 1e-6)
    W_norm = (W - W_mean) / W_std  # (V, D)

    # Set embeddings
    w["embed.weight"] = W_norm.astype(np.float32)
    w["pos.weight"] = np.zeros_like(w["pos.weight"], dtype=np.float32)

    # Set final norm to identity (weight=1, bias=0)
    w["norm.weight"] = np.ones_like(w["norm.weight"], dtype=np.float32)
    w["norm.bias"] = np.zeros_like(w["norm.bias"], dtype=np.float32)

    # Zero out all transformer blocks to be safe
    for key in list(w.keys()):
        if key.startswith("block"):
            w[key] = np.zeros_like(w[key], dtype=np.float32)

    # Compute pooled features for all training examples
    # pooled = (X @ W_norm) / lengths
    # X is counts (N, V). X @ W_norm gives sum of embeddings for each doc.
    pooled = (X @ W_norm) / lengths[:, None]  # (N, D)

    # Fit head via ridge regression
    # Augment with bias column
    P_aug = np.hstack([pooled, np.ones((N, 1), dtype=np.float32)])  # (N, D+1)
    # One-hot targets
    Y_onehot = np.zeros((N, 2), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0

    # Try a few lambda values via 5-fold CV to pick best
    best_lambda = 1.0
    best_acc = -1.0
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    # Simple 5-fold CV
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    for lam in lambdas:
        accs = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            P_train = P_aug[train_idx]
            Y_train = Y_onehot[train_idx]
            P_val = P_aug[val_idx]
            Y_val = y[val_idx]
            # Ridge closed form
            A = P_train.T @ P_train + lam * np.eye(D+1, dtype=np.float32)
            b = P_train.T @ Y_train
            W_head = np.linalg.solve(A, b)  # (D+1, 2)
            # Predict
            logits_val = P_val @ W_head  # (N_val, 2)
            preds = np.argmax(logits_val, axis=1)
            acc = (preds == Y_val).mean()
            accs.append(acc)
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lambda = lam

    # Final fit on all data with best lambda
    A = P_aug.T @ P_aug + best_lambda * np.eye(D+1, dtype=np.float32)
    b = P_aug.T @ Y_onehot
    W_head = np.linalg.solve(A, b)  # (D+1, 2)

    # Set head weights and bias
    w["head.weight"] = W_head[:D, :].T.astype(np.float32)  # (2, D)
    w["head.bias"] = W_head[D, :].astype(np.float32)       # (2,)

    # Diagnostics
    diagnostics = {
        "best_lambda": float(best_lambda),
        "cv_accuracy": float(best_acc),
        "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
        "head_bias": w["head.bias"].tolist(),
        "embed_norm_mean": float(np.mean(np.linalg.norm(w["embed.weight"], axis=1))),
        "train_accuracy": float((np.argmax(P_aug @ W_head, axis=1) == y).mean())
    }

    return {"weights": w, "diagnostics": diagnostics}
