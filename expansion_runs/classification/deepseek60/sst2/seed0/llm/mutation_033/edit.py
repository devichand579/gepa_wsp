import numpy as np

def edit(w, f):
    # Final hidden features and labels from the full feedback pool
    X = f["features"].astype(np.float64)   # [4000, 32]
    y = f["y"].astype(np.int64)            # [4000]
    N, D = X.shape

    # Class means
    mu0 = X[y == 0].mean(axis=0)
    mu1 = X[y == 1].mean(axis=0)

    # Within-class scatter
    X0 = X[y == 0] - mu0
    X1 = X[y == 1] - mu1
    Sw = (X0.T @ X0 + X1.T @ X1) / (N - 2)

    # Small ridge for numerical stability
    Sw += 1e-3 * np.eye(D)

    # LDA direction and bias
    w_vec = np.linalg.solve(Sw, mu1 - mu0)
    prior1 = (y == 1).sum() / N
    prior0 = (y == 0).sum() / N
    b = -0.5 * (mu1 + mu0) @ w_vec + np.log(prior1 / prior0)

    # Scale so the logit difference has unit variance on the feedback pool
    d = X @ w_vec + b
    std_d = np.std(d)
    if std_d > 1e-6:
        w_vec /= std_d
        b /= std_d

    # Write the new head. Keep the symmetric two-row form.
    w["head.weight"][0] = -w_vec
    w["head.weight"][1] = w_vec
    w["head.bias"][0] = -b
    w["head.bias"][1] = b

    # Diagnostics
    logits = X @ w_vec + b
    preds = (logits > 0).astype(np.int64)
    acc = float((preds == y).mean())

    diagnostics = {
        "lda_train_acc": acc,
        "w_norm": float(np.linalg.norm(w_vec)),
        "b": float(b),
        "std_d": float(std_d),
    }

    return {"weights": w, "diagnostics": diagnostics}
