import numpy as np

def edit(w, f):
    X = f["features"].astype(np.float64)
    y = f["y"].astype(np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Fisher LDA ---
    X0 = X[y == 0]
    X1 = X[y == 1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    n0, n1 = len(X0), len(X1)
    S0 = np.cov(X0, rowvar=False)
    S1 = np.cov(X1, rowvar=False)
    S = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
    lam = 1e-3
    S_reg = S + lam * np.eye(X.shape[1])
    w_lda = np.linalg.solve(S_reg, mu1 - mu0)
    b_lda = -np.dot(w_lda, (mu0 + mu1) / 2.0)

    # --- Least squares ---
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    lam_ls = 1e-3
    A = X_aug.T @ X_aug + lam_ls * np.eye(X_aug.shape[1])
    b_vec = X_aug.T @ y.astype(np.float64)
    w_ls = np.linalg.solve(A, b_vec)
    w_ls_w = w_ls[:32]
    w_ls_b = w_ls[32]

    def acc(w_vec, b):
        logits = X @ w_vec + b
        pred = (logits > 0).astype(np.int64)
        return float(np.mean(pred == y))

    acc_lda = acc(w_lda, b_lda)
    acc_ls = acc(w_ls_w, w_ls_b)

    if acc_lda >= acc_ls:
        w_final, b_final, chosen = w_lda, b_lda, "lda"
    else:
        w_final, b_final, chosen = w_ls_w, w_ls_b, "ls"

    w_final = np.clip(w_final, -1000.0, 1000.0)
    b_final = np.clip(b_final, -1000.0, 1000.0)

    # Overwrite head: logit1 - logit0 = 2*(w_final·x + b_final)
    w["head.weight"][1] = w_final.astype(np.float32)
    w["head.weight"][0] = (-w_final).astype(np.float32)
    w["head.bias"][1] = np.float32(b_final)
    w["head.bias"][0] = np.float32(-b_final)

    diag = {
        "chosen": chosen,
        "acc_lda": acc_lda,
        "acc_ls": acc_ls,
        "final_train_acc": acc(w_final, b_final),
        "w_norm": float(np.linalg.norm(w_final)),
        "b": float(b_final),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
    }
    return {"weights": w, "diagnostics": diag}
