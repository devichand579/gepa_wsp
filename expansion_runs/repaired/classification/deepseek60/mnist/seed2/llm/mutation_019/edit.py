import numpy as np

def edit(w, f):
    # Extract fixed hidden features and labels from the feedback pool
    H = f["features"]          # (2000, 128) ReLU outputs of fc1
    y = f["y"]                 # (2000,) integer labels

    # Compute class means
    classes = np.arange(10)
    mu = np.zeros((10, H.shape[1]), dtype=np.float64)
    for k in classes:
        mu[k] = H[y == k].mean(axis=0)

    # Compute pooled within-class covariance
    S = np.zeros((H.shape[1], H.shape[1]), dtype=np.float64)
    for i in range(H.shape[0]):
        diff = H[i] - mu[y[i]]
        S += np.outer(diff, diff)
    S /= (H.shape[0] - 10)

    # Regularize covariance for numerical stability
    alpha = 0.01
    S_reg = S + alpha * np.eye(H.shape[1])

    # LDA weights: W = mu @ inv(S_reg)  -> (10, 128)
    W = mu @ np.linalg.inv(S_reg)

    # LDA biases: b_k = -0.5 * mu_k^T W_k^T  (priors are uniform)
    b = -0.5 * np.sum(mu * W, axis=1)

    # Update only fc2 parameters
    w["fc2.weight"] = W.astype(np.float32)
    w["fc2.bias"] = b.astype(np.float32)

    # Diagnostics: evaluate on the feedback pool
    logits = H @ W.T + b
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    log_probs = logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
    ce = -np.mean(log_probs[np.arange(len(y)), y])

    diagnostics = {
        "train_accuracy": float(acc),
        "train_cross_entropy": float(ce),
        "fc2_weight_norm": float(np.linalg.norm(W)),
        "fc2_bias_norm": float(np.linalg.norm(b)),
        "alpha": alpha
    }

    return {"weights": w, "diagnostics": diagnostics}
