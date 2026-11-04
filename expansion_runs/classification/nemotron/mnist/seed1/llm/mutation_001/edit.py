def edit(w, f):
    """
    Closed‑form ridge‑regression update of fc2 using the paired reflection batch.
    """
    # ------- batch data -------
    X = f["layer.fc2.input0"]                # (128, 128) features into fc2
    y_batch = f["batch.y"]                   # (128,) integer labels

    # one‑hot encode labels (10 classes)
    Y = np.eye(10, dtype=np.float32)[y_batch]  # (128, 10)

    # ridge regularisation (small λ stabilises the solve)
    lam = np.float32(1e-3)
    I128 = np.eye(128, dtype=np.float32)

    # solve (Xᵀ X + λ I) wᵀ = Xᵀ Y   →   weightᵀ = (Xᵀ X + λ I)⁻¹ Xᵀ Y
    A = X.T @ X + lam * I128                 # (128, 128)
    B = X.T @ Y                              # (128, 10)
    weight_T = np.linalg.solve(A, B)         # (128, 10)  <-- this is fc2.weightᵀ

    # assign new fc2 weight (shape [10, 128])
    w["fc2.weight"] = weight_T.astype(np.float32).T   # (10, 128)

    # ------- bias update -------
    # bias = mean( Y - X @ weightᵀ ) over the batch
    residuals = Y - X @ weight_T             # (128, 10)
    w["fc2.bias"] = residuals.mean(axis=0).astype(np.float32)   # (10,)

    # ------- diagnostics on the full 2000‑sample pool -------
    # flatten input images
    x_flat = f["x"].reshape(-1, 784)        # (2000, 784)

    # forward through fc1 (unchanged)
    h = np.maximum(0.0, x_flat @ w["fc1.weight"].T + w["fc1.bias"])   # (2000, 128)

    # forward through updated fc2
    logits = h @ w["fc2.weight"].T + w["fc2.bias"]   # (2000, 10)

    # numerically stable softmax
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    pred = probs.argmax(axis=1)               # (2000,)
    accuracy = (pred == f["y"]).mean()
    cross_entropy = -np.log(probs[np.arange(2000), f["y"]]).mean()

    diag = {
        "train_accuracy": float(accuracy),
        "train_cross_entropy": float(cross_entropy),
        "fc2_weight_norm": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_bias": w["fc2.bias"].tolist()
    }

    return {"weights": w, "diagnostics": diag}
