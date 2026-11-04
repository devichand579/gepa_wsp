import numpy as np

def edit(w, f):
    # Current pooled features (N, 32) and labels (N,)
    X = np.asarray(f["features"], dtype=np.float64)
    y = np.asarray(f["y"], dtype=np.float64)
    N, D = X.shape

    # Center features to avoid huge intercept coupling
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    # Ridge on ±1 targets: solve (Xc^T Xc + lam I) w = Xc^T y
    lam = 1e-2
    A = Xc.T @ Xc + lam * np.eye(D)
    b = Xc.T @ y
    w_dir = np.linalg.solve(A, b)

    # Bias from class prior (balanced here, but compute anyway)
    p1 = y.mean()
    bias = np.log(p1 / (1.0 - p1 + 1e-12))

    # Map to 2-class head: logit0 = -(w·x + b), logit1 = +(w·x + b)
    # head.weight is [2, 32], head.bias is [2]
    # We want logit1 - logit0 = 2*(w·x + b) => head[1]-head[0] = 2*w, bias diff = 2*b
    # Keep head symmetric: head[1] = w, head[0] = -w, bias[1] = b, bias[0] = -b
    # But the original head had head[1] = -head[0], so this preserves that symmetry.
    # However, to avoid changing the scale of logits too much, we can just set:
    # head[1] = w, head[0] = -w, bias[1] = b, bias[0] = -b
    # This gives logit1 - logit0 = 2*(w·x + b), which is fine.
    # Actually, the original head had head[1] = -head[0] and bias[1] = -bias[0].
    # We'll keep that symmetry.

    # Normalize w_dir to a reasonable scale (original head had norm ~0.4445*sqrt(32) ≈ 2.5)
    # But ridge already gives a scale. Let's just use it.
    # To be safe, clip to avoid extreme values.
    w_dir = np.clip(w_dir, -10.0, 10.0)
    bias = np.clip(bias, -10.0, 10.0)

    # Update head weights and bias
    w["head.weight"] = np.vstack([-w_dir, w_dir]).astype(np.float32)
    w["head.bias"] = np.array([-bias, bias], dtype=np.float32)

    # Diagnostics
    # Compute training accuracy on the full pool using the new head
    logits = X @ w_dir + bias
    pred = (logits > 0).astype(np.int64)
    acc = float((pred == y).mean())

    diagnostics = {
        "method": "closed_form_ridge_head_on_current_features",
        "lam": lam,
        "train_acc_head": acc,
        "w_dir_norm": float(np.linalg.norm(w_dir)),
        "w_bias": float(bias),
        "feature_mean_norm": float(np.linalg.norm(mu)),
    }

    return {"weights": w, "diagnostics": diagnostics}
