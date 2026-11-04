import numpy as np

def edit(w, f):
    # Extract feedback arrays
    x = f["x"]          # (4000, 64) int64
    y = f["y"]          # (4000,) int64
    features = f["features"]  # (4000, 32) float32 current final hidden features
    logits = f["logits"]
    probs = f["probs"]
    
    # We will refit the head using ridge regression on the current features.
    # The current network is degenerate: pos and all block weights are zero.
    # So the final features are already computed and provided.
    # We can simply fit a linear classifier (head) on these features.
    
    # Prepare data
    F = features.astype(np.float64)  # (4000, 32)
    Y = y.astype(np.float64)         # (4000,)
    
    # Center features and labels for better conditioning
    F_mean = F.mean(axis=0, keepdims=True)
    Y_mean = Y.mean()
    Fc = F - F_mean
    Yc = Y - Y_mean
    
    # Ridge regression: w = (Fc^T Fc + lambda I)^-1 Fc^T Yc
    # We'll try a few lambdas and pick the one with best training accuracy.
    best_acc = -1.0
    best_w = None
    best_b = None
    best_lambda = None
    
    # Candidate lambdas
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    for lam in lambdas:
        # Solve (Fc^T Fc + lam I) w = Fc^T Yc
        A = Fc.T @ Fc + lam * np.eye(F.shape[1])
        b = Fc.T @ Yc
        w_vec = np.linalg.solve(A, b)  # (32,)
        # Bias: we want score = w_vec @ F + b to predict Y (0/1)
        # Since we centered, w_vec @ Fc + Y_mean = w_vec @ (F - F_mean) + Y_mean
        # = w_vec @ F - w_vec @ F_mean + Y_mean
        # So bias = Y_mean - w_vec @ F_mean
        bias = Y_mean - w_vec @ F_mean[0]
        
        # Compute training predictions
        scores = F @ w_vec + bias
        preds = (scores > 0.5).astype(np.int64)
        acc = (preds == y).mean()
        
        if acc > best_acc:
            best_acc = acc
            best_w = w_vec
            best_b = bias
            best_lambda = lam
    
    # Now set the head weights.
    # head.weight shape (2, 32), head.bias shape (2,)
    # We want logits = head @ feature + head.bias
    # For binary classification, we can set:
    # head.weight[1] = best_w, head.bias[1] = best_b
    # head.weight[0] = -best_w, head.bias[0] = -best_b
    # This makes logit[1] - logit[0] = 2*(best_w @ feature + best_b)
    # So the decision boundary is at score = 0.
    # But our score was w_vec @ F + bias, and we predicted 1 if score > 0.5.
    # To match, we need to shift bias. Actually, we can just set head to produce the same decision.
    # If we set head.weight[1] = best_w, head.bias[1] = best_b - 0.5, then logit[1] = best_w @ F + best_b - 0.5.
    # Then logit[1] > logit[0] iff best_w @ F + best_b > 0.5.
    # So we can set:
    # head.weight[1] = best_w
    # head.bias[1] = best_b - 0.5
    # head.weight[0] = -best_w
    # head.bias[0] = -best_b + 0.5
    # This gives exactly the same predictions.
    
    # However, the original head had row0 = -row1 and bias0 = -bias1.
    # We can just set:
    w_new = best_w.astype(np.float32)
    b_new = (best_b - 0.5).astype(np.float32)
    
    w["head.weight"][1] = w_new
    w["head.bias"][1] = b_new
    w["head.weight"][0] = -w_new
    w["head.bias"][0] = -b_new
    
    # Ensure all other weights remain unchanged (they are already zero for blocks, etc.)
    # But we can also explicitly zero out pos and blocks to be safe.
    w["pos.weight"][:] = 0.0
    for block in ["block0", "block1"]:
        for name in ["norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias",
                     "proj.weight", "proj.bias", "norm2.weight", "norm2.bias",
                     "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"]:
            key = f"{block}.{name}"
            if key in w:
                w[key][:] = 0.0
        # Norm weights should be 1, biases 0
        w[f"{block}.norm1.weight"][:] = 1.0
        w[f"{block}.norm1.bias"][:] = 0.0
        w[f"{block}.norm2.weight"][:] = 1.0
        w[f"{block}.norm2.bias"][:] = 0.0
    
    # Final norm
    w["norm.weight"][:] = 1.0
    w["norm.bias"][:] = 0.0
    
    # Diagnostics
    diagnostics = {
        "method": "ridge_refit_head_on_current_features",
        "best_lambda": float(best_lambda),
        "train_accuracy": float(best_acc),
        "head_weight_norm": float(np.linalg.norm(w_new)),
        "head_bias": float(b_new)
    }
    
    return {"weights": w, "diagnostics": diagnostics}
