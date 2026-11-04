import numpy as np

def edit(w, f):
    # Extract head weights and bias
    W = w["head.weight"].copy()  # (10, 1568)
    b = w["head.bias"].copy()    # (10,)
    
    # Get features and labels
    X = f["features"]  # (2000, 1568)
    y = f["y"]         # (2000,)
    
    # Compute current logits
    logits = X @ W.T + b  # (2000, 10)
    pred = np.argmax(logits, axis=1)
    wrong = np.where(pred != y)[0]
    
    diagnostics = {
        "n_wrong_before": int(len(wrong)),
        "accuracy_before": float(np.mean(pred == y)),
    }
    
    if len(wrong) == 0:
        return {"weights": w, "diagnostics": diagnostics}
    
    # Process each wrong sample (usually 1)
    for i in wrong:
        true_class = int(y[i])
        pred_class = int(pred[i])
        x_i = X[i]  # (1568,)
        
        # Current margin for this sample
        margin = logits[i, true_class] - logits[i, pred_class]
        norm2 = np.dot(x_i, x_i)
        if norm2 == 0:
            # Fallback to bias adjustment
            d = (-margin) / 2.0 + 1e-3
            b[true_class] += d
            b[pred_class] -= d
            continue
        
        # Minimum c to fix the error: margin + 2*c*norm2 > 0
        c_min = (-margin) / (2.0 * norm2) + 1e-6
        
        # Compute exact bounds for c from all other samples
        # For each sample j, for each k != y_j, we need:
        # logit_j[y_j] + c*s_j*(I(y_j==true)-I(y_j==pred)) > logit_j[k] + c*s_j*(I(k==true)-I(k==pred))
        # => diff > c * coeff
        # where diff = logit_j[y_j] - logit_j[k]
        # coeff = s_j * (I(k==true)-I(k==pred)-I(y_j==true)+I(y_j==pred))
        # s_j = x_i · x_j
        
        # Compute s_j for all j
        s = X @ x_i  # (2000,)
        
        c_low = c_min
        c_high = np.inf
        
        # We can vectorize the constraints
        # For each j, we need to check all k != y_j.
        # Let's create a mask for k != y_j.
        # We can loop over j (2000) - it's fast enough.
        for j in range(len(y)):
            if j == i:
                continue
            yj = y[j]
            aj = pred[j]  # current argmax
            sj = s[j]
            if sj == 0:
                continue
            # For all k != yj, compute diff and coeff
            # diff = logits[j, yj] - logits[j, k]
            # coeff = sj * (I(k==true) - I(k==pred) - I(yj==true) + I(yj==pred))
            # We only need to consider k that could be the max. But we can just check all k != yj.
            # To be safe, check all k != yj.
            for k in range(10):
                if k == yj:
                    continue
                diff = logits[j, yj] - logits[j, k]
                # coeff
                I_k_true = 1.0 if k == true_class else 0.0
                I_k_pred = 1.0 if k == pred_class else 0.0
                I_yj_true = 1.0 if yj == true_class else 0.0
                I_yj_pred = 1.0 if yj == pred_class else 0.0
                coeff = sj * (I_k_true - I_k_pred - I_yj_true + I_yj_pred)
                if coeff > 0:
                    # diff > c * coeff => c < diff / coeff
                    bound = diff / coeff
                    if bound < c_high:
                        c_high = bound
                elif coeff < 0:
                    # diff > c * coeff => c > diff / coeff (since coeff negative, dividing flips)
                    bound = diff / coeff
                    if bound > c_low:
                        c_low = bound
        
        # Also consider weight bounds: |W + c * delta| <= 1000
        # delta for true_class: +x_i, for pred_class: -x_i
        # For true_class: W[true_class] + c*x_i <= 1000 => c <= (1000 - W[true_class])/x_i where x_i > 0
        # and W[true_class] + c*x_i >= -1000 => c >= (-1000 - W[true_class])/x_i where x_i < 0
        # We can compute the max allowed c from weight bounds.
        # Since c is small, we can just check after. But let's compute bounds.
        # For each element in x_i, if x_i > 0: c <= (1000 - W[true_class])/x_i, c >= (-1000 - W[true_class])/x_i
        # if x_i < 0: c <= (-1000 - W[true_class])/x_i, c >= (1000 - W[true_class])/x_i
        # Actually, we only need to ensure we don't exceed 1000. We can just clip at the end.
        # But clipping might break the fix. So let's compute the max c that keeps weights within bounds.
        # For true_class:
        pos_mask = x_i > 0
        if np.any(pos_mask):
            c_high = min(c_high, np.min((1000.0 - W[true_class][pos_mask]) / x_i[pos_mask]))
            c_low = max(c_low, np.max((-1000.0 - W[true_class][pos_mask]) / x_i[pos_mask]))
        neg_mask = x_i < 0
        if np.any(neg_mask):
            c_high = min(c_high, np.min((-1000.0 - W[true_class][neg_mask]) / x_i[neg_mask]))
            c_low = max(c_low, np.max((1000.0 - W[true_class][neg_mask]) / x_i[neg_mask]))
        # For pred_class: W[pred_class] - c*x_i <= 1000 => -c*x_i <= 1000 - W[pred_class] => c*x_i >= W[pred_class] - 1000
        # if x_i > 0: c >= (W[pred_class] - 1000)/x_i
        # if x_i < 0: c <= (W[pred_class] - 1000)/x_i
        if np.any(pos_mask):
            c_low = max(c_low, np.max((W[pred_class][pos_mask] - 1000.0) / x_i[pos_mask]))
        if np.any(neg_mask):
            c_high = min(c_high, np.min((W[pred_class][neg_mask] - 1000.0) / x_i[neg_mask]))
        # Also lower bound: W[pred_class] - c*x_i >= -1000 => -c*x_i >= -1000 - W[pred_class] => c*x_i <= 1000 + W[pred_class]
        # if x_i > 0: c <= (1000 + W[pred_class])/x_i
        # if x_i < 0: c >= (1000 + W[pred_class])/x_i
        if np.any(pos_mask):
            c_high = min(c_high, np.min((1000.0 + W[pred_class][pos_mask]) / x_i[pos_mask]))
        if np.any(neg_mask):
            c_low = max(c_low, np.max((1000.0 + W[pred_class][neg_mask]) / x_i[neg_mask]))
        
        # Choose c
        if c_low <= c_high:
            # Pick a value slightly above c_low, but not too close to c_high
            c = c_low + min(1e-3, (c_high - c_low) * 0.5)
            # Ensure c is not too large
            c = min(c, c_high - 1e-6)
            if c < c_min:
                c = c_min
        else:
            # Fallback to bias adjustment
            d = (-margin) / 2.0 + 1e-3
            b[true_class] += d
            b[pred_class] -= d
            # Update logits for next iteration
            logits = X @ W.T + b
            pred = np.argmax(logits, axis=1)
            continue
        
        # Apply update
        W[true_class] += c * x_i
        W[pred_class] -= c * x_i
        
        # Recompute logits and check
        new_logits = X @ W.T + b
        new_pred = np.argmax(new_logits, axis=1)
        new_wrong = np.where(new_pred != y)[0]
        
        # If we introduced new errors, revert and try bias adjustment
        if len(new_wrong) > len(wrong):
            W[true_class] -= c * x_i
            W[pred_class] += c * x_i
            # Bias adjustment
            d = (-margin) / 2.0 + 1e-3
            b[true_class] += d
            b[pred_class] -= d
            logits = X @ W.T + b
            pred = np.argmax(logits, axis=1)
        else:
            logits = new_logits
            pred = new_pred
    
    # Clip weights to [-1000, 1000] and ensure float32
    W = np.clip(W, -1000.0, 1000.0).astype(np.float32)
    b = np.clip(b, -1000.0, 1000.0).astype(np.float32)
    
    w["head.weight"] = W
    w["head.bias"] = b
    
    # Final diagnostics
    final_logits = X @ W.T + b
    final_pred = np.argmax(final_logits, axis=1)
    final_acc = float(np.mean(final_pred == y))
    diagnostics.update({
        "n_wrong_after": int(np.sum(final_pred != y)),
        "accuracy_after": final_acc,
        "max_abs_head_weight": float(np.max(np.abs(W))),
        "max_abs_head_bias": float(np.max(np.abs(b))),
    })
    
    return {"weights": w, "diagnostics": diagnostics}
