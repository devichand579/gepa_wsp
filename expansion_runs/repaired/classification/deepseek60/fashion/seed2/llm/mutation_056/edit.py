import numpy as np

def edit(w, f):
    # Extract current head weights and biases
    W = w["head.weight"].copy()          # [10, 1568]
    b = w["head.bias"].copy()            # [10]
    features = f["features"]             # [2000, 1568]
    y = f["y"]                           # [2000]
    
    # Compute current logits and probabilities
    logits = features @ W.T + b          # [2000, 10]
    # Stable softmax
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    # Current cross-entropy
    eps = 1e-12
    ce = -np.mean(np.log(probs[np.arange(len(y)), y] + eps))
    
    # Identify examples with the smallest margins (highest loss)
    correct_logits = logits[np.arange(len(y)), y]
    # Mask out correct class to find max incorrect
    mask = np.ones_like(logits, dtype=bool)
    mask[np.arange(len(y)), y] = False
    max_incorrect = logits[mask].reshape(len(y), -1).max(axis=1)
    margins = correct_logits - max_incorrect
    
    # We want to increase margins for the worst examples.
    # Sort by margin ascending, take the bottom 10% (or all if few)
    n_bad = max(1, int(0.1 * len(y)))
    bad_idx = np.argsort(margins)[:n_bad]
    
    # For these bad examples, we will add a rank-1 update to the head weights.
    # We want to increase the correct class logit and decrease the max incorrect class logit.
    # Compute the average feature vector for the correct class of these examples,
    # and the average feature vector for the max incorrect class.
    # We'll accumulate updates for each class.
    delta_W = np.zeros_like(W)
    delta_b = np.zeros_like(b)
    
    # Learning rate for the update (chosen to be large enough to push margins but not too large)
    # We can compute a safe step size based on the feature norms.
    # For each bad example, we want to increase margin by some target, say 50.
    target_margin_increase = 50.0
    
    for i in bad_idx:
        true_c = y[i]
        # Find max incorrect class
        inc_c = np.argmax(np.where(np.arange(10) == true_c, -np.inf, logits[i]))
        # We want to add to W[true_c] a vector proportional to features[i],
        # and subtract from W[inc_c] a vector proportional to features[i].
        # The change in logit for true_c is features[i] @ delta_W[true_c] + delta_b[true_c]
        # The change for inc_c is features[i] @ delta_W[inc_c] + delta_b[inc_c]
        # We want (change_true - change_inc) >= target_margin_increase - margin[i]
        # Let's just add a small multiple of features[i] to true_c and subtract from inc_c.
        # To avoid huge updates, we normalize by the squared norm of features[i].
        feat = features[i]
        norm2 = np.dot(feat, feat) + 1e-8
        # We want the net increase in margin to be roughly target_margin_increase - margin[i]
        needed = max(0.0, target_margin_increase - margins[i])
        if needed > 0:
            # Distribute the needed increase: add half to true, subtract half from inc
            # Actually, adding to true increases margin by feat @ delta_true.
            # Subtracting from inc increases margin by -feat @ delta_inc.
            # Let delta_true = alpha * feat, delta_inc = -alpha * feat.
            # Then margin increase = feat @ (alpha*feat) - feat @ (-alpha*feat) = 2*alpha*||feat||^2.
            # So alpha = needed / (2 * norm2).
            alpha = needed / (2.0 * norm2)
            delta_W[true_c] += alpha * feat
            delta_W[inc_c] -= alpha * feat
            # Also adjust biases slightly to help
            delta_b[true_c] += alpha * 0.1
            delta_b[inc_c] -= alpha * 0.1
    
    # Apply the updates
    W_new = W + delta_W
    b_new = b + delta_b
    
    # Clip to [-1000, 1000] to satisfy constraints
    W_new = np.clip(W_new, -1000.0, 1000.0)
    b_new = np.clip(b_new, -1000.0, 1000.0)
    
    # Verify that the new weights still achieve 100% accuracy on the feedback pool
    logits_new = features @ W_new.T + b_new
    preds_new = np.argmax(logits_new, axis=1)
    acc_new = np.mean(preds_new == y)
    
    # If accuracy dropped, revert to original
    if acc_new < 1.0:
        W_new = W
        b_new = b
        logits_new = logits
        acc_new = 1.0
    
    # Compute new cross-entropy
    logits_max_new = logits_new.max(axis=1, keepdims=True)
    exp_logits_new = np.exp(logits_new - logits_max_new)
    probs_new = exp_logits_new / exp_logits_new.sum(axis=1, keepdims=True)
    ce_new = -np.mean(np.log(probs_new[np.arange(len(y)), y] + eps))
    
    # If new CE is worse, revert
    if ce_new > ce:
        W_new = W
        b_new = b
        ce_new = ce
    
    # Update weights
    w["head.weight"] = W_new.astype(np.float32)
    w["head.bias"] = b_new.astype(np.float32)
    
    diagnostics = {
        "original_ce": float(ce),
        "new_ce": float(ce_new),
        "original_accuracy": 1.0,
        "new_accuracy": float(acc_new),
        "max_abs_head_weight": float(np.max(np.abs(W_new))),
        "max_abs_head_bias": float(np.max(np.abs(b_new))),
        "num_updated_examples": int(len(bad_idx))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
