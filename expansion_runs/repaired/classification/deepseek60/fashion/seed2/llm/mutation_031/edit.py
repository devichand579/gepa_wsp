import numpy as np

def edit(w, f):
    # Extract current head parameters
    W = w["head.weight"].astype(np.float64).copy()  # [10, 1568]
    b = w["head.bias"].astype(np.float64).copy()    # [10]
    
    X = f["features"].astype(np.float64)  # [2000, 1568]
    y = f["y"].astype(np.int64)           # [2000]
    
    # Initial logits from feedback
    logits = f["logits"].astype(np.float64)
    preds = np.argmax(logits, axis=1)
    mis = np.where(preds != y)[0]
    
    fixed_count = 0
    for i in mis:
        x = X[i]  # [1568]
        t = y[i]
        
        # Recompute current logits for this sample
        cur_logits = W @ x + b
        p = np.argmax(cur_logits)
        if p == t:
            continue
        
        m = cur_logits[t] - cur_logits[p]  # negative
        norm2 = np.dot(x, x)
        if norm2 == 0:
            continue
        
        alpha_req = -m / (2.0 * norm2)
        if alpha_req <= 0:
            continue
        
        # Compute alpha_max to not break any currently correct sample
        L = X @ W.T + b  # [2000, 10]
        s = X @ x        # [2000]
        
        alpha_max = np.inf
        for j in range(len(y)):
            if j == i:
                continue
            yj = y[j]
            # Only constrain samples that are currently correct
            if np.argmax(L[j]) != yj:
                continue
            for k in range(10):
                if k == yj:
                    continue
                margin = L[j, yj] - L[j, k]
                dy = 0
                if yj == t: dy = 1
                elif yj == p: dy = -1
                dk = 0
                if k == t: dk = 1
                elif k == p: dk = -1
                c = dy - dk
                if c == 0:
                    continue
                val = s[j] * c
                if val < 0:
                    bound = margin / (-val)
                    if bound < alpha_max:
                        alpha_max = bound
        
        # Weight bounds
        def max_alpha_for_row(row, sign, x):
            v = sign * x
            bounds = []
            pos = v > 0
            if np.any(pos):
                bounds.append(np.min((1000.0 - row[pos]) / v[pos]))
            neg = v < 0
            if np.any(neg):
                bounds.append(np.min((-1000.0 - row[neg]) / v[neg]))
            if bounds:
                return np.min(bounds)
            return np.inf
        
        alpha_max_w = min(max_alpha_for_row(W[t], 1.0, x),
                          max_alpha_for_row(W[p], -1.0, x))
        alpha_max = min(alpha_max, alpha_max_w)
        
        if alpha_max <= 0:
            continue
        
        alpha = min(alpha_req * 1.1, alpha_max * 0.9)
        if alpha <= 0:
            continue
        
        # Apply rank-1 update
        W[t] += alpha * x
        W[p] -= alpha * x
        
        # Clip to [-1000, 1000]
        W = np.clip(W, -1000.0, 1000.0)
        fixed_count += 1
    
    # Write back
    w["head.weight"] = W.astype(np.float32)
    w["head.bias"] = b.astype(np.float32)
    
    # Diagnostics
    final_logits = X @ W.T + b
    final_preds = np.argmax(final_logits, axis=1)
    final_acc = np.mean(final_preds == y)
    final_mis = np.where(final_preds != y)[0].tolist()
    
    diagnostics = {
        "initial_misclassified": mis.tolist(),
        "fixed_count": fixed_count,
        "final_accuracy": float(final_acc),
        "final_misclassified": final_mis,
        "max_abs_head_weight": float(np.max(np.abs(W))),
        "max_abs_head_bias": float(np.max(np.abs(b)))
    }
    
    return {"weights": w, "diagnostics": diagnostics}
