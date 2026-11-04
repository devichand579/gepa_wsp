import numpy as np

def edit(w, f):
    # Extract head weights and bias
    W = w["head.weight"]  # (10, 1568)
    b = w["head.bias"]    # (10,)
    
    features = f["features"]  # (2000, 1568)
    y = f["y"]                # (2000,)
    logits = f["logits"].copy()  # (2000, 10)
    
    N, D = features.shape
    num_classes = W.shape[0]
    
    # Iterate to fix misclassified samples
    for iteration in range(10):
        preds = np.argmax(logits, axis=1)
        mis = np.where(preds != y)[0]
        if len(mis) == 0:
            break
        
        for i in mis:
            x = features[i]  # (D,)
            true = y[i]
            pred = preds[i]
            
            # Current margin
            margin = logits[i, true] - logits[i, pred]
            norm2 = np.dot(x, x)
            if norm2 == 0:
                continue
            
            # Minimum alpha to fix this sample
            alpha_min = (-margin + 1e-4) / (2.0 * norm2)
            if alpha_min <= 0:
                alpha_min = 1e-4
            
            # Compute alpha_max from constraints
            alpha_max = np.inf
            
            # 1. Margin constraints from correctly classified samples
            s_all = features @ x  # (N,)
            for j in range(N):
                if preds[j] != y[j]:
                    continue  # only constrain correct samples
                yj = y[j]
                # c_k = I(yj==true) - I(k==true) - I(yj==pred) + I(k==pred)
                c = np.zeros(num_classes)
                if yj == true:
                    c += 1
                if yj == pred:
                    c -= 1
                c[true] -= 1
                c[pred] += 1
                sc = s_all[j] * c
                for k in range(num_classes):
                    if k == yj:
                        continue
                    if sc[k] < 0:
                        margin_jk = logits[j, yj] - logits[j, k]
                        if margin_jk > 0:
                            bound = margin_jk / (-sc[k])
                            if bound < alpha_max:
                                alpha_max = bound
            
            # 2. Weight bounds
            # For true row: d = x
            w_true = W[true]
            for f_idx in range(D):
                xf = x[f_idx]
                if xf > 0:
                    bound = (1000.0 - w_true[f_idx]) / xf
                    if bound < alpha_max:
                        alpha_max = bound
                elif xf < 0:
                    bound = (-1000.0 - w_true[f_idx]) / xf
                    if bound < alpha_max:
                        alpha_max = bound
            # For pred row: d = -x
            w_pred = W[pred]
            for f_idx in range(D):
                xf = x[f_idx]
                d = -xf
                if d > 0:
                    bound = (1000.0 - w_pred[f_idx]) / d
                    if bound < alpha_max:
                        alpha_max = bound
                elif d < 0:
                    bound = (-1000.0 - w_pred[f_idx]) / d
                    if bound < alpha_max:
                        alpha_max = bound
            
            # Choose alpha
            if alpha_max == np.inf:
                alpha = alpha_min * 1.1
            else:
                if alpha_max > alpha_min * 1.1:
                    alpha = alpha_min * 1.1
                else:
                    alpha = (alpha_min + alpha_max) / 2.0
            if alpha < alpha_min:
                alpha = alpha_min
            if alpha_max != np.inf and alpha > alpha_max:
                alpha = alpha_max * 0.99
            
            # Apply update
            W[true] += alpha * x
            W[pred] -= alpha * x
            
            # Update logits for all samples
            logits[:, true] += alpha * s_all
            logits[:, pred] -= alpha * s_all
    
    # Ensure weights are within bounds
    np.clip(W, -1000.0, 1000.0, out=W)
    
    # Update w
    w["head.weight"] = W.astype(np.float32)
    # bias unchanged
    
    return w
