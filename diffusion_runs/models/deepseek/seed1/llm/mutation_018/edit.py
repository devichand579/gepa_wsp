import numpy as np
import numpy.lib.stride_tricks as stride_tricks

def edit(w, f):
    # Extract feedback arrays
    F = f['features']          # [2000,64,16,16]
    Y = f['y']                 # [2000,1,16,16]
    X = f['x']                 # [2000,1,16,16]
    T = f['t']                 # [2000]
    N, C, H, W_ = F.shape
    
    # Helper: forward pass through conv layers (no autograd, just numpy)
    def conv2d(x, weight, bias, pad=1):
        # x: [N, Cin, H, W], weight: [Cout, Cin, 3, 3]
        N, Cin, H, W = x.shape
        Cout = weight.shape[0]
        x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
        # Use sliding_window_view for efficiency
        patches = stride_tricks.sliding_window_view(x_pad, (3,3), axis=(2,3))  # [N,Cin,H,W,3,3]
        patches = patches.transpose(0,2,3,1,4,5)  # [N,H,W,Cin,3,3]
        patches = patches.reshape(N*H*W, Cin*9)
        w_flat = weight.reshape(Cout, Cin*9).T  # [Cin*9, Cout]
        out = patches @ w_flat + bias  # [N*H*W, Cout]
        out = out.reshape(N, H, W, Cout).transpose(0,3,1,2)  # [N,Cout,H,W]
        return out

    def forward(w, x, t):
        # Build input: concat x with t/31 broadcast
        t_norm = (t.astype(np.float32) / 31.0).reshape(-1,1,1,1)
        t_channel = np.broadcast_to(t_norm, (x.shape[0],1,x.shape[2],x.shape[3]))
        inp = np.concatenate([x, t_channel], axis=1)  # [N,2,16,16]
        # conv1 + ReLU
        h1 = conv2d(inp, w['conv1.weight'], w['conv1.bias'])
        h1 = np.maximum(h1, 0)
        # conv2 + ReLU
        h2 = conv2d(h1, w['conv2.weight'], w['conv2.bias'])
        h2 = np.maximum(h2, 0)
        # conv3 + ReLU
        h3 = conv2d(h2, w['conv3.weight'], w['conv3.bias'])
        h3 = np.maximum(h3, 0)
        # out (linear)
        out = conv2d(h3, w['out.weight'], w['out.bias'])
        return out, h3

    # --- Variant 1: refit only the out layer on cached features ---
    def refit_out(features, targets):
        N, C, H, W_ = features.shape
        # Build design matrix in chunks
        XtX = np.zeros((C*9+1, C*9+1), dtype=np.float64)
        Xty = np.zeros((C*9+1,), dtype=np.float64)
        chunk = 100
        for start in range(0, N, chunk):
            end = min(start+chunk, N)
            F_chunk = features[start:end]
            Y_chunk = targets[start:end]
            F_pad = np.pad(F_chunk, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
            patches = stride_tricks.sliding_window_view(F_pad, (3,3), axis=(2,3))  # [chunk,C,H,W,3,3]
            patches = patches.transpose(0,2,3,1,4,5)  # [chunk,H,W,C,3,3]
            X_chunk = patches.reshape(-1, C*9)
            X_chunk = np.concatenate([X_chunk, np.ones((X_chunk.shape[0],1), dtype=X_chunk.dtype)], axis=1)
            y_chunk = Y_chunk.reshape(-1)
            XtX += X_chunk.T @ X_chunk
            Xty += X_chunk.T @ y_chunk
        # Small ridge for numerical stability
        XtX += 1e-6 * np.eye(XtX.shape[0])
        W_sol = np.linalg.solve(XtX, Xty)
        out_w = W_sol[:C*9].reshape(1, C, 3, 3).astype(np.float32)
        out_b = W_sol[C*9:C*9+1].astype(np.float32)
        return out_w, out_b

    # Compute MSE of a given set of weights on the full training set
    def compute_mse(w):
        pred, _ = forward(w, X, T)
        return np.mean((pred - Y)**2)

    # Current MSE
    mse_current = compute_mse(w)
    
    # Refit out on cached features
    out_w_refit, out_b_refit = refit_out(F, Y)
    w_refit = {k: v.copy() for k, v in w.items()}
    w_refit['out.weight'] = out_w_refit
    w_refit['out.bias'] = out_b_refit
    mse_refit = compute_mse(w_refit)
    
    # --- Variant 2: enrich conv2 and conv3 with structured mixing, then refit out ---
    # We create new conv2 and conv3 weights that combine identity, spatial diffusion, and channel mixing.
    # This is a deterministic, non-iterative heuristic.
    def make_enriched_conv(in_ch, out_ch, seed=0):
        rng = np.random.RandomState(seed)
        w_new = np.zeros((out_ch, in_ch, 3, 3), dtype=np.float32)
        # Identity passthrough for min(in_ch, out_ch) channels
        for i in range(min(in_ch, out_ch)):
            w_new[i, i, 1, 1] = 1.0
        # Spatial diffusion: add small weights to neighbors
        for i in range(min(in_ch, out_ch)):
            w_new[i, i, 0, 1] = 0.1
            w_new[i, i, 1, 0] = 0.1
            w_new[i, i, 1, 2] = 0.1
            w_new[i, i, 2, 1] = 0.1
        # Channel mixing: add small random weights
        for i in range(out_ch):
            for j in range(in_ch):
                if i != j:
                    w_new[i, j, 1, 1] = 0.05 * rng.randn()
        return w_new

    w_enriched = {k: v.copy() for k, v in w.items()}
    w_enriched['conv2.weight'] = make_enriched_conv(32, 64, seed=1)
    w_enriched['conv2.bias'] = np.zeros(64, dtype=np.float32)
    w_enriched['conv3.weight'] = make_enriched_conv(64, 64, seed=2)
    w_enriched['conv3.bias'] = np.zeros(64, dtype=np.float32)
    
    # Recompute features with enriched hidden layers
    _, F_enriched = forward(w_enriched, X, T)
    # Refit out on new features
    out_w_enriched, out_b_enriched = refit_out(F_enriched, Y)
    w_enriched['out.weight'] = out_w_enriched
    w_enriched['out.bias'] = out_b_enriched
    mse_enriched = compute_mse(w_enriched)
    
    # Choose the best variant
    if mse_refit <= mse_enriched and mse_refit <= mse_current:
        best_w = w_refit
        best_mse = mse_refit
        chosen = 'refit'
    elif mse_enriched <= mse_refit and mse_enriched <= mse_current:
        best_w = w_enriched
        best_mse = mse_enriched
        chosen = 'enriched'
    else:
        best_w = w
        best_mse = mse_current
        chosen = 'current'
    
    # Ensure all arrays are float32 and within bounds
    for k in best_w:
        best_w[k] = np.clip(best_w[k].astype(np.float32), -1000, 1000)
    
    diagnostics = {
        'mse_current': float(mse_current),
        'mse_refit': float(mse_refit),
        'mse_enriched': float(mse_enriched),
        'chosen': chosen,
        'final_mse': float(best_mse)
    }
    
    return {'weights': best_w, 'diagnostics': diagnostics}
