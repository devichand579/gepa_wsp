import numpy as np
import math

def edit(w, f):
    # ---- helpers ----
    def conv_forward(x, weight, bias):
        # x: (N,Cin,H,W), weight: (Cout,Cin,3,3), bias: (Cout,), pad=1
        N, Cin, H, W = x.shape
        Cout = weight.shape[0]
        xp = np.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        out = np.zeros((N, Cout, H, W), dtype=np.float32)
        for di in range(3):
            for dj in range(3):
                patch = xp[:, :, di:di+H, dj:dj+W]  # (N,Cin,H,W)
                # out[n,o,i,j] += sum_c weight[o,c,di,dj]*patch[n,c,i,j]
                out += np.einsum('oc,ncij->noij', weight[:,:,di,dj], patch, optimize=True)
        out += bias[None,:,None,None]
        return out

    def relu(x):
        return np.maximum(x, 0.0)

    # ---- pull arrays ----
    x = np.asarray(f['x'], dtype=np.float32)          # (2000,1,16,16)
    t = np.asarray(f['t'], dtype=np.float32)          # (2000,)
    y = np.asarray(f['y'], dtype=np.float32)          # (2000,1,16,16)
    feats = np.asarray(f['features'], dtype=np.float32)  # (2000,64,16,16) = out input
    pred = np.asarray(f['prediction'], dtype=np.float32)

    N = x.shape[0]
    H = W = 16

    # ---- current MSE (sanity) ----
    cur_mse = float(np.mean((pred - y) ** 2))

    # =========================================================
    # Part A: closed-form ridge refit of the linear readout `out`
    #   pred[n,0,i,j] = sum_{c,di,dj} W[0,c,di,dj]*Fpad[n,c,i+di-1,j+dj-1] + b
    # Build normal equations G (577x577), h (577) using shifted features.
    # =========================================================
    F = feats  # (N,64,16,16)
    C = F.shape[1]  # 64
    K = 3
    # shifted features with zero padding: S[(di,dj)] shape (N,C,H,W)
    Fpad = np.pad(F, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    shifts = []
    for di in range(K):
        for dj in range(K):
            shifts.append(Fpad[:, :, di:di+H, dj:dj+W])  # (N,C,H,W)
    # design matrix columns: (c, di, dj) -> 64*9 = 576, plus bias
    ncol = C * K * K + 1

    # accumulate G = X^T X, h = X^T y  (float64 for stability)
    G = np.zeros((ncol, ncol), dtype=np.float64)
    h = np.zeros(ncol, dtype=np.float64)
    yv = y[:, 0, :, :].reshape(N, H * W).astype(np.float64)  # (N, 256)

    # Build per-example design in chunks to limit memory
    # For each example: X_n is (256, 576) from shifts, plus bias col.
    # We'll accumulate via matmul on reshaped (256, 576).
    for n in range(N):
        # Xn: (256, 576)
        cols = []
        for s in shifts:
            cols.append(s[n].reshape(C, H * W).T)  # (256, C)
        Xn = np.concatenate(cols, axis=1)  # (256, 576)
        Xb = np.concatenate([Xn, np.ones((H * W, 1), dtype=np.float64)], axis=1)  # (256,577)
        G += Xb.T @ Xb
        h += Xb.T @ yv[n]

    # tiny ridge for numerical stability
    lam = 1e-6 * np.trace(G) / ncol
    G_reg = G + lam * np.eye(ncol)
    try:
        sol = np.linalg.solve(G_reg, h)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(G_reg, h, rcond=None)[0]

    W_new = sol[:C * K * K].reshape(C, K, K).astype(np.float32)  # (C,3,3)
    b_new = np.float32(sol[-1])

    # compute refit MSE on the cached features (full 2000)
    def readout_mse(Wc, bc):
        # Wc: (C,3,3)
        acc = np.zeros((N, H * W), dtype=np.float64)
        for idx, (di, dj) in enumerate([(a, b) for a in range(K) for b in range(K)]):
            acc += (shifts[idx].reshape(N, C, H * W) * Wc[:, di, dj][None, :, None]).sum(axis=1)
        acc += bc
        return float(np.mean((acc - yv) ** 2))

    mse_refit = readout_mse(W_new, b_new)

    # =========================================================
    # Part B: optional enrichment of conv2/conv3 (deterministic,
    # no gradients). Build structured mixing operators, recompute
    # features by forward pass, refit out, keep if better.
    # =========================================================
    def build_enriched():
        # conv1 unchanged
        c1w = np.array(w['conv1.weight'], dtype=np.float32)
        c1b = np.array(w['conv1.bias'], dtype=np.float32)
        c2w = np.array(w['conv2.weight'], dtype=np.float32)
        c2b = np.array(w['conv2.bias'], dtype=np.float32)
        c3w = np.array(w['conv3.weight'], dtype=np.float32)
        c3b = np.array(w['conv3.bias'], dtype=np.float32)

        # Add small spatial diffusion + channel mixing to conv2/conv3
        # spatial Laplacian-like kernel
        lap = np.zeros((3, 3), dtype=np.float32)
        lap[1, 1] = 1.0
        lap[0, 1] = lap[2, 1] = lap[1, 0] = lap[1, 2] = -0.25

        # channel mixing: identity + small random-free deterministic mix
        def enrich(weight, bias, alpha_spatial, alpha_channel):
            Cout, Cin, _, _ = weight.shape
            w2 = weight.copy()
            # spatial diffusion on each (o,c) kernel
            for o in range(Cout):
                for c in range(Cin):
                    w2[o, c] += alpha_spatial * lap * weight[o, c, 1, 1]
            # channel mixing: mix output channels via a fixed orthogonal-ish matrix
            # use a simple deterministic mixing (Hadamard-like) scaled small
            mix = np.zeros((Cout, Cin), dtype=np.float32)
            for o in range(Cout):
                for c in range(Cin):
                    if o == c:
                        mix[o, c] = 1.0
                    else:
                        mix[o, c] = alpha_channel * math.cos((o + 1) * (c + 1) * 0.1) / max(Cin, 1)
            wmix = np.einsum('oc,ocxy->ocxy', mix, weight, optimize=True)
            w2 = w2 + 0.5 * wmix
            return w2, bias

        c2w2, c2b2 = enrich(c2w, c2b, 0.15, 0.05)
        c3w2, c3b2 = enrich(c3w, c3b, 0.15, 0.05)
        return c1w, c1b, c2w2, c2b2, c3w2, c3b2

    def forward_features(c1w, c1b, c2w, c2b, c3w, c3b, xin, tin):
        # inp: concat x with t/31
        tch = (tin / 31.0).reshape(-1, 1, 1, 1) * np.ones((xin.shape[0], 1, H, W), dtype=np.float32)
        inp = np.concatenate([xin, tch], axis=1).astype(np.float32)
        h1 = relu(conv_forward(inp, c1w, c1b))
        h2 = relu(conv_forward(h1, c2w, c2b))
        h3 = relu(conv_forward(h2, c3w, c3b))
        return h3

    best = {
        'mse': mse_refit,
        'out.weight': W_new.reshape(1, C, K, K),
        'out.bias': np.array([b_new], dtype=np.float32),
        'conv2.weight': np.array(w['conv2.weight'], dtype=np.float32),
        'conv2.bias': np.array(w['conv2.bias'], dtype=np.float32),
        'conv3.weight': np.array(w['conv3.weight'], dtype=np.float32),
        'conv3.bias': np.array(w['conv3.bias'], dtype=np.float32),
        'conv1.weight': np.array(w['conv1.weight'], dtype=np.float32),
        'conv1.bias': np.array(w['conv1.bias'], dtype=np.float32),
        'variant': 'readout_refit',
    }

    try:
        c1w, c1b, c2w2, c2b2, c3w2, c3b2 = build_enriched()
        # recompute features for all 2000 (in chunks to limit memory)
        F2 = np.empty_like(F)
        chunk = 250
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            F2[s:e] = forward_features(c1w, c1b, c2w2, c2b2, c3w2, c3b2, x[s:e], t[s:e])
        # refit out on F2
        F2pad = np.pad(F2, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
        shifts2 = []
        for di in range(K):
            for dj in range(K):
                shifts2.append(F2pad[:, :, di:di+H, dj:dj+W])
        G2 = np.zeros((ncol, ncol), dtype=np.float64)
        h2 = np.zeros(ncol, dtype=np.float64)
        for n in range(N):
            cols = [s[n].reshape(C, H * W).T for s in shifts2]
            Xn = np.concatenate(cols, axis=1)
            Xb = np.concatenate([Xn, np.ones((H * W, 1), dtype=np.float64)], axis=1)
            G2 += Xb.T @ Xb
            h2 += Xb.T @ yv[n]
        lam2 = 1e-6 * np.trace(G2) / ncol
        G2_reg = G2 + lam2 * np.eye(ncol)
        try:
            sol2 = np.linalg.solve(G2_reg, h2)
        except np.linalg.LinAlgError:
            sol2 = np.linalg.lstsq(G2_reg, h2, rcond=None)[0]
        W2 = sol2[:C * K * K].reshape(C, K, K).astype(np.float32)
        b2 = np.float32(sol2[-1])
        # mse of enriched
        acc = np.zeros((N, H * W), dtype=np.float64)
        for idx, (di, dj) in enumerate([(a, b) for a in range(K) for b in range(K)]):
            acc += (shifts2[idx].reshape(N, C, H * W) * W2[:, di, dj][None, :, None]).sum(axis=1)
        acc += b2
        mse_enriched = float(np.mean((acc - yv) ** 2))

        if mse_enriched < best['mse'] - 1e-6:
            best = {
                'mse': mse_enriched,
                'out.weight': W2.reshape(1, C, K, K),
                'out.bias': np.array([b2], dtype=np.float32),
                'conv2.weight': c2w2,
                'conv2.bias': c2b2,
                'conv3.weight': c3w2,
                'conv3.bias': c3b2,
                'conv1.weight': c1w,
                'conv1.bias': c1b,
                'variant': 'enriched_refit',
            }
    except Exception:
        pass

    # ---- write back ----
    w['out.weight'] = np.clip(best['out.weight'], -1000.0, 1000.0).astype(np.float32)
    w['out.bias'] = np.clip(best['out.bias'], -1000.0, 1000.0).astype(np.float32)
    w['conv1.weight'] = np.clip(best['conv1.weight'], -1000.0, 1000.0).astype(np.float32)
    w['conv1.bias'] = np.clip(best['conv1.bias'], -1000.0, 1000.0).astype(np.float32)
    w['conv2.weight'] = np.clip(best['conv2.weight'], -1000.0, 1000.0).astype(np.float32)
    w['conv2.bias'] = np.clip(best['conv2.bias'], -1000.0, 1000.0).astype(np.float32)
    w['conv3.weight'] = np.clip(best['conv3.weight'], -1000.0, 1000.0).astype(np.float32)
    w['conv3.bias'] = np.clip(best['conv3.bias'], -1000.0, 1000.0).astype(np.float32)

    diagnostics = {
        'current_mse': cur_mse,
        'readout_refit_mse': mse_refit,
        'chosen_variant': best['variant'],
        'chosen_mse': best['mse'],
        'n_train': int(N),
        'n_params_readout': int(ncol),
    }
    return {"weights": w, "diagnostics": diagnostics}
