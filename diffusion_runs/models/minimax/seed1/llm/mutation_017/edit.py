import numpy as np

def _conv(x, W, b):
    # x: (N,C,H,W) float32; W: (O,C,3,3); b: (O,) -> cross-correlation, pad 1
    N, C, H, Wd = x.shape
    O = W.shape[0]
    xp = np.pad(x, ((0,0),(0,0),(1,1),(1,1)))
    sw = np.lib.stride_tricks.sliding_window_view(xp, (3,3), axis=(2,3))  # (N,C,H,W,3,3)
    sw = sw.transpose(0,2,3,1,4,5).reshape(N*H*Wd, C*9)
    Wm = W.reshape(O, C*9).T
    out = sw @ Wm
    out = out.reshape(N,H,Wd,O).transpose(0,3,1,2)
    if b is not None:
        out = out + b.reshape(1,-1,1,1)
    return out

def _patches(feat, k=3):
    # feat: (n,64,16,16) -> (n*256, 576) with row = pixel-major, col = (ch,ky,kx)
    n, C, H, Wd = feat.shape
    xp = np.pad(feat, ((0,0),(0,0),(1,1),(1,1)))
    sw = np.lib.stride_tricks.sliding_window_view(xp, (k,k), axis=(2,3))  # (n,C,H,W,k,k)
    sw = sw.transpose(0,2,3,1,4,5).reshape(n*H*Wd, C*k*k)
    return sw

def edit(w, f):
    W = {k: np.array(v, dtype=np.float64) for k, v in w.items()}
    diag = {}

    x  = np.asarray(f["x"], dtype=np.float32)
    t  = np.asarray(f["t"]).astype(np.int64)
    y  = np.asarray(f["y"], dtype=np.float32)
    clean = np.asarray(f["clean"], dtype=np.float32)
    feats = np.asarray(f["features"], dtype=np.float32)
    pred0 = np.asarray(f["prediction"], dtype=np.float32)
    ab = np.asarray(f["alphabar"], dtype=np.float64)

    ids = np.asarray(f["batch.ids"]).astype(np.int64)
    bxc = np.asarray(f["batch.x"], dtype=np.float32)
    bt  = np.asarray(f["batch.t"]).astype(np.int64)

    # ---------------- 1. verify forward semantics against layer traces ----------------
    l1i = np.asarray(f["layer.conv1.input0"], dtype=np.float32)
    l1o = np.asarray(f["layer.conv1.output"], dtype=np.float32)
    l2i = np.asarray(f["layer.conv2.input0"], dtype=np.float32)
    l2o = np.asarray(f["layer.conv2.output"], dtype=np.float32)
    l3i = np.asarray(f["layer.conv3.input0"], dtype=np.float32)
    l3o = np.asarray(f["layer.conv3.output"], dtype=np.float32)
    loi = np.asarray(f["layer.out.input0"], dtype=np.float32)
    loo = np.asarray(f["layer.out.output"], dtype=np.float32)

    c1 = _conv(l1i, W["conv1.weight"], W["conv1.bias"])
    d_raw1 = float(np.abs(c1 - l1o).max())
    d_relu1 = float(np.abs(np.maximum(c1,0) - l1o).max())
    relu_after = d_relu1 < d_raw1
    diag["conv1_raw_diff"] = d_raw1
    diag["conv1_relu_diff"] = d_relu1

    def act(z):
        return np.maximum(z, 0.0) if relu_after else z

    a1 = act(c1)
    diag["conv2_input_matches_act"] = float(np.abs(a1 - l2i).max())
    diag["conv2_input_matches_raw"] = float(np.abs(c1 - l2i).max())

    c2 = _conv(l2i, W["conv2.weight"], W["conv2.bias"])
    a2 = act(c2)
    diag["conv3_input_matches_act"] = float(np.abs(a2 - l3i).max())
    diag["conv3_input_matches_raw"] = float(np.abs(c2 - l3i).max())

    c3 = _conv(l3i, W["conv3.weight"], W["conv3.bias"])
    a3 = act(c3)
    diag["out_input_matches_act"] = float(np.abs(a3 - loi).max())
    diag["out_input_matches_raw"] = float(np.abs(c3 - loi).max())
    diag["conv3_out_vs_out_in"] = float(np.abs(l3o - loi).max())

    # which of the candidate tensors is the true out-layer input?
    cands = {"act_conv3": a3, "raw_conv3": c3, "l3o": l3o, "l2o": l2o}
    best, bestd = None, 1e30
    for k_, v_ in cands.items():
        d_ = float(np.abs(v_ - loi).max())
        if d_ < bestd:
            bestd, best = d_, k_
    diag["out_input_source"] = best
    diag["out_input_diff"] = bestd

    # does cached 'features' equal the out-layer input for the batch?
    fb = feats[ids]
    diag["feat_vs_outin"] = float(np.abs(fb - loi).max())
    diag["feat_vs_l3o"] = float(np.abs(fb - l3o).max())
    diag["feat_vs_act3"] = float(np.abs(fb - a3).max())
    diag["feat_vs_raw3"] = float(np.abs(fb - c3).max())
    feats_is_outin = diag["feat_vs_outin"] < 1e-4

    # ---------------- 2. baseline MSE from supplied prediction ----------------
    mse0 = float(np.mean((pred0 - y) ** 2))
    diag["mse_old"] = mse0

    # ---------------- 3. closed-form readout fit ----------------
    def fit_readout(F, Y, lam=1e-3, chunk=64):
        """F: (n,64,16,16) hidden maps, Y: (n,1,16,16) targets.
        Returns (Wout (1,64,3,3), bout (1,)) solving ridge least squares."""
        n = F.shape[0]
        A = np.zeros((576, 576), dtype=np.float64)
        B = np.zeros((576, 256), dtype=np.float64)
        s_y = np.zeros(256, dtype=np.float64)
        s_yy = 0.0
        cnt = 0
        for i in range(0, n, chunk):
            Fc = F[i:i+chunk]
            Yc = Y[i:i+chunk]
            P = _patches(Fc).astype(np.float64)            # (m,576)
            T = Yc[:, 0].reshape(-1, 256).astype(np.float64)  # (m,256)
            A += P.T @ P
            B += P.T @ T
            s_y += T.sum(axis=0)
            s_yy += float((T*T).sum())
            cnt += P.shape[0]
        # centered ridge (intercept = mean target)
        mu = s_y / cnt
        # center A: A_c = A - cnt*mu*mu^T ; B_c = B - cnt*mu*mu^T  (since sum P not tracked,
        # use the fact that we also need sum of P; track it)
        return A, B, mu, s_yy, cnt

    # we need sum of P as well -> redo with tracking
    def fit_full(F, Y, lam=1e-3, chunk=64):
        n = F.shape[0]
        A = np.zeros((576, 576), dtype=np.float64)
        B = np.zeros((576, 256), dtype=np.float64)
        SP = np.zeros(576, dtype=np.float64)
        SY = np.zeros(256, dtype=np.float64)
        SYY = 0.0
        cnt = 0
        for i in range(0, n, chunk):
            Fc = F[i:i+chunk]
            Yc = Y[i:i+chunk]
            P = _patches(Fc).astype(np.float64)
            T = Yc[:, 0].reshape(-1, 256).astype(np.float64)
            A += P.T @ P
            B += P.T @ T
            SP += P.sum(axis=0)
            SY += T.sum(axis=0)
            SYY += float((T*T).sum())
            cnt += P.shape[0]
        mu = SY / cnt
        Ac = A - np.outer(SP, SP) / cnt
        Bc = B - np.outer(SP, SY) / cnt
        tr = np.trace(Ac) / 576.0
        R = Ac + lam * max(tr, 1e-12) * np.eye(576)
        Wo = np.linalg.solve(R, Bc)            # (576,256)
        bo = mu - (SP / cnt) @ Wo               # (256,)
        # in-sample mse
        num = SYY - 2.0 * float((bo @ SY) + (Wo.ravel() @ B.ravel())) + \
              float(bo @ bo) * cnt + float(np.sum(Wo * (Ac @ Wo)))
        mse = num / (cnt * 256.0)
        return Wo, bo, mse

    Wo = None; bo = None; mse_fit = None
    if feats_is_outin:
        Wo, bo, mse_fit = fit_full(feats, y)
        diag["fit_source"] = "cached_features"
    else:
        # recompute hidden representation ourselves (self-contained fallback)
        n_all = x.shape[0]
        F_all = np.zeros((n_all, 64, 16, 16), dtype=np.float32)
        tt = (t.astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
        chunk = 100
        for i in range(0, n_all, chunk):
            xc = x[i:i+chunk]
            inp = np.concatenate([xc, np.broadcast_to(tt[i:i+chunk], (xc.shape[0], 1, 16, 16))], axis=1)
            h1 = act(_conv(inp, W["conv1.weight"], W["conv1.bias"]))
            h2 = act(_conv(h1, W["conv2.weight"], W["conv2.bias"]))
            h3 = act(_conv(h2, W["conv3.weight"], W["conv3.bias"]))
            F_all[i:i+chunk] = h3.astype(np.float32)
        Wo, bo, mse_fit = fit_full(F_all, y)
        diag["fit_source"] = "recomputed_hidden"

    Wout = Wo.reshape(1, 64, 3, 3)
    bout = bo.reshape(1)

    # ---------------- 4. independent verification with our own forward ----------------
    def forward_mse(Wc1, b1, Wc2, b2, Wc3, b3, Wo_, bo_, idx=None):
        if idx is None:
            xx, ttv, yy = x, t, y
        else:
            xx, ttv, yy = x[idx], t[idx], y[idx]
        n = xx.shape[0]
        tot = 0.0; cnt = 0
        chunk = 100
        for i in range(0, n, chunk):
            xc = xx[i:i+chunk]
            tbc = (ttv[i:i+chunk].astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
            inp = np.concatenate([xc, np.broadcast_to(tbc, (xc.shape[0], 1, 16, 16))], axis=1)
            h1 = act(_conv(inp, Wc1, b1))
            h2 = act(_conv(h1, Wc2, b2))
            h3 = act(_conv(h2, Wc3, b3))
            o = _conv(h3, Wo_, bo_)
            d = o - yy[i:i+chunk].astype(np.float64)
            tot += float((d*d).sum()); cnt += d.size
        return tot / cnt

    # sanity: our forward must reproduce the supplied prediction on the batch
    bidx = ids
    mse_pred_check = forward_mse(W["conv1.weight"], W["conv1.bias"], W["conv2.weight"],
                                 W["conv2.bias"], W["conv3.weight"], W["conv3.bias"],
                                 W["out.weight"], W["out.bias"], idx=bidx)
    diag["fwd_check_mse_on_batch"] = mse_pred_check
    diag["supplied_pred_mse_on_batch"] = float(np.mean((pred0[bidx] - y[bidx]) ** 2))

    mse_new = forward_mse(W["conv1.weight"], W["conv1.bias"], W["conv2.weight"], W["conv2.bias"],
                          W["conv3.weight"], W["conv3.bias"], Wout, bout)
    diag["mse_new"] = float(mse_new)
    diag["mse_fit_closedform"] = float(mse_fit)

    # ---------------- 5. assemble output weights ----------------
    out = {}
    for k in w.keys():
        out[k] = np.array(w[k], dtype=np.float32)
    out["out.weight"] = np.clip(Wout.astype(np.float32), -1000.0, 1000.0)
    out["out.bias"] = np.clip(bout.astype(np.float32), -1000.0, 1000.0)

    # ---------------- 6. optional closed-form improvement of conv3 (greedy layer fit) ----
    # Only attempted if the readout fit is verified good AND time permits.
    # We fit conv3 (and its bias) so that its post-ReLU output better explains the target
    # through the (already fitted) linear readout:  min || Wout^T h3_raw + b_out - y ||^2
    # solved per output channel by ridge on the 64-dim receptive field vectors.
    try:
        if float(mse_new) < 0.9 * mse0:
            # recompute hidden maps (input of conv3) for all examples
            n_all = x.shape[0]
            H2 = np.zeros((n_all, 64, 16, 16), dtype=np.float32)
            tt = (t.astype(np.float32) / 31.0).reshape(-1, 1, 1, 1)
            chunk = 100
            for i in range(0, n_all, chunk):
                xc = x[i:i+chunk]
                inp = np.concatenate([xc, np.broadcast_to(tt[i:i+chunk], (xc.shape[0], 1, 16, 16))], axis=1)
                h1 = act(_conv(inp, W["conv1.weight"], W["conv1.bias"]))
                h2 = act(_conv(h1, W["conv2.weight"], W["conv2.bias"]))
                H2[i:i+chunk] = h2.astype(np.float32)

            # For each conv3 output channel c, the readout contributes
            #   out(p) = sum_c Wo[c] . patch_c(p) + bo
            # patch_c(p) = ReLU( sum_k W3[c,k] . patch_k(p) + b3[c] )
            # We choose W3[c,:] to make ReLU(...) match a target "pre-activation" field
            # g_c(p) = (target contribution) so that ReLU(g_c) approximates it.
            # Least squares: for each c, we want W3[c] . patch_k(p) ~ g_c(p) - b3[c].
            # Choose g_c(p) = sign(Wo[c]) * |Wo[c]| * patch_c_current(p)  (keep current shape)
            # -> this is a self-consistent "distillation" of the readout into conv3.
            # Simpler and safer: keep conv3 as is.
            pass
    except Exception:
        pass

    diag["out_w_absmean"] = float(np.abs(out["out.weight"]).mean())
    diag["out_b_mean"] = float(out["out.bias"].mean())
    return {"weights": out, "diagnostics": diag}
