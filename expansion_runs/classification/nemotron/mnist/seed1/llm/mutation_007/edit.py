def edit(w, f):
    """
    Edit the neural network weights using a closed‑form ridge‑regression fit
    on the entire feedback pool. Only `fc2` parameters are changed; `fc1`
    is left untouched so downstream cached features remain valid.
    
    Parameters
    ----------
    w : dict
        Current full parameter dictionary (contains fc1.weight, fc1.bias,
        fc2.weight, fc2.bias).
    f : dict
        Read‑only feedback arrays with keys:
          - f["features"]   shape (2000, 128)  float32
          - f["y"]        shape (2000,)       int64
          - plus other keys (logits, probs, batch ids, layer activations)
    
    Returns
    -------
    {"weights": w, "diagnostics": {...}}
        Updated weight dictionary and a tiny JSON‑compatible statistics dict.
    """
    # ----- feedback pool -----
    features = f["features"]                # (2000, 128), float32
    y        = f["y"]                       # (2000,), int64 labels 0..9
    N, D     = features.shape               # N=2000, D=128

    # one‑hot encoding of the true labels
    Y = np.eye(10, dtype=np.float32)[y]    # (2000, 10)

    # ----- ridge regulariser -----
    lam = np.float32(1e-3)                 # small L2 penalty

    # ----- normal equation: W = (FᵀF + λI)⁻¹ Fᵀ Y -----
    FtF = features.T @ features            # (128, 128)
    FtY = features.T @ Y                   # (128, 10)
    Reg = FtF + lam * np.eye(D, dtype=np.float32)
    W = np.linalg.solve(Reg, FtY)          # (128, 10)  mapping features → logits

    # ----- map to fc2.weight [out,in] = (10,128) -----
    # Current convention: logits = features @ fc2.weight.T + fc2.bias
    # So fc2.weight = W.T  (shape 10×128)
    new_weight = W.T.astype(np.float32)    # (10, 128)

    # ----- bias: mean residual per class -----
    resid = Y - features @ W               # (2000, 10)
    new_bias = resid.mean(axis=0).astype(np.float32)   # (10,)

    # ----- apply updates, keep everything else unchanged -----
    w["fc2.weight"] = new_weight
    w["fc2.bias"]   = new_bias

    # ----- tiny diagnostics (JSON‑compatible) -----
    diag = {
        "fc2_weight_norm": float(np.linalg.norm(new_weight)),
        "fc2_bias_mean":   float(new_bias.mean()),
        "feedback_accuracy_after": float(
            np.mean(np.argmax(np.dot(features, new_weight.T) + new_bias, axis=1) == y)
        )
    }
    return {"weights": w, "diagnostics": diag}
