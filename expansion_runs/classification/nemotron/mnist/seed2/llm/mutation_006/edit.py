def edit(w, f):
    """
    Edit the neural network weights using a closed‑form ridge regression on the
    full feedback pool. Only fc2 is modified; fc1 stays unchanged so downstream
    cached features remain valid.

    Parameters
    ----------
    w : dict
        Current full parameter dictionary containing at least
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias".
    f : dict
        Feedback arrays with the exact keys/shapes listed in the prompt.
        Required keys: "features" (shape [2000,128]), "y" (shape [2000]).

    Returns
    -------
    {"weights": w, "diagnostics": dict}
        Updated weight dictionary and a small JSON‑compatible statistics dict.
    """
    import numpy as np

    # ----- feedback data -----
    features = f["features"]               # (2000, 128) float32
    y        = f["y"]                      # (2000,) int64

    N, D = features.shape                  # N=2000, D=128
    C = 10                                 # number of classes

    # one‑hot encode the labels
    Y_onehot = np.eye(C, dtype=np.float32)[y]   # (2000, 10)

    # ridge regularisation (lambda = 0 → pure least‑squares)
    lam = np.float32(0.0)

    # ----- closed‑form solution for fc2 weight -----
    FT_F = features.T @ features                # (128, 128)
    FT_Y = features.T @ Y_onehot                # (128, 10)

    FT_F_reg = FT_F + lam * np.eye(D, dtype=np.float32)

    # solve (FᵀF + λI) Wᵀ = FᵀY  →  Wᵀ = (FᵀF+λI)⁻¹ FᵀY
    A = np.linalg.solve(FT_F_reg, FT_Y)         # (128, 10)
    new_fc2_weight = A.T                        # (10, 128)  → matches [out,in]

    # ----- bias term -----
    # residuals = Y_onehot - features @ Wᵀ
    residuals = Y_onehot - (features @ new_fc2_weight.T)   # (2000, 10)
    new_fc2_bias = residuals.mean(axis=0)                 # (10,)

    # ----- update weights (preserve shapes & float32) -----
    w["fc2.weight"] = new_fc2_weight.astype(np.float32)
    w["fc2.bias"]   = new_fc2_bias.astype(np.float32)

    # ----- diagnostics -----
    diag = {
        "fc2_weight_mean": float(w["fc2.weight"].mean()),
        "fc2_weight_std":  float(w["fc2.weight"].std()),
        "fc2_bias_mean":   float(w["fc2.bias"].mean()),
        "fc2_bias_std":    float(w["fc2.bias"].std()),
        "ridge_lambda":    float(lam),
    }

    return {"weights": w, "diagnostics": diag}
