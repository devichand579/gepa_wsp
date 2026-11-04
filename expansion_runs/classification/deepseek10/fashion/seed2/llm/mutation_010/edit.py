import numpy as np

def edit(w, f):
    W = w["head.weight"].astype(np.float64)
    b = w["head.bias"].astype(np.float64)
    Z = f["logits"].astype(np.float64)          # [2000,10] current logits
    y = f["y"].astype(np.int64)

    # CE as a function of temperature c (logits scale by c)
    def ce(c):
        Zc = c * Z
        m = Zc.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(Zc - m).sum(axis=1))
        return float(np.mean(lse - Zc[np.arange(len(y)), y]))

    # one-shot grid over c (scalar hyperparameter), no weight training
    grid = np.concatenate([
        np.linspace(1.0, 6.0, 51),
        np.linspace(6.0, 30.0, 49),
        np.linspace(30.0, 200.0, 35),
    ])
    ces = np.array([ce(c) for c in grid])
    best_c = float(grid[int(np.argmin(ces))])
    ce0 = ce(1.0)
    ce_best = float(ces.min())

    # sanity: argmax must be preserved
    acc_before = float(np.mean(np.argmax(Z, axis=1) == y))
    acc_after = float(np.mean(np.argmax(best_c * Z, axis=1) == y))

    w["head.weight"] = (best_c * W).astype(np.float32)
    w["head.bias"] = (best_c * b).astype(np.float32)

    # clip to safety
    for k in ("head.weight", "head.bias"):
        v = w[k]
        np.clip(v, -1000.0, 1000.0, out=v)
        w[k] = v.astype(np.float32)

    return {
        "weights": w,
        "diagnostics": {
            "temperature": best_c,
            "ce_before": ce0,
            "ce_after": ce_best,
            "acc_before": acc_before,
            "acc_after": acc_after,
            "head_weight_norm": float(np.linalg.norm(w["head.weight"])),
            "head_bias_norm": float(np.linalg.norm(w["head.bias"])),
        },
    }
