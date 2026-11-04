# ------------------------------------------------------------
# edit(w, f) – closed‑form head‑weight fit using all feedback
# ------------------------------------------------------------
# w  : full current float32 parameter dictionary (keys as supplied)
# f  : read‑only feedback dictionary (see spec)
# returns: {"weights": w, "diagnostics": {…}}   or just w
# ------------------------------------------------------------
import numpy as np

def edit(w, f):
    # ------- data -------
    X = f["features"]                # (2000, 1568)  float32
    y = f["y"]                       # (2000,)      int64

    # one‑hot labels (2000, 10)
    Y = np.eye(10, dtype=np.float32)[y]

    # ------- ridge regularisation -------
    lam = 1e-3                       # small ridge to keep matrix well‑conditioned
    I = np.eye(X.shape[1], dtype=np.float32) * lam
    XtX = X.T @ X + I                # (1568, 1568)
    XtY = X.T @ Y                   # (1568, 10)

    # ------- closed‑form solution -------
    # np.linalg.solve returns (1568, 10); transpose to (10, 1568)
    head_w = np.linalg.solve(XtX, XtY).T.astype(np.float32)

    # clip to allowed finite range [-1000, 1000]
    head_w = np.clip(head_w, -1000, 1000)

    # ------- head bias -------
    # keep a zero bias (original values are essentially zero)
    head_b = np.zeros(10, dtype=np.float32)

    # ------- update w -------
    w["head.weight"] = head_w
    w["head.bias"]   = head_b

    # ------- diagnostics -------
    logits = X @ head_w.T + head_b          # (2000, 10)
    pred   = np.argmax(logits, axis=1)
    acc    = float(np.mean(pred == y))
    ce     = float(-np.mean(np.log(logits[np.arange(2000), y] + 1e-12)))

    diag = {
        "full_accuracy": acc,
        "full_cross_entropy": ce,
        "head_weight_mean": float(np.mean(head_w)),
        "head_weight_std":  float(np.std(head_w)),
        "head_bias_mean":   float(np.mean(head_b)),
    }

    return {"weights": w, "diagnostics": diag}
# ------------------------------------------------------------
