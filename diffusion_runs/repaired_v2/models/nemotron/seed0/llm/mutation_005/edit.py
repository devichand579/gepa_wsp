import numpy as np

def edit(w, f):
    """
    Closed‑form edit of the DDPM epsilon‑prediction network.
    Adjusts the output bias by the mean residual (y - prediction)
    so that all predictions are shifted uniformly.
    No autograd, back‑propagation, or iterative optimiser is used.
    """
    # Pull the permitted feedback arrays (shapes are fixed by the task)
    y       = f['feedback_arrays']['y']          # target Gaussian epsilon, shape [2000,1,16,16]
    pred    = f['feedback_arrays']['prediction']# current epsilon prediction, same shape

    # Compute the mean residual over every element (scalar)
    residual = y - pred
    mean_res = float(np.mean(residual).astype(np.float32))

    # Update the output bias (shape [1]) by the computed shift
    w['out.bias'] = (w['out.bias'] + mean_res).astype(np.float32)

    # Keep values finite and within the allowed range |value| <= 1000
    if abs(w['out.bias']) > 1000:
        w['out.bias'] = np.clip(w['out.bias'], -1000, 1000).astype(np.float32)

    # Return the modified weights together with a tiny diagnostics dict
    return {"weights": w,
            "diagnostics": {"bias_shift": mean_res}}
