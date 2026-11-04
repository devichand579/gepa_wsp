import numpy as np

def edit(w, f):
    # Read current weights
    fc2_w = w["fc2.weight"].copy()
    fc2_b = w["fc2.bias"].copy()
    
    # Get full feedback features and labels
    X = f["features"]  # shape (2000, 128)
    y = f["y"]         # shape (2000,)
    
    # Compute standard deviation of each feature across the full pool
    feat_std = np.std(X, axis=0)  # shape (128,)
    
    # Identify features with very small standard deviation (near-constant)
    # From the provided statistics, indices 10-18 have std < 0.02.
    # Use a conservative threshold to avoid removing useful features.
    threshold = 0.05
    dead_features = np.where(feat_std < threshold)[0]
    
    # Zero out the corresponding columns in fc2.weight
    if len(dead_features) > 0:
        fc2_w[:, dead_features] = 0.0
    
    # Update the weights dictionary
    w["fc2.weight"] = fc2_w.astype(np.float32)
    w["fc2.bias"] = fc2_b.astype(np.float32)
    
    # Diagnostics
    diagnostics = {
        "dead_features_count": int(len(dead_features)),
        "dead_features_indices": dead_features.tolist(),
        "feat_std_min": float(feat_std.min()),
        "feat_std_max": float(feat_std.max()),
        "fc2_weight_norm_before": float(np.linalg.norm(w["fc2.weight"])),
        "fc2_weight_norm_after": float(np.linalg.norm(fc2_w)),
    }
    
    return {"weights": w, "diagnostics": diagnostics}
