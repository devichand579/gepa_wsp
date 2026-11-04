import numpy as np

def edit(w, f):
    """
    Closed‑form update of the output convolution using globally‑pooled
    final‑layer features and the target epsilon.

    Permitted feedback arrays (taken from ``f``):
        - x          : noisy x_t   [2000,1,16,16]
        - t          : timestep      [2000]  int64
        - y          : target ε       [2000,1,16,16]
        - features   : final hidden
                       feature maps [2000,64,16,16]   (conv3.output)
        - prediction : current ε‑pred [2000,1,16,16]
        - alphabar   : 32 schedule coeff.

    Only ``features`` and ``y`` are used; all other arrays are ignored but
    kept intact.
    """
    # ------------------------------------------------------------------
    # 1. Extract and reshape the data we will actually use.
    # ------------------------------------------------------------------
    # Final hidden features: [2000, 64, 16, 16]
    features = f['features']                     # float32
    # Target epsilon: [2000, 1, 16, 16]
    y        = f['y']                            # float32

    # Global average pool over the 16×16 spatial grid.
    #   • features  ->  [2000, 64]
    #   • y         ->  [2000]   (single scalar per example)
    f_pooled = features.mean(axis=(2, 3))      # (2000, 64)
    y_flat   = y.mean(axis=(1, 2))             # (2000,)

    # Ensure float32 (the spec requires finite float32, |val| ≤ 1000)
    F = np.asarray(f_pooled, dtype=np.float32)
    y_vec = np.asarray(y_flat, dtype=np.float32)

    # ------------------------------------------------------------------
    # 2. Closed‑form linear regression:  w = (FᵀF)^{-1} Fᵀ y
    # ------------------------------------------------------------------
    # np.linalg.lstsq solves  F @ w ≈ y_vec  without forming the inverse.
    # It returns the least‑squares solution w of shape (64,).
    w_sol, residues, rank, s = np.linalg.lstsq(F, y_vec, rcond=None)

    # Reshape to (1, 64) – this is the “output‑weight vector”.
    w_sol = w_sol.reshape(1, 64)          # (1, 64)

    # ------------------------------------------------------------------
    # 3. Map the 1×64 vector onto the network’s output tensor.
    #    The original out.weight has shape [1, 64, 3, 3].
    #    We broadcast the vector across the 3×3 spatial kernel (repeat
    #    along both dimensions).  This preserves the exact tensor shape
    #    and keeps all other parameters untouched.
    # ------------------------------------------------------------------
    new_w = {k: v.copy() for k, v in w.items()}

    # out.weight  : [1, 64, 3, 3]
    # Start from zeros, then fill with the learned vector tiled 3×3.
    out_w = np.zeros_like(new_w['out.weight'], dtype=np.float32)
    # w_sol is (1,64); broadcast to (1,64,1,1) then tile.
    tile = w_sol[:, :, np.newaxis, np.newaxis]      # (1,64,1,1)
    out_w[:] = np.repeat(np.repeat(tile, 3, axis=2), 3, axis=3)
    new_w['out.weight'] = out_w

    # out.bias    : shape [1]  (scalar)
    # Bias = mean( y - F @ w )
    bias_val = float(np.mean(y_vec - F @ w_sol))
    new_w['out.bias'] = np.array(bias_val, dtype=np.float32)

    # ------------------------------------------------------------------
    # 4. Return the updated weight dictionary and a tiny diagnostics block.
    # ------------------------------------------------------------------
    diagnostics = {
        'weights_updated': ['out.weight', 'out.bias'],
        'output_weight_norm': float(np.linalg.norm(w_sol)),
        'bias': float(bias_val),
        'samples_used': int(F.shape[0])
    }

    return new_w, diagnostics
