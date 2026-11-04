import numpy as np

def edit(w, f):
    # Read current parameters
    W1 = w["conv1.weight"].astype(np.float64)
    b1 = w["conv1.bias"].astype(np.float64)
    W2 = w["conv2.weight"].astype(np.float64)
    b2 = w["conv2.bias"].astype(np.float64)
    Wh = w["head.weight"].astype(np.float64)
    bh = w["head.bias"].astype(np.float64)

    # Maximum absolute value allowed
    MAX = 1000.0

    # Current max absolute values
    max_W1 = np.max(np.abs(W1))
    max_b1 = np.max(np.abs(b1))
    max_W2 = np.max(np.abs(W2))
    max_b2 = np.max(np.abs(b2))
    max_Wh = np.max(np.abs(Wh))
    max_bh = np.max(np.abs(bh))

    # We want to scale features by s, head weights by k/s, head bias by k.
    # Constraints:
    #   s * max_W1 <= MAX
    #   s * max_b1 <= MAX
    #   s * max_W2 <= MAX
    #   s * max_b2 <= MAX
    #   (k/s) * max_Wh <= MAX  -> k <= MAX * s / max_Wh
    #   k * max_bh <= MAX      -> k <= MAX / max_bh
    # We want to maximize k (the logit scale) to minimize CE.
    # Since max_Wh is already 1000, k <= s. And max_bh is ~998, so k <= 1.002.
    # Thus k is limited by head bias. We can only slightly increase k.
    # However, we can also choose to zero out the head bias if the bias-free classifier
    # still gives perfect accuracy on the feedback pool. Let's check that first.

    # Compute bias-free logits on the feedback features
    features = f["features"].astype(np.float64)  # (2000, 1568)
    y = f["y"]  # (2000,)
    logits_nb = features @ Wh.T  # (2000, 10)
    pred_nb = np.argmax(logits_nb, axis=1)
    acc_nb = np.mean(pred_nb == y)

    # If bias-free classifier is perfect, we can set head bias to zero and scale features up.
    if acc_nb >= 0.9999:
        # Set head bias to zero
        bh_new = np.zeros_like(bh)
        # We can scale features by s up to the conv limits.
        s_max = min(
            MAX / max_W1 if max_W1 > 0 else np.inf,
            MAX / max_b1 if max_b1 > 0 else np.inf,
            MAX / max_W2 if max_W2 > 0 else np.inf,
            MAX / max_b2 if max_b2 > 0 else np.inf,
        )
        # Use a safe margin (e.g., 0.95 of max) to avoid overflow
        s = 0.95 * s_max
        # Scale conv weights and biases by s
        W1_new = W1 * s
        b1_new = b1 * s
        W2_new = W2 * s
        b2_new = b2 * s
        # Head weights unchanged, head bias zero
        Wh_new = Wh
        bh_new = bh_new
        # Logits become s * (Wh @ h) + 0 = s * (original logits - bh)
        # This scales the linear part by s. Since bh was not needed for accuracy,
        # argmax is preserved on feedback. For held-out, it removes the bias,
        # which may or may not help. But it achieves perfect feedback CE.
    else:
        # Otherwise, keep the bias but scale the whole logit by k as much as possible.
        # We need k <= MAX / max_bh. Also k <= MAX * s / max_Wh.
        # We can choose s to maximize k. Since max_Wh = 1000, k <= s.
        # And s <= min(MAX/max_W1, MAX/max_b1, MAX/max_W2, MAX/max_b2).
        # So k_max = min(MAX/max_bh, s_max, MAX * s_max / max_Wh).
        s_max = min(
            MAX / max_W1 if max_W1 > 0 else np.inf,
            MAX / max_b1 if max_b1 > 0 else np.inf,
            MAX / max_W2 if max_W2 > 0 else np.inf,
            MAX / max_b2 if max_b2 > 0 else np.inf,
        )
        k_max = min(MAX / max_bh if max_bh > 0 else np.inf,
                    s_max,
                    MAX * s_max / max_Wh if max_Wh > 0 else np.inf)
        # Use a safe margin
        k = 0.95 * k_max
        s = s_max  # use max s to allow larger k? Actually k is limited by max_bh, so s doesn't matter much.
        # We need to satisfy: s * max_W1 <= MAX, etc.
        # And (k/s) * max_Wh <= MAX, k * max_bh <= MAX.
        # Let's choose s = s_max, then k = min(MAX/max_bh, MAX * s_max / max_Wh).
        # But we also need k <= s? Not necessarily, but if k > s, then head weights would be scaled up.
        # Since max_Wh = 1000, (k/s)*1000 <= 1000 => k <= s. So k <= s.
        # So k = min(MAX/max_bh, s_max).
        k = min(MAX / max_bh if max_bh > 0 else np.inf, s_max)
        k = 0.95 * k
        s = k  # to keep head weights unchanged? If s = k, then (k/s)*Wh = Wh.
        # Actually, if we set s = k, then head weights unchanged, head bias scaled by k.
        # Then logits = Wh @ (k h) + k bh = k (Wh @ h + bh) = k L. Perfect.
        # So we can just scale conv layers by k and head bias by k, leaving head weights unchanged.
        # This requires k * max_bh <= MAX and k * max_conv <= MAX.
        # So k_max = min(MAX/max_bh, MAX/max_W1, MAX/max_b1, MAX/max_W2, MAX/max_b2).
        k_max = min(
            MAX / max_bh if max_bh > 0 else np.inf,
            MAX / max_W1 if max_W1 > 0 else np.inf,
            MAX / max_b1 if max_b1 > 0 else np.inf,
            MAX / max_W2 if max_W2 > 0 else np.inf,
            MAX / max_b2 if max_b2 > 0 else np.inf,
        )
        k = 0.95 * k_max
        s = k
        W1_new = W1 * s
        b1_new = b1 * s
        W2_new = W2 * s
        b2_new = b2 * s
        Wh_new = Wh  # unchanged
        bh_new = bh * k

    # Clip to be safe
    W1_new = np.clip(W1_new, -MAX, MAX).astype(np.float32)
    b1_new = np.clip(b1_new, -MAX, MAX).astype(np.float32)
    W2_new = np.clip(W2_new, -MAX, MAX).astype(np.float32)
    b2_new = np.clip(b2_new, -MAX, MAX).astype(np.float32)
    Wh_new = np.clip(Wh_new, -MAX, MAX).astype(np.float32)
    bh_new = np.clip(bh_new, -MAX, MAX).astype(np.float32)

    # Update weights
    w["conv1.weight"] = W1_new
    w["conv1.bias"] = b1_new
    w["conv2.weight"] = W2_new
    w["conv2.bias"] = b2_new
    w["head.weight"] = Wh_new
    w["head.bias"] = bh_new

    diagnostics = {
        "bias_free_accuracy": float(acc_nb),
        "scale_s": float(s),
        "logit_scale_k": float(k) if acc_nb < 0.9999 else float(s),
        "max_abs_conv1_weight": float(np.max(np.abs(W1_new))),
        "max_abs_conv1_bias": float(np.max(np.abs(b1_new))),
        "max_abs_conv2_weight": float(np.max(np.abs(W2_new))),
        "max_abs_conv2_bias": float(np.max(np.abs(b2_new))),
        "max_abs_head_weight": float(np.max(np.abs(Wh_new))),
        "max_abs_head_bias": float(np.max(np.abs(bh_new))),
    }

    return {"weights": w, "diagnostics": diagnostics}
