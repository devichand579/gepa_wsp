def edit(w, f):
    import numpy as np

    # ---------- helper: conv2d with padding=1, stride=1 ----------
    def conv2d(x, weight, bias, padding=1):
        N, C, H, W = x.shape
        out_C, _, kH, kW = weight.shape
        H_out = H + 2 * padding - kH + 1
        W_out = W + 2 * padding - kW + 1
        x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        out = np.zeros((N, out_C, H_out, W_out), dtype=x.dtype)
        for i in range(kH):
            for j in range(kW):
                x_slice = x_pad[:, :, i:i + H_out, j:j + W_out]
                w_slice = weight[:, :, i, j]
                out += np.einsum('nchw,oc->nohw', x_slice, w_slice)
        out += bias.reshape(1, -1, 1, 1)
        return out

    # ---------- helper: maxpool2d kernel=2 stride=2 ----------
    def maxpool2d(x, kernel_size=2, stride=2):
        N, C, H, W = x.shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1
        x_reshaped = x.reshape(N, C, H_out, kernel_size, W_out, kernel_size)
        return x_reshaped.max(axis=(3, 5))

    # ---------- load data ----------
    x = f["x"]          # (2000, 1, 28, 28)
    y = f["y"]          # (2000,)
    N = x.shape[0]

    # ---------- 1. PCA filters for conv1 ----------
    # extract all 3x3 patches from padded images
    x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
    patches = np.lib.stride_tricks.sliding_window_view(x_pad, (3, 3), axis=(2, 3))  # (N,1,28,28,3,3)
    patches = patches.reshape(-1, 9)          # (N*784, 9)
    mean_patch = patches.mean(axis=0)
    centered = patches - mean_patch
    cov = centered.T @ centered / centered.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    # take top 16 eigenvectors
    filters1 = eigvecs[:, -16:].T             # (16, 9)
    filters1 = filters1.reshape(16, 1, 3, 3)
    bias1 = - (filters1.reshape(16, 9) @ mean_patch)

    # ---------- 2. compute first-layer activations ----------
    h1 = conv2d(x, filters1, bias1, padding=1)
    h1 = np.maximum(h1, 0.0)
    h1 = maxpool2d(h1, 2, 2)                  # (N,16,14,14)

    # ---------- 3. PCA filters for conv2 ----------
    h1_pad = np.pad(h1, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
    patches2 = np.lib.stride_tricks.sliding_window_view(h1_pad, (3, 3), axis=(2, 3))  # (N,16,14,14,3,3)
    patches2 = patches2.transpose(0, 2, 3, 1, 4, 5)   # (N,14,14,16,3,3)
    patches2 = patches2.reshape(-1, 16 * 3 * 3)       # (N*196, 144)
    mean_patch2 = patches2.mean(axis=0)
    centered2 = patches2 - mean_patch2
    cov2 = centered2.T @ centered2 / centered2.shape[0]
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    filters2 = eigvecs2[:, -32:].T                    # (32, 144)
    filters2 = filters2.reshape(32, 16, 3, 3)
    bias2 = - (filters2.reshape(32, 144) @ mean_patch2)

    # ---------- 4. compute final features ----------
    h2 = conv2d(h1, filters2, bias2, padding=1)
    h2 = np.maximum(h2, 0.0)
    h2 = maxpool2d(h2, 2, 2)                  # (N,32,7,7)
    F = h2.reshape(N, -1)                     # (N, 1568)

    # ---------- 5. closed-form ridge regression for head ----------
    Y = np.eye(10)[y]                         # (N,10)
    F_aug = np.hstack([F, np.ones((N, 1))])   # (N,1569)
    lam = 1e-2
    A = F_aug.T @ F_aug + lam * np.eye(F_aug.shape[1])
    B = F_aug.T @ Y
    W_aug = np.linalg.solve(A, B)             # (1569,10)
    head_weight = W_aug[:1568, :].T           # (10,1568)
    head_bias = W_aug[1568, :]                # (10,)

    # ---------- 6. write back into w ----------
    w["conv1.weight"] = filters1.astype(np.float32)
    w["conv1.bias"]   = bias1.astype(np.float32)
    w["conv2.weight"] = filters2.astype(np.float32)
    w["conv2.bias"]   = bias2.astype(np.float32)
    w["head.weight"]  = head_weight.astype(np.float32)
    w["head.bias"]    = head_bias.astype(np.float32)

    return w
