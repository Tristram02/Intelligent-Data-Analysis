import numpy as np
from skimage import io, color, transform, data
import matplotlib.pyplot as plt
from typing import Tuple
import math

def img_to_grayscale_array(img, target_size=(128, 128)):
    if img.ndim == 3:
        img = color.rgb2gray(img)  # float 0..1
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        # może być float 0..1 lub inny typ
        img = (img / np.max(img) * 255).astype(np.uint8)
    # resize
    img_resized = transform.resize(img, target_size, anti_aliasing=True, preserve_range=True)
    img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)
    return img_resized

def split_into_patches(img: np.ndarray, n: int) -> np.ndarray:
    N = img.shape[0]
    assert img.shape[0] == img.shape[1], "Obraz musi być kwadratowy"
    assert N % n == 0, "Rozmiar obrazu musi dzielić się przez rozmiar ramki"
    patches = []
    for r in range(0, N, n):
        for c in range(0, N, n):
            block = img[r:r+n, c:c+n].astype(np.float64).flatten()
            patches.append(block)
    return np.array(patches)  # shape (num_patches, n*n)

def normalize_vectors(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms_safe = norms.copy()
    norms_safe[norms_safe == 0] = 1.0
    Xn = X / norms_safe
    return Xn, norms.flatten()

def init_weights_random(num_neurons: int, dim: int, rng: np.random.RandomState):
    W = rng.randn(num_neurons, dim)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    return W

def train_som_wta(X_normalized: np.ndarray, num_neurons: int, epochs: int, eta: float, rng: np.random.RandomState):
    num_samples, dim = X_normalized.shape
    W = init_weights_random(num_neurons, dim, rng)
    touch_count = np.zeros(num_neurons, dtype=int)

    for ep in range(epochs):
        order = rng.permutation(num_samples)
        for idx in order:
            x = X_normalized[idx]
            responses = W.dot(x)
            winner = np.argmax(responses)
            touch_count[winner] += 1
            W[winner] = W[winner] + eta * (x - W[winner])
            W[winner] /= np.linalg.norm(W[winner]) + 1e-12

    return W, touch_count

def remove_dead_neurons(W: np.ndarray, touch_count: np.ndarray):
    alive_mask = touch_count > 0
    if np.sum(alive_mask) == 0:
        alive_mask[0] = True
    W_alive = W[alive_mask]
    return W_alive, alive_mask

def encode_image(patches: np.ndarray, W: np.ndarray, norms_original: np.ndarray, q: int):
    Xn, _ = normalize_vectors(patches)
    responses = Xn.dot(W.T) 
    winners = np.argmax(responses, axis=1)

    T = np.rint(norms_original / q).astype(int)
    return winners.astype(int), T

def decode_image(L: np.ndarray, T: np.ndarray, W: np.ndarray, n: int):
    q = Q_GLOBAL
    num_patches = len(L)
    dim = W.shape[1]

    patches_rec = np.zeros((num_patches, dim), dtype=float)
    for i in range(num_patches):
        k = L[i]
        patches_rec[i] = q * T[i] * W[k]

    patches_side = int(math.sqrt(num_patches))
    N = patches_side * n
    img_rec = np.zeros((N, N), dtype=float)
    idx = 0
    for r_block in range(patches_side):
        for c_block in range(patches_side):
            block = patches_rec[idx].reshape((n, n))
            img_rec[r_block*n:(r_block+1)*n, c_block*n:(c_block+1)*n] = block
            idx += 1

    img_rec_clipped = np.clip(img_rec, 0, 255)
    return img_rec_clipped.astype(np.uint8)

def compute_mse_psnr(img_orig: np.ndarray, img_rec: np.ndarray) -> Tuple[float, float]:

    img_orig = img_orig.astype(np.float64)
    img_rec = img_rec.astype(np.float64)
    mse = np.mean((img_orig - img_rec) ** 2)
    if mse == 0:
        return 0.0, float('inf')
    psnr = 10 * np.log10((255.0 ** 2) / mse)
    return mse, psnr

def compute_cr(N: int, n: int, K: int, W: np.ndarray, L: np.ndarray, T: np.ndarray) -> float:
    num_frames = (N // n) ** 2

    if K <= 1:
        bits_per_index = 1
    else:
        bits_per_index = int(np.floor(np.log2(max(1, K-1))) + 1)

    maxT = int(np.max(T)) if len(T) > 0 else 0
    bits_per_T = int(np.floor(np.log2(max(1, maxT))) + 1) if maxT > 0 else 1

    bits_per_weight = 8
    N_W = bits_per_weight * W.shape[0] * W.shape[1]
    N_L = num_frames * bits_per_index
    N_T = num_frames * bits_per_T
    total_bits_compressed = N_W + N_L + N_T
    orig_bits = 8 * N * N
    CR = orig_bits / total_bits_compressed
    return CR

def experiment_on_image(img: np.ndarray, frame_n=2, Ks=[4,8,16,32,64,128,256], epochs=20, q=10, eta=0.2, rng_seed=42):

    rng = np.random.RandomState(rng_seed)

    N = img.shape[0]
    patches = split_into_patches(img, frame_n)
    patches_normalized, norms = normalize_vectors(patches)
    results = []
    global Q_GLOBAL
    Q_GLOBAL = q
    for K in Ks:

        W_init, touch_count = train_som_wta(patches_normalized, K, epochs=epochs, eta=eta, rng=rng)

        W_alive, alive_mask = remove_dead_neurons(W_init, touch_count)
        K_alive = W_alive.shape[0]

        L, T = encode_image(patches, W_alive, norms, q=q)

        img_rec = decode_image(L, T, W_alive, frame_n)
        mse, psnr = compute_mse_psnr(img, img_rec)
        cr = compute_cr(N, frame_n, K_alive, W_alive, L, T)
        results.append({
            'K_initial': K,
            'K_alive': K_alive,
            'psnr': psnr,
            'mse': mse,
            'cr': cr,
            'img_rec': img_rec,
            'W_alive': W_alive,
            'L': L, 'T': T
        })
        print(f"K={K:3d} -> alive={K_alive:3d}, PSNR={psnr:.2f} dB, CR={cr:.3f}")
    # wykres CR vs PSNR
    psnrs = [r['psnr'] for r in results]
    crs = [r['cr'] for r in results]
    plt.figure(figsize=(6,4))
    plt.plot(psnrs, crs, marker='o')
    plt.ylabel('Compression Ratio (CR)')
    plt.xlabel('PSNR (dB)')
    plt.title('CR vs PSNR for SOM compression (frame 2x2)')
    plt.grid(True)
    plt.savefig("plots/plot_frame_4.png")
    return results

if __name__ == "__main__":
    img_sk = io.imread("zelda.png")
    img = img_to_grayscale_array(img_sk, target_size=(128,128))
    epochs = 20
    q = 10
    eta = 0.2
    frame_n = 4

    print("epochs: ", epochs)
    print("q: ", q)
    print("eta: ", eta)
    print("frame_n: ", frame_n)

    results = experiment_on_image(img, frame_n=frame_n, 
                                  Ks=[4,8,16,32,64,128,256], 
                                  epochs=epochs, q=q, eta=eta, rng_seed=0)

    first = results[-1]
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title('original'); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(first['img_rec'], cmap='gray'); plt.title(f"reconstructed K={first['K_initial']} (alive={first['K_alive']})"); plt.axis('off')
    plt.savefig("plots/image_frame_4.png")
