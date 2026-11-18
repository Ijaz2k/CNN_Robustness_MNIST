import numpy as np
from scipy.ndimage import gaussian_filter, rotate

# Semua fungsi: input X berbentuk (N, 28, 28, 1), nilai 0-1 (float)

def add_gaussian_noise(X, sigma, seed=42):
    """Tambah noise Gaussian. seed tetap agar hasil bisa direproduksi."""
    rng = np.random.default_rng(seed)
    noisy = X + rng.normal(0.0, sigma, X.shape)
    return np.clip(noisy, 0.0, 1.0).astype('float32')

def add_blur(X, sigma):
    """Gaussian blur. sigma=0 berarti tanpa blur."""
    if sigma == 0:
        return X.copy()
    out = [gaussian_filter(x, sigma=(sigma, sigma, 0)) for x in X]
    return np.stack(out).astype('float32')

def add_rotation(X, angle):
    """Rotasi dalam derajat, ukuran gambar tetap 28x28, latar diisi 0 (hitam)."""
    if angle == 0:
        return X.copy()
    out = [rotate(x, angle, axes=(1, 0), reshape=False,
                  order=1, mode='constant', cval=0.0) for x in X]
    return np.stack(out).astype('float32')