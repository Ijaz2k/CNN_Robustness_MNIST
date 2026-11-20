from pathlib import Path
import numpy as np
from tensorflow.keras.datasets import mnist
from perturbations import add_gaussian_noise, add_blur, add_rotation

# Dataset Generator
# Lokasi output: <root repo>/dataset/perturbed/
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'dataset' / 'perturbed'
OUT.mkdir(parents=True, exist_ok=True)

# Level gangguan
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
BLUR_LEVELS  = [0.0, 0.5, 1.0, 1.5, 2.0]
ROT_LEVELS   = [-45, -30, -15, 0, 15, 30, 45]

# Load data test MNIST dan normalisasi ke (10000, 28, 28, 1)
(_, _), (X_test, y_test) = mnist.load_data()
X = (X_test.astype('float32') / 255.0)[..., None]

def to_uint8(a):
    return np.round(a * 255).astype('uint8')   # hemat ukuran file

def generate(name, levels, fn):
    stacked = np.stack([to_uint8(fn(X, lv)) for lv in levels])
    np.savez_compressed(OUT / f'{name}.npz',
                        X=stacked,                 # (n_level, 10000, 28, 28, 1)
                        y=y_test,                  # label asli 0-9
                        levels=np.array(levels))
    print(f'{name}: tersimpan, shape {stacked.shape}')

generate('noise',    NOISE_LEVELS, add_gaussian_noise)
generate('blur',     BLUR_LEVELS,  add_blur)

generate('rotation', ROT_LEVELS,   add_rotation)
print('Selesai ->', OUT)