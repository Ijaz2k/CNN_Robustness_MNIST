# Analisis Ketahanan CNN terhadap Noise, Blur, dan Rotasi pada Citra Angka Tulisan Tangan

CNN untuk pengenalan angka tulisan tangan (MNIST) beserta analisis ketahanannya terhadap gangguan citra: **noise**, **blur**, dan **rotasi**.

| | |
| **Nama** | Rachmat Mukti Ilyas |
| **NIM** | 101022330200 |
| **Mata Kuliah** | Desain SoC dan FPGA |

---

## Daftar Isi

1. [Struktur Repositori](#struktur-repositori)
2. [Latar Belakang dan Tujuan](#latar-belakang-dan-tujuan)
3. [Dataset](#dataset)
4. [Arsitektur CNN](#arsitektur-cnn)
5. [Metode Pengujian Ketahanan](#metode-pengujian-ketahanan)
6. [Hasil](#hasil)
7. [Analisis](#analisis)
8. [Kesimpulan](#kesimpulan)
9. [Cara Menjalankan](#cara-menjalankan)
10. [Status Pengerjaan](#status-pengerjaan)
11. [Referensi](#referensi)

---

## Struktur Repositori

| Folder / File | Isi |
|---|---|
| `dataset/` | Informasi dataset, contoh gambar gangguan (`samples/`), dan dataset hasil generate (`perturbed/`, tidak di-upload) |
| `python/` | Algoritma CNN: training, pembuatan gangguan, dan pengujian ketahanan |
| `verilog/` | Implementasi hardware (tidak digunakan pada tugas ini) |
| `report/` | Laporan hasil implementasi |
| `README.md` | Dokumen ini |

```
CNN_Robustness_MNIST/
├── dataset/
│   ├── README.md
│   ├── samples/
│   │   └── contoh_gangguan.png
│   └── perturbed/                 # dibuat oleh generate_datasets.py (di-ignore git)
├── python/
│   ├── perturbations.py           # fungsi noise, blur, rotasi
│   ├── generate_datasets.py       # membuat dataset uji dengan gangguan
│   ├── train_cnn_mnist.ipynb      # training CNN
│   ├── robustness_test.ipynb      # pengujian ketahanan
│   ├── model_cnn_mnist.h5         # bobot model hasil training
│   └── requirements.txt
├── verilog/
│   └── README.md
├── report/
│   ├── laporan_UTS.pdf
│   └── figures/
├── .gitignore
└── README.md
```

---

## Latar Belakang dan Tujuan

**Latar belakang:** TODO (1-2 paragraf: mengapa ketahanan model terhadap gangguan citra penting, misalnya kualitas kamera, goresan, tulisan miring).

**Rumusan masalah:**
- TODO (contoh: Seberapa besar penurunan akurasi CNN saat citra diberi noise, blur, atau rotasi?)

**Tujuan:**
- Melatih CNN untuk pengenalan angka tulisan tangan pada MNIST.
- Menguji akurasi CNN pada citra yang diberi noise, blur, dan rotasi dengan beberapa tingkat gangguan.
- TODO (tambahkan bila ada, misalnya membandingkan model dengan dan tanpa augmentasi)

---

## Dataset

**MNIST Handwritten Digits**: 10 kelas (angka 0-9), citra grayscale 28 × 28 piksel, label *one-hot encoded*.

| Bagian | Jumlah |
|---|---|
| Training | 55.000 |
| Validation | 5.000 |
| Test | 10.000 |

Sumber data: dimuat otomatis dengan `tf.keras.datasets.mnist.load_data()`.

### Dataset dengan gangguan

Gangguan hanya diterapkan pada **data test**. Data latih tetap bersih. Dataset dibuat dengan `python/generate_datasets.py`.

| Gangguan | Parameter | Level |
|---|---|---|
| Gaussian noise | σ | 0, 0.1, 0.2, 0.3, 0.5 |
| Gaussian blur | σ | 0, 0.5, 1.0, 1.5, 2.0 |
| Rotasi | sudut (derajat) | −45, −30, −15, 0, 15, 30, 45 |

### Contoh citra

TODO: masukkan gambar dari `dataset/samples/`

![Contoh gangguan](dataset/samples/contoh_gangguan.png)

---

## Arsitektur CNN

Arsitektur mengikuti referensi dari slide dosen.

| Layer | Output Shape | Parameter |
|---|---|---|
| Conv2D (16, 3×3, ReLU, same) | (None, 28, 28, 16) | 160 |
| MaxPooling2D (2×2) | (None, 14, 14, 16) | 0 |
| Conv2D (32, 3×3, ReLU, same) | (None, 14, 14, 32) | 4.640 |
| MaxPooling2D (2×2) | (None, 7, 7, 32) | 0 |
| Flatten | (None, 1568) | 0 |
| Dense (64, ReLU) | (None, 64) | 100.416 |
| Dense (10, Softmax) | (None, 10) | 650 |
| **Total** | | **105.866** |

**Konfigurasi training:**

| Parameter | Nilai |
|---|---|
| Optimizer | Adam |
| Loss | categorical_crossentropy |
| Epochs | 10 |
| Batch size | 100 |

TODO: cocokkan tabel dengan hasil `model.summary()` yang sebenarnya.

---

## Metode Pengujian Ketahanan

TODO: jelaskan alur pengujian.

1. Latih CNN pada data MNIST bersih.
2. Muat dataset uji bergangguan dari `dataset/perturbed/`.
3. Hitung akurasi model pada tiap jenis dan tiap level gangguan.
4. Bandingkan dengan akurasi baseline (tanpa gangguan).

TODO (opsional): eksperimen kedua dengan model yang dilatih memakai augmentasi.

---

## Hasil

### Baseline (tanpa gangguan)

| Metrik | Nilai |
|---|---|
| Test accuracy | TODO |
| Test loss | TODO |

TODO: masukkan grafik training vs validation (`report/figures/`).

### Akurasi terhadap Gaussian noise

| σ | 0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| Akurasi | TODO | TODO | TODO | TODO | TODO |

### Akurasi terhadap Gaussian blur

| σ | 0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| Akurasi | TODO | TODO | TODO | TODO | TODO |

### Akurasi terhadap rotasi

| Sudut | −45° | −30° | −15° | 0° | 15° | 30° | 45° |
|---|---|---|---|---|---|---|---|
| Akurasi | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### Grafik dan visualisasi

TODO: grafik akurasi vs level gangguan, confusion matrix, contoh prediksi benar dan salah.

---

## Analisis

TODO:
- Jenis gangguan mana yang paling menurunkan akurasi, dan mengapa.
- Digit mana yang paling rentan (berdasarkan confusion matrix).
- Pada level berapa akurasi turun secara signifikan.
- Perbandingan dengan model beraugmentasi (jika dikerjakan).
- Keterbatasan penelitian.

---

## Kesimpulan

TODO

---

## Cara Menjalankan

**1. Clone repositori dan buat environment**

```bash
git clone https://github.com/USERNAME/CNN_Robustness_MNIST.git
cd CNN_Robustness_MNIST
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r python/requirements.txt
```

**2. Buat dataset bergangguan**

```bash
python python/generate_datasets.py
```

**3. Latih model**

Jalankan `python/train_cnn_mnist.ipynb`.

**4. Uji ketahanan**

Jalankan `python/robustness_test.ipynb`.

> Catatan: TensorFlow hanya mendukung versi Python tertentu. Gunakan Python 3.11 atau 3.12, dan cek tensorflow.org/install/pip untuk versi terbaru yang didukung.

---

## Status Pengerjaan

- [x] Struktur repositori
- [x] Fungsi gangguan (`perturbations.py`)
- [x] Pembuatan dataset bergangguan
- [ ] Training CNN
- [ ] Pengujian ketahanan
- [ ] Analisis hasil
- [ ] Laporan akhir

---

## Referensi

- Slide kuliah: Aplikasi CNN pada Pengenalan Angka, Informatika ITS.
- LeCun, Y., Cortes, C., Burges, C. *The MNIST database of handwritten digits.*
- Repositori acuan struktur: https://github.com/rizki-ap/CNN
- TODO: tambahkan referensi lain
