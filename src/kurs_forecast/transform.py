"""Transformasi Box-Cox dan differencing.

Skripsi menemukan lambda Box-Cox = -0,9999 (dibulatkan -1) pada data training,
sehingga transformasi yang dipakai adalah Z* = 1/Z. Modul ini mengestimasi
lambda sendiri (bukan hard-code) supaya alur tetap valid bila data diganti.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def estimate_lambda(x: pd.Series | np.ndarray, lower: float = -1.0, upper: float = 1.0) -> float:
    """Mengestimasi lambda Box-Cox dengan maximum likelihood pada rentang [-1, 1].

    Rentang dibatasi persis seperti `forecast::BoxCox.lambda` di R yang dipakai
    skripsi (default lower=-1, upper=1). Tanpa pembatasan, MLE pada data kurs
    ini lari ke nilai ekstrem (sekitar -7) yang tidak bermakna dan membuat
    deret hasil transformasi praktis konstan.
    """
    from scipy.optimize import minimize_scalar

    arr = np.asarray(x, dtype=float)
    hasil = minimize_scalar(lambda lam: -stats.boxcox_llf(lam, arr), bounds=(lower, upper), method="bounded")
    return float(hasil.x)


def standardize(z: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Menstandardisasi deret hasil Box-Cox agar skalanya wajar untuk optimasi ARIMA.

    Dengan lambda mendekati -1, nilai Box-Cox berorde 1e-5 (sigma^2 sekitar 1e-13
    pada skripsi). Optimizer numerik bekerja jauh lebih stabil pada skala ~1, dan
    karena standardisasi hanya transformasi affine, model ARIMA tanpa konstanta
    tidak berubah maknanya. Mengembalikan (z_terstandar, mean, std).
    """
    mean, std = float(np.mean(z)), float(np.std(z))
    if std == 0:
        raise ValueError("Deret konstan, tidak bisa distandardisasi.")
    return (z - mean) / std, mean, std


def destandardize(z_s: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.asarray(z_s, dtype=float) * std + mean


def boxcox_transform(x: pd.Series | np.ndarray, lam: float) -> np.ndarray:
    """Menerapkan Box-Cox: (x^lam - 1)/lam, atau log(x) saat lam = 0.

    Catatan: skripsi memakai bentuk sederhana x^lam (tanpa pengurangan 1 dan
    pembagian lam). Keduanya ekuivalen untuk pemodelan karena hanya berbeda
    oleh transformasi linear; di sini dipakai bentuk baku scipy agar
    inversnya konsisten.
    """
    x = np.asarray(x, dtype=float)
    if abs(lam) < 1e-8:
        return np.log(x)
    return (np.power(x, lam) - 1.0) / lam


def boxcox_inverse(y: pd.Series | np.ndarray, lam: float) -> np.ndarray:
    """Membalik transformasi Box-Cox ke skala kurs asli."""
    y = np.asarray(y, dtype=float)
    if abs(lam) < 1e-8:
        return np.exp(y)
    return np.power(lam * y + 1.0, 1.0 / lam)


def difference(x: pd.Series | np.ndarray, d: int = 1) -> np.ndarray:
    """Differencing sebanyak d kali; mengembalikan array tanpa nilai NaN."""
    arr = np.asarray(x, dtype=float)
    for _ in range(d):
        arr = np.diff(arr)
    return arr
