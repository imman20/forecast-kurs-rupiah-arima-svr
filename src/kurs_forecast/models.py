"""Model ARIMA dan SVR.

ARIMA mengikuti alur skripsi: Box-Cox, differencing 2, kandidat model dari
ACF/PACF, seleksi berdasarkan signifikansi parameter, white noise, normalitas,
lalu RMSE/MAPE training. SVR memakai input lag-1 (hasil PACF skripsi) dengan
grid search pada C, epsilon, gamma seperti Tabel 4.7 skripsi, tetapi memakai
validasi silang runtun waktu (TimeSeriesSplit) agar pemilihan hyperparameter
tidak melihat data testing sama sekali.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA

from .diagnostics import ljung_box, normality_tests
from .metrics import all_metrics
from .transform import boxcox_inverse, boxcox_transform, destandardize, estimate_lambda, standardize

# Kandidat model persis seperti yang diidentifikasi skripsi dari plot ACF/PACF differencing 2.
KANDIDAT_ARIMA = [(0, 2, 1), (1, 2, 0), (2, 2, 0), (1, 2, 1), (2, 2, 1)]


@dataclass
class ArimaResult:
    order: tuple[int, int, int]
    lam: float
    params: dict
    pvalues: dict
    aic: float
    semua_signifikan: bool
    ljung_box: pd.DataFrame
    normalitas: dict
    metrik_train: dict
    fitted_train: np.ndarray
    model: object = field(repr=False)
    z_mean: float = 0.0
    z_std: float = 1.0


def fit_arima(train: pd.Series, order: tuple[int, int, int], lam: float | None = None) -> ArimaResult:
    """Fit satu model ARIMA pada data yang sudah ditransformasi Box-Cox.

    Fitted values dikembalikan ke skala kurs asli supaya metrik training
    sebanding dengan metrik testing.
    """
    if lam is None:
        lam = estimate_lambda(train)
    z, z_mean, z_std = standardize(boxcox_transform(train.values, lam))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(z, order=order, trend="n").fit()
    # Hanya parameter AR/MA yang diuji signifikansinya (sigma2 bukan parameter struktural).
    # Input berupa ndarray sehingga params tidak berlabel; dipetakan lewat param_names.
    semua_params = dict(zip(model.param_names, np.asarray(model.params, dtype=float)))
    semua_pvalues = dict(zip(model.param_names, np.asarray(model.pvalues, dtype=float)))
    names = [n for n in model.param_names if n != "sigma2"]
    params = {n: semua_params[n] for n in names}
    pvalues = {n: semua_pvalues[n] for n in names}
    d = order[1]
    fitted_z = model.fittedvalues[d:]  # d nilai pertama belum punya prediksi yang berarti
    fitted = boxcox_inverse(destandardize(fitted_z, z_mean, z_std), lam)
    resid = model.resid[d:]
    return ArimaResult(
        order=order,
        lam=lam,
        params=params,
        pvalues=pvalues,
        aic=float(model.aic),
        semua_signifikan=all(p < 0.05 for p in pvalues.values()),
        ljung_box=ljung_box(resid),
        normalitas=normality_tests(resid),
        metrik_train=all_metrics(train.values[d:], fitted),
        fitted_train=fitted,
        model=model,
        z_mean=z_mean,
        z_std=z_std,
    )


def forecast_arima(result: ArimaResult, steps: int) -> np.ndarray:
    """Peramalan h-langkah ke depan, dikembalikan ke skala kurs asli."""
    fc_z = np.asarray(result.model.forecast(steps=steps))
    return boxcox_inverse(destandardize(fc_z, result.z_mean, result.z_std), result.lam)


def pilih_arima_terbaik(train: pd.Series, kandidat=KANDIDAT_ARIMA) -> tuple[ArimaResult, pd.DataFrame]:
    """Menjalankan seluruh kandidat lalu memilih dengan kriteria skripsi.

    Urutan kriteria: (1) semua parameter signifikan, (2) residual white noise
    pada semua lag yang diuji, (3) RMSE training terkecil. Prinsip parsimoni
    diterapkan secara implisit karena model dengan parameter tidak signifikan
    gugur lebih dulu.
    """
    lam = estimate_lambda(train)
    hasil = [fit_arima(train, order, lam) for order in kandidat]
    tabel = pd.DataFrame(
        {
            "model": [f"ARIMA{r.order}" for r in hasil],
            "parameter": [", ".join(f"{k}={v:.4f}" for k, v in r.params.items()) for r in hasil],
            "semua_signifikan": [r.semua_signifikan for r in hasil],
            "white_noise": [bool(r.ljung_box["white_noise"].all()) for r in hasil],
            "normal_ks": [r.normalitas["normal_ks"] for r in hasil],
            "AIC": [r.aic for r in hasil],
            "RMSE_train": [r.metrik_train["RMSE"] for r in hasil],
            "MAPE_train": [r.metrik_train["MAPE"] for r in hasil],
        }
    )
    layak = [r for r in hasil if r.semua_signifikan and bool(r.ljung_box["white_noise"].all())]
    if not layak:  # fallback: jangan gagal total, pilih AIC terkecil dan beri tahu lewat tabel
        layak = hasil
    terbaik = min(layak, key=lambda r: r.metrik_train["RMSE"])
    tabel["terpilih"] = tabel["model"] == f"ARIMA{terbaik.order}"
    return terbaik, tabel


# ----------------------------------------------------------------------------
# SVR
# ----------------------------------------------------------------------------

def buat_fitur_lag(serie: pd.Series, n_lag: int = 1) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Membentuk matriks X (lag) dan target y. Mengikuti skripsi: hanya lag-1."""
    df = pd.DataFrame({"y": serie.values}, index=serie.index)
    for k in range(1, n_lag + 1):
        df[f"lag{k}"] = df["y"].shift(k)
    df = df.dropna()
    X = df[[f"lag{k}" for k in range(1, n_lag + 1)]].values
    return X, df["y"].values, df.index


# Grid persis seperti Tabel 4.7 skripsi: C = 2^0..2^9, gamma = 2^-3..2^3, epsilon = 0..1 langkah 0,1.
GRID_SVR = {
    "svr__C": [2.0**k for k in range(0, 10)],
    "svr__gamma": [2.0**k for k in range(-3, 4)],
    "svr__epsilon": [round(0.1 * k, 1) for k in range(0, 11)],
}


@dataclass
class SvrResult:
    kernel: str
    best_params: dict
    cv_rmse: float
    metrik_train: dict
    fitted_train: np.ndarray
    pipeline: Pipeline = field(repr=False)
    n_lag: int = 1


def fit_svr(train: pd.Series, kernel: str = "rbf", n_lag: int = 1, n_splits: int = 5, grid: dict | None = None) -> SvrResult:
    """Grid search SVR dengan TimeSeriesSplit.

    Fitur distandardisasi (StandardScaler) karena kernel RBF sensitif terhadap
    skala; target juga distandardisasi secara manual supaya epsilon pada grid
    0..1 bermakna (epsilon 0,1 pada skala rupiah 15.000 praktis nol).
    """
    X, y, _ = buat_fitur_lag(train, n_lag)
    y_mean, y_std = y.mean(), y.std()
    y_s = (y - y_mean) / y_std
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel=kernel, degree=2))])
    gs = GridSearchCV(
        pipe,
        grid or GRID_SVR,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gs.fit(X, y_s)
    fitted = gs.predict(X) * y_std + y_mean
    best = gs.best_estimator_
    # Simpan statistik target di dalam pipeline agar prediksi nanti bisa di-inverse.
    best.y_mean_, best.y_std_ = y_mean, y_std
    return SvrResult(
        kernel=kernel,
        best_params={k.replace("svr__", ""): v for k, v in gs.best_params_.items()},
        cv_rmse=float(-gs.best_score_ * y_std),
        metrik_train=all_metrics(y, fitted),
        fitted_train=fitted,
        pipeline=best,
        n_lag=n_lag,
    )


def predict_svr(result: SvrResult, train: pd.Series, test: pd.Series, recursive: bool = False) -> np.ndarray:
    """Prediksi data testing.

    recursive=False (default, seperti skripsi): memakai nilai aktual minggu
    sebelumnya sebagai input (peramalan satu langkah ke depan).
    recursive=True: memakai prediksi sendiri sebagai input (peramalan multi
    langkah), lebih jujur untuk horizon panjang tetapi galatnya menumpuk.
    """
    pipe = result.pipeline
    n_lag = result.n_lag
    history = list(train.values[-n_lag:])
    preds = []
    for t in range(len(test)):
        x = np.array(history[-n_lag:][::-1]).reshape(1, -1)
        p = float(pipe.predict(x)[0] * pipe.y_std_ + pipe.y_mean_)
        preds.append(p)
        history.append(p if recursive else float(test.values[t]))
    return np.array(preds)


def forecast_svr_future(result: SvrResult, serie: pd.Series, steps: int) -> np.ndarray:
    """Peramalan rekursif ke depan setelah akhir data (dipakai aplikasi Streamlit)."""
    pipe = result.pipeline
    n_lag = result.n_lag
    history = list(serie.values[-n_lag:])
    preds = []
    for _ in range(steps):
        x = np.array(history[-n_lag:][::-1]).reshape(1, -1)
        p = float(pipe.predict(x)[0] * pipe.y_std_ + pipe.y_mean_)
        preds.append(p)
        history.append(p)
    return np.array(preds)
