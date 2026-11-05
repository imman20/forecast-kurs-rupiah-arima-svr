"""Uji diagnostik runtun waktu.

Mencakup tiga kelompok uji yang dipakai skripsi (stasioneritas, white noise,
normalitas residual) ditambah uji Augmented Dickey-Fuller yang lebih baku
daripada sekadar membaca plot ACF/PACF secara visual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(x: np.ndarray | pd.Series) -> dict:
    """Augmented Dickey-Fuller. H0: ada akar unit (tidak stasioner)."""
    stat, pvalue, lags, nobs, crit, _ = adfuller(np.asarray(x, dtype=float), autolag="AIC")
    return {
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "lags": int(lags),
        "nobs": int(nobs),
        "stasioner": bool(pvalue < 0.05),
    }


def kpss_test(x: np.ndarray | pd.Series) -> dict:
    """KPSS. H0: stasioner. Dipakai sebagai pelengkap ADF karena hipotesisnya berkebalikan."""
    stat, pvalue, lags, _ = kpss(np.asarray(x, dtype=float), regression="c", nlags="auto")
    return {"statistic": float(stat), "pvalue": float(pvalue), "lags": int(lags), "stasioner": bool(pvalue >= 0.05)}


def ljung_box(residuals: np.ndarray | pd.Series, lags: tuple[int, ...] = (12, 24, 36, 48)) -> pd.DataFrame:
    """Uji Ljung-Box pada beberapa lag sekaligus, mengikuti Tabel 4.2 skripsi.

    H0: residual white noise (tidak ada autokorelasi). Lag yang melebihi panjang
    data dibuang supaya fungsi tetap aman untuk data pendek.
    """
    res = np.asarray(residuals, dtype=float)
    valid = [lag for lag in lags if lag < len(res)]
    out = acorr_ljungbox(res, lags=valid, return_df=True)
    out.index.name = "lag"
    out["white_noise"] = out["lb_pvalue"] >= 0.05
    return out


def normality_tests(residuals: np.ndarray | pd.Series) -> dict:
    """Kolmogorov-Smirnov (seperti skripsi) dan Shapiro-Wilk (lebih kuat untuk n kecil).

    KS dihitung terhadap normal dengan mean dan sd residual (varian Lilliefors
    sederhana), bukan normal standar, agar hasilnya bermakna.
    """
    res = np.asarray(residuals, dtype=float)
    mu, sd = res.mean(), res.std(ddof=1)
    ks_stat, ks_p = stats.kstest(res, stats.norm(loc=mu, scale=sd).cdf)
    sw_stat, sw_p = stats.shapiro(res)
    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "shapiro_statistic": float(sw_stat),
        "shapiro_pvalue": float(sw_p),
        "normal_ks": bool(ks_p >= 0.05),
        "normal_shapiro": bool(sw_p >= 0.05),
    }


def terasvirta_proxy(x: np.ndarray | pd.Series, lag: int = 1) -> dict:
    """Uji nonlinieritas ala Terasvirta (versi regresi Volterra sederhana).

    Skripsi memakai `tseries::terasvirta.test` di R. Di Python tidak ada padanan
    langsung, jadi dipakai pendekatan yang sama secara prinsip: regresi y_t pada
    y_{t-1}, lalu uji apakah suku kuadrat dan kubik dari prediksi linier menambah
    penjelasan secara signifikan (uji F). H0: hubungan linier.
    """
    import statsmodels.api as sm

    arr = np.asarray(x, dtype=float)
    arr = (arr - arr.mean()) / arr.std()  # uji ini invarian skala; standardisasi mencegah matriks rank-deficient
    y = arr[lag:]
    X = np.column_stack([arr[lag - k - 1 : len(arr) - k - 1] for k in range(lag)])
    X1 = sm.add_constant(X)
    lin = sm.OLS(y, X1).fit()
    fitted = lin.fittedvalues
    X2 = np.column_stack([X1, fitted**2, fitted**3])
    nonlin = sm.OLS(y, X2).fit()
    f_stat, f_p, _ = nonlin.compare_f_test(lin)
    return {"f_statistic": float(f_stat), "pvalue": float(f_p), "nonlinier": bool(f_p < 0.05)}
