"""Fungsi visualisasi (matplotlib) yang dipakai notebook, pipeline, dan laporan."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend tanpa layar supaya aman di CI dan server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

WARNA = {"aktual": "#1f2937", "train": "#2563eb", "test": "#dc2626", "arima": "#16a34a", "svr": "#f59e0b"}


def _simpan(fig, path: Path | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_series(train: pd.Series, test: pd.Series, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train.index, train.values, color=WARNA["train"], label=f"Training ({len(train)} minggu)")
    ax.plot(test.index, test.values, color=WARNA["test"], label=f"Testing ({len(test)} minggu)")
    ax.axvline(test.index[0], color="grey", ls="--", lw=1)
    ax.set_title("Kurs IDR/USD mingguan, Okt 2021 sampai Okt 2023")
    ax.set_ylabel("Rupiah per 1 USD")
    ax.legend()
    ax.grid(alpha=0.3)
    return _simpan(fig, path)


def plot_acf_pacf(x, judul: str, lags: int = 20, path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    plot_acf(np.asarray(x), lags=lags, ax=axes[0], title=f"ACF: {judul}")
    plot_pacf(np.asarray(x), lags=lags, ax=axes[1], title=f"PACF: {judul}", method="ywm")
    fig.tight_layout()
    return _simpan(fig, path)


def plot_perbandingan_test(test: pd.Series, pred_arima, pred_svr, path: Path | None = None, label_arima="ARIMA", label_svr="SVR RBF"):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(test.index, test.values, color=WARNA["aktual"], marker="o", ms=4, label="Aktual (testing)")
    ax.plot(test.index, pred_arima, color=WARNA["arima"], marker="s", ms=3, ls="--", label=label_arima)
    ax.plot(test.index, pred_svr, color=WARNA["svr"], marker="^", ms=3, ls="--", label=label_svr)
    ax.set_title("Prediksi data testing: ARIMA vs SVR")
    ax.set_ylabel("Rupiah per 1 USD")
    ax.legend()
    ax.grid(alpha=0.3)
    return _simpan(fig, path)


def plot_fit_train(train: pd.Series, fitted, label: str, offset: int = 0, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train.index, train.values, color=WARNA["aktual"], label="Aktual (training)")
    ax.plot(train.index[offset:], fitted, color=WARNA["svr"] if "SVR" in label else WARNA["arima"], ls="--", label=label)
    ax.set_title(f"Kecocokan model pada data training: {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    return _simpan(fig, path)


def plot_residual(resid, judul: str, path: Path | None = None):
    from scipy import stats

    resid = np.asarray(resid, float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(resid, color=WARNA["train"])
    axes[0].axhline(0, color="red", ls="--")
    axes[0].set_title("Residual terhadap waktu")
    axes[1].hist(resid, bins=15, density=True, color="#cbd5e1", edgecolor="white")
    xs = np.linspace(resid.min(), resid.max(), 200)
    axes[1].plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std(ddof=1)), color="red")
    axes[1].set_title("Histogram residual")
    stats.probplot(resid, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q plot")
    fig.suptitle(judul)
    fig.tight_layout()
    return _simpan(fig, path)


def plot_var(returns: pd.Series, var_h: float, var_p: float, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(returns * 100, bins=20, color="#cbd5e1", edgecolor="white", density=True)
    ax.axvline(var_h * 100, color=WARNA["test"], ls="--", label=f"VaR historis 95% = {var_h*100:.2f}%")
    ax.axvline(var_p * 100, color=WARNA["arima"], ls=":", label=f"VaR parametrik 95% = {var_p*100:.2f}%")
    ax.set_title("Distribusi log return mingguan kurs IDR/USD dan VaR")
    ax.set_xlabel("Log return mingguan (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _simpan(fig, path)
