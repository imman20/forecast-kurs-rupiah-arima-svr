"""Metrik akurasi peramalan dan uji Diebold-Mariano."""

from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(actual, predicted) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual, predicted) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(a - p)))


def mape(actual, predicted) -> float:
    """MAPE dalam persen. Aman karena kurs selalu positif (tidak ada pembagian nol)."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs((a - p) / a)) * 100.0)


def all_metrics(actual, predicted) -> dict:
    return {"RMSE": rmse(actual, predicted), "MAE": mae(actual, predicted), "MAPE": mape(actual, predicted)}


def diebold_mariano(actual, pred_1, pred_2, h: int = 1, power: int = 2) -> dict:
    """Uji Diebold-Mariano (1995) dengan koreksi Harvey-Leybourne-Newbold (1997).

    Membandingkan apakah selisih galat kuadrat dua model berbeda nyata secara
    statistik, sesuatu yang tidak bisa dijawab hanya dengan membandingkan RMSE.
    H0: kedua model mempunyai akurasi yang sama.
    Statistik negatif berarti model 1 lebih akurat daripada model 2.
    """
    a = np.asarray(actual, float)
    e1 = a - np.asarray(pred_1, float)
    e2 = a - np.asarray(pred_2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_mean = d.mean()

    # Varian jangka panjang dari d dengan autokovarians sampai lag h-1
    def autocov(k: int) -> float:
        return float(np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n) if k > 0 else float(np.sum((d - d_mean) ** 2) / n)

    gamma = autocov(0) + 2 * sum(autocov(k) for k in range(1, h))
    dm = d_mean / np.sqrt(gamma / n)
    # Koreksi HLN untuk sampel kecil
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    pvalue = 2 * stats.t.sf(np.abs(hln), df=n - 1)
    return {"dm_statistic": float(hln), "pvalue": float(pvalue), "model_1_lebih_baik": bool(hln < 0 and pvalue < 0.05)}
