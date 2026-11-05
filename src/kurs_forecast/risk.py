"""Modul risiko: Value at Risk (VaR) dan Expected Shortfall (ES) dari return kurs.

Ini pengembangan di luar skripsi. Tujuannya menunjukkan bagaimana hasil
peramalan kurs dapat dipakai untuk mengukur risiko posisi valuta asing,
yang relevan untuk posisi aktuaria, risk, dan treasury.

Konvensi: return dihitung sebagai log return mingguan kurs IDR per USD.
Return positif berarti rupiah melemah (butuh lebih banyak rupiah per dolar),
sehingga VaR di sini mengukur kerugian pemegang kewajiban dalam dolar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def log_returns(serie: pd.Series) -> pd.Series:
    r = np.log(serie).diff().dropna()
    r.name = "log_return"
    return r


def var_historis(returns: pd.Series, alpha: float = 0.05) -> float:
    """VaR historis: kuantil empiris dari distribusi return (tanpa asumsi distribusi)."""
    return float(np.quantile(returns, 1 - alpha))


def var_parametrik(returns: pd.Series, alpha: float = 0.05) -> float:
    """VaR parametrik dengan asumsi return normal: mu + z * sigma."""
    mu, sd = returns.mean(), returns.std(ddof=1)
    return float(mu + stats.norm.ppf(1 - alpha) * sd)


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """Rata-rata return pada ekor di luar VaR historis (kerugian saat kondisi buruk)."""
    v = var_historis(returns, alpha)
    ekor = returns[returns >= v]
    return float(ekor.mean()) if len(ekor) else v


def ringkasan_risiko(serie: pd.Series, alpha: float = 0.05, nominal_usd: float = 100_000) -> dict:
    """Ringkasan VaR/ES dalam persen dan rupiah untuk posisi nominal tertentu.

    Contoh: perusahaan punya kewajiban 100.000 USD; berapa tambahan rupiah
    terburuk yang perlu disiapkan dalam seminggu pada tingkat keyakinan 95%?
    """
    r = log_returns(serie)
    kurs_akhir = float(serie.iloc[-1])
    vh, vp, es = var_historis(r, alpha), var_parametrik(r, alpha), expected_shortfall(r, alpha)
    # Konversi log return ke rupiah: kurs_akhir * (exp(r) - 1) * nominal
    ke_rupiah = lambda x: float(kurs_akhir * (np.exp(x) - 1.0) * nominal_usd)
    return {
        "alpha": alpha,
        "n_return": int(len(r)),
        "mean_return_pct": float(r.mean() * 100),
        "std_return_pct": float(r.std(ddof=1) * 100),
        "var_historis_pct": vh * 100,
        "var_parametrik_pct": vp * 100,
        "es_historis_pct": es * 100,
        "nominal_usd": nominal_usd,
        "kurs_akhir": kurs_akhir,
        "var_historis_rp": ke_rupiah(vh),
        "var_parametrik_rp": ke_rupiah(vp),
        "es_historis_rp": ke_rupiah(es),
    }
