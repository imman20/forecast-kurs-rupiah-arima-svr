"""Unit test untuk metrik, uji Diebold-Mariano, dan modul risiko."""

import numpy as np
import pandas as pd
import pytest

from kurs_forecast.metrics import all_metrics, diebold_mariano, mae, mape, rmse
from kurs_forecast.risk import expected_shortfall, log_returns, ringkasan_risiko, var_historis, var_parametrik


def test_metrik_prediksi_sempurna_nol():
    a = np.array([100.0, 200.0, 300.0])
    assert rmse(a, a) == 0 and mae(a, a) == 0 and mape(a, a) == 0


def test_metrik_nilai_terhitung_manual():
    a = np.array([100.0, 200.0])
    p = np.array([110.0, 190.0])
    assert rmse(a, p) == pytest.approx(10.0)
    assert mae(a, p) == pytest.approx(10.0)
    assert mape(a, p) == pytest.approx((10 / 100 + 10 / 200) / 2 * 100)
    assert set(all_metrics(a, p)) == {"RMSE", "MAE", "MAPE"}


def test_diebold_mariano_mendeteksi_model_lebih_baik():
    rng = np.random.default_rng(0)
    aktual = rng.normal(15000, 100, 60)
    pred_baik = aktual + rng.normal(0, 10, 60)
    pred_buruk = aktual + rng.normal(0, 200, 60)
    hasil = diebold_mariano(aktual, pred_baik, pred_buruk)
    assert hasil["dm_statistic"] < 0
    assert hasil["pvalue"] < 0.05
    assert hasil["model_1_lebih_baik"]


def test_diebold_mariano_simetris():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 40)
    p1, p2 = a + rng.normal(0, 1, 40), a + rng.normal(0, 1, 40)
    h12, h21 = diebold_mariano(a, p1, p2), diebold_mariano(a, p2, p1)
    assert h12["dm_statistic"] == pytest.approx(-h21["dm_statistic"])


def test_var_urutan_logis():
    rng = np.random.default_rng(2)
    s = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, 300))) * 15000)
    r = log_returns(s)
    vh, es = var_historis(r), expected_shortfall(r)
    assert len(r) == 299
    assert es >= vh  # expected shortfall selalu >= VaR pada tingkat yang sama
    assert var_parametrik(r) > r.mean()


def test_ringkasan_risiko_kunci_dan_tanda():
    s = pd.Series(np.linspace(14000, 15600, 105) + np.sin(np.arange(105)) * 200)
    out = ringkasan_risiko(s, nominal_usd=1_000)
    for k in ("var_historis_pct", "var_parametrik_pct", "es_historis_pct", "var_historis_rp"):
        assert k in out
    assert out["n_return"] == 104
