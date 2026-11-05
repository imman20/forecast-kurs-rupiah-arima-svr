"""Unit test untuk model ARIMA dan SVR (dibuat ringan agar CI cepat)."""

import numpy as np
import pytest

from kurs_forecast.data import load_kurs, split_train_test
from kurs_forecast.models import buat_fitur_lag, fit_arima, fit_svr, forecast_arima, predict_svr


@pytest.fixture(scope="module")
def data():
    return split_train_test(load_kurs())


def test_arima_021_mereplikasi_skripsi(data):
    """Skripsi: ARIMA(0,2,1) dengan MA(1) = -0,98277 dan p-value < 0,05."""
    tr, te = data
    r = fit_arima(tr, (0, 2, 1))
    assert r.params["ma.L1"] == pytest.approx(-0.983, abs=0.01)
    assert r.semua_signifikan
    assert bool(r.ljung_box["white_noise"].all())
    fc = forecast_arima(r, len(te))
    assert fc.shape == (21,)
    assert np.all(fc > 10_000) and np.all(fc < 20_000)  # berada di skala rupiah yang masuk akal


def test_fitur_lag_bentuk(data):
    tr, _ = data
    X, y, idx = buat_fitur_lag(tr, n_lag=2)
    assert X.shape == (82, 2) and y.shape == (82,)
    assert X[0, 0] == tr.values[1] and X[0, 1] == tr.values[0]  # lag1 = t-1, lag2 = t-2


def test_svr_grid_kecil_prediksi_masuk_akal(data):
    tr, te = data
    grid = {"svr__C": [16.0], "svr__gamma": [1.0], "svr__epsilon": [0.1]}  # parameter skripsi
    r = fit_svr(tr, kernel="rbf", grid=grid, n_splits=3)
    pred = predict_svr(r, tr, te)
    assert pred.shape == (21,)
    assert np.mean(np.abs(pred - te.values) / te.values) < 0.03  # MAPE < 3 %
    pred_rek = predict_svr(r, tr, te, recursive=True)
    assert pred_rek.shape == (21,)
