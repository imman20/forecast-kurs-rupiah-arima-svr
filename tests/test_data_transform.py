"""Unit test untuk modul data dan transform."""

import numpy as np
import pandas as pd
import pytest

from kurs_forecast.data import load_kurs, split_train_test
from kurs_forecast.transform import boxcox_inverse, boxcox_transform, destandardize, difference, estimate_lambda, standardize


def test_load_kurs_bentuk_dan_isi():
    s = load_kurs()
    assert len(s) == 105
    assert s.index.is_monotonic_increasing
    assert (s > 0).all()
    assert s.iloc[0] == 14220 and s.iloc[-1] == 15605  # nilai pertama dan terakhir Lampiran 1


def test_split_kronologis():
    s = load_kurs()
    tr, te = split_train_test(s)
    assert len(tr) == 84 and len(te) == 21
    assert tr.index[-1] < te.index[0]


def test_split_n_train_tidak_valid():
    s = load_kurs()
    with pytest.raises(ValueError):
        split_train_test(s, n_train=len(s))


def test_boxcox_roundtrip():
    x = np.array([14000.0, 15000.0, 16000.0])
    for lam in (-1.0, -0.5, 0.0, 0.5, 1.0):
        y = boxcox_transform(x, lam)
        np.testing.assert_allclose(boxcox_inverse(y, lam), x, rtol=1e-10)


def test_lambda_terbatas_dan_mendekati_minus_satu():
    """Skripsi memperoleh lambda = -0,9999 dengan batas [-1, 1]; hasil harus di batas itu."""
    s = load_kurs()
    tr, _ = split_train_test(s)
    lam = estimate_lambda(tr)
    assert -1.0 <= lam <= 1.0
    assert lam == pytest.approx(-1.0, abs=1e-3)


def test_standardize_roundtrip_dan_konstan():
    z = np.array([1.0, 2.0, 3.0, 4.0])
    zs, m, sd = standardize(z)
    assert abs(zs.mean()) < 1e-12 and abs(zs.std() - 1) < 1e-12
    np.testing.assert_allclose(destandardize(zs, m, sd), z)
    with pytest.raises(ValueError):
        standardize(np.ones(5))


def test_difference_panjang():
    x = pd.Series(np.arange(10, dtype=float))
    assert len(difference(x, 1)) == 9 and len(difference(x, 2)) == 8
    assert np.allclose(difference(x, 1), 1.0)  # deret linear: diff pertama konstan
