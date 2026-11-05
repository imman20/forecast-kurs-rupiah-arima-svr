"""Menjalankan seluruh alur analisis dan menyimpan hasil ke reports/ dan assets/.

Jalankan dari root repo:
    python -m kurs_forecast.pipeline

Semua angka di README dan laporan diambil dari file reports/hasil.json yang
dihasilkan skrip ini, sehingga tidak ada angka yang ditulis tangan.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import diagnostics as dg
from . import plots
from .data import ROOT, load_kurs, split_train_test
from .metrics import all_metrics, diebold_mariano
from .models import fit_svr, forecast_arima, pilih_arima_terbaik, predict_svr
from .risk import log_returns, ringkasan_risiko
from .transform import boxcox_transform, difference, estimate_lambda, standardize

REPORTS = ROOT / "reports"
ASSETS = ROOT / "assets"
PROCESSED = ROOT / "data" / "processed"


def jalankan(verbose: bool = True) -> dict:
    for p in (REPORTS, ASSETS, PROCESSED):
        p.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    serie = load_kurs()
    train, test = split_train_test(serie)
    log(f"Data: {len(serie)} minggu | training {len(train)} | testing {len(test)}")

    # --- 1. Karakteristik data -------------------------------------------------
    plots.plot_series(train, test, ASSETS / "01_data_train_test.png")
    lam = estimate_lambda(train)
    z, _, _ = standardize(boxcox_transform(train.values, lam))  # distandardisasi agar uji numerik stabil
    karakteristik = {
        "n_total": int(len(serie)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "kurs_min": float(serie.min()),
        "kurs_max": float(serie.max()),
        "kurs_mean": float(serie.mean()),
        "terasvirta": dg.terasvirta_proxy(serie.values),
        "boxcox_lambda": lam,
        "adf_level": dg.adf_test(train.values),
        "adf_boxcox": dg.adf_test(z),
        "adf_diff1": dg.adf_test(difference(z, 1)),
        "adf_diff2": dg.adf_test(difference(z, 2)),
        "kpss_level": dg.kpss_test(train.values),
        "kpss_diff2": dg.kpss_test(difference(z, 2)),
    }
    plots.plot_acf_pacf(train.values, "training (level)", path=ASSETS / "02_acf_pacf_level.png")
    plots.plot_acf_pacf(difference(z, 1), "Box-Cox, differencing 1", path=ASSETS / "03_acf_pacf_diff1.png")
    plots.plot_acf_pacf(difference(z, 2), "Box-Cox, differencing 2", path=ASSETS / "04_acf_pacf_diff2.png")
    log(f"Lambda Box-Cox = {lam:.4f}; ADF diff2 p = {karakteristik['adf_diff2']['pvalue']:.4f}")

    # --- 2. ARIMA ----------------------------------------------------------------
    arima_terbaik, tabel_arima = pilih_arima_terbaik(train)
    tabel_arima.to_csv(REPORTS / "tabel_kandidat_arima.csv", index=False)
    pred_arima = forecast_arima(arima_terbaik, len(test))
    metrik_arima = all_metrics(test.values, pred_arima)
    d = arima_terbaik.order[1]
    plots.plot_fit_train(train, arima_terbaik.fitted_train, f"ARIMA{arima_terbaik.order}", offset=d, path=ASSETS / "05_fit_train_arima.png")
    plots.plot_residual(arima_terbaik.model.resid[d:], f"Diagnostik residual ARIMA{arima_terbaik.order}", ASSETS / "06_residual_arima.png")
    log(f"ARIMA terbaik {arima_terbaik.order}: test RMSE {metrik_arima['RMSE']:.2f}, MAPE {metrik_arima['MAPE']:.3f}%")

    # --- 3. SVR ------------------------------------------------------------------
    hasil_svr = {}
    for kernel in ("rbf", "poly"):
        res = fit_svr(train, kernel=kernel)
        pred = predict_svr(res, train, test)
        pred_rek = predict_svr(res, train, test, recursive=True)
        hasil_svr[kernel] = {
            "result": res,
            "pred": pred,
            "pred_rekursif": pred_rek,
            "metrik_test": all_metrics(test.values, pred),
            "metrik_test_rekursif": all_metrics(test.values, pred_rek),
        }
        mt = hasil_svr[kernel]["metrik_test"]
        log(f"SVR {kernel}: params {res.best_params}, test RMSE {mt['RMSE']:.2f}, MAPE {mt['MAPE']:.3f}%")
    svr_rbf = hasil_svr["rbf"]
    plots.plot_fit_train(train, svr_rbf["result"].fitted_train, "SVR RBF", offset=1, path=ASSETS / "07_fit_train_svr_rbf.png")

    # --- 4. Perbandingan ---------------------------------------------------------
    plots.plot_perbandingan_test(test, pred_arima, svr_rbf["pred"], ASSETS / "08_perbandingan_testing.png", label_arima=f"ARIMA{arima_terbaik.order}")
    dm = diebold_mariano(test.values, svr_rbf["pred"], pred_arima)
    prediksi = pd.DataFrame(
        {
            "tanggal": test.index,
            "aktual": test.values,
            "arima": pred_arima,
            "svr_rbf": svr_rbf["pred"],
            "svr_rbf_rekursif": svr_rbf["pred_rekursif"],
            "svr_poly": hasil_svr["poly"]["pred"],
        }
    )
    prediksi.to_csv(PROCESSED / "prediksi_testing.csv", index=False)

    # --- 5. Risiko ---------------------------------------------------------------
    risiko = ringkasan_risiko(serie)
    r = log_returns(serie)
    plots.plot_var(r, risiko["var_historis_pct"] / 100, risiko["var_parametrik_pct"] / 100, ASSETS / "09_var_return.png")

    hasil = {
        "karakteristik": karakteristik,
        "arima": {
            "order": list(arima_terbaik.order),
            "lambda": arima_terbaik.lam,
            "params": arima_terbaik.params,
            "pvalues": arima_terbaik.pvalues,
            "aic": arima_terbaik.aic,
            "ljung_box": arima_terbaik.ljung_box.reset_index().to_dict(orient="records"),
            "normalitas": arima_terbaik.normalitas,
            "metrik_train": arima_terbaik.metrik_train,
            "metrik_test": metrik_arima,
            "tabel_kandidat": tabel_arima.to_dict(orient="records"),
        },
        "svr": {
            k: {
                "best_params": v["result"].best_params,
                "cv_rmse": v["result"].cv_rmse,
                "metrik_train": v["result"].metrik_train,
                "metrik_test": v["metrik_test"],
                "metrik_test_rekursif": v["metrik_test_rekursif"],
            }
            for k, v in hasil_svr.items()
        },
        "diebold_mariano_svr_vs_arima": dm,
        "risiko": risiko,
        "skripsi": {  # angka pembanding dari naskah skripsi (R), untuk transparansi
            "arima_order": [0, 2, 1],
            "arima_ma1": -0.98277,
            "arima_test": {"RMSE": 425.398, "MAPE": 2.49},
            "svr_rbf_params": {"C": 16, "epsilon": 0.1, "gamma": 1},
            "svr_rbf_test": {"RMSE": 93.653, "MAPE": 0.539},
        },
    }
    with open(REPORTS / "hasil.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False, default=_json_default)
    log(f"Hasil disimpan ke {REPORTS / 'hasil.json'}")
    return hasil


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"Tipe {type(o)} tidak bisa diserialisasi")


if __name__ == "__main__":
    jalankan()
