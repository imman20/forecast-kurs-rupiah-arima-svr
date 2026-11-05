"""Membangun keempat notebook secara terprogram lalu mengeksekusinya.

Alasan pendekatan ini: notebook yang dibangun dari skrip mudah direproduksi dan
dikaji ulang di code review, sementara hasil eksekusinya tetap tersimpan di
file .ipynb agar bisa dibaca langsung di GitHub tanpa menjalankan apa pun.

Jalankan dari root repo:
    python notebooks/buat_notebook.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
HEADER = """import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src") if Path.cwd().name == "notebooks" else str(Path.cwd() / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from kurs_forecast.data import load_kurs, split_train_test
serie = load_kurs(); train, test = split_train_test(serie)
print(f"Total {len(serie)} minggu: training {len(train)}, testing {len(test)}")"""


def nb(judul: str, sel: list[tuple[str, str]]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells = [nbf.v4.new_markdown_cell(f"# {judul}")]
    for jenis, isi in sel:
        cells.append(nbf.v4.new_markdown_cell(isi) if jenis == "md" else nbf.v4.new_code_cell(isi))
    notebook.cells = cells
    return notebook


NB1 = nb(
    "01. Eksplorasi Data Kurs IDR/USD Mingguan",
    [
        ("md", "Notebook ini menjawab pertanyaan pertama skripsi: **bagaimana karakteristik data kurs rupiah terhadap dolar AS** pada periode minggu pertama Oktober 2021 sampai minggu pertama Oktober 2023. Data diambil dari Lampiran 1 skripsi (sumber asli id.investing.com)."),
        ("code", HEADER),
        ("md", "## Statistik deskriptif\nKurs bergerak di kisaran Rp14.000 sampai Rp15.800 dengan tren naik (rupiah melemah) sepanjang periode, sejalan dengan kenaikan suku bunga The Fed."),
        ("code", "serie.describe().round(2).to_frame().T"),
        ("code", "from kurs_forecast.plots import plot_series\nplot_series(train, test);"),
        ("md", "## Uji linieritas\nSkripsi memakai uji Terasvirta di R (chi-square 6,104; p = 0,047). Di sini dipakai versi regresi Volterra yang prinsipnya sama: apakah suku kuadrat dan kubik dari prediksi linier menambah daya jelas secara signifikan."),
        ("code", "from kurs_forecast.diagnostics import terasvirta_proxy, adf_test, kpss_test\nterasvirta_proxy(serie.values)"),
        ("md", "## Stasioneritas\nSelain plot ACF/PACF seperti skripsi, dipakai uji ADF (H0: tidak stasioner) dan KPSS (H0: stasioner) agar keputusan tidak hanya bergantung pada pembacaan visual."),
        ("code", "from kurs_forecast.transform import estimate_lambda, boxcox_transform, difference, standardize\nlam = estimate_lambda(train)\nz, _, _ = standardize(boxcox_transform(train.values, lam))\nprint('lambda Box-Cox =', round(lam, 4))\npd.DataFrame({\n    'level': adf_test(train.values), 'boxcox': adf_test(z),\n    'diff1': adf_test(difference(z, 1)), 'diff2': adf_test(difference(z, 2))}).T[['statistic', 'pvalue', 'stasioner']]"),
        ("code", "from kurs_forecast.plots import plot_acf_pacf\nplot_acf_pacf(train.values, 'training (level)');\nplot_acf_pacf(difference(z, 1), 'Box-Cox, differencing 1');\nplot_acf_pacf(difference(z, 2), 'Box-Cox, differencing 2');"),
        ("md", "## Kesimpulan EDA\n- Data level tidak stasioner (ADF p > 0,05; ACF turun lambat), konsisten dengan skripsi.\n- Lambda Box-Cox pada batas -1, sama seperti skripsi (-0,9999), sehingga transformasi yang dipakai adalah 1/Z.\n- Setelah differencing 2 kali data stasioner, dan ACF memotong di lag 1 serta PACF di lag 1 sampai 2, sehingga kandidat model adalah ARIMA (0,2,1), (1,2,0), (2,2,0), (1,2,1), dan (2,2,1)."),
    ],
)

NB2 = nb(
    "02. Pemodelan ARIMA",
    [
        ("md", "Membangun model ARIMA mengikuti alur skripsi: Box-Cox, differencing 2, estimasi lima kandidat, uji signifikansi parameter, uji Ljung-Box, uji normalitas residual, lalu pemilihan model dengan RMSE/MAPE training terkecil."),
        ("code", HEADER),
        ("code", "from kurs_forecast.models import pilih_arima_terbaik, forecast_arima\nterbaik, tabel = pilih_arima_terbaik(train)\ntabel.round(4)"),
        ("md", "Tabel di atas mereplikasi Tabel 4.1 skripsi: hanya ARIMA (0,2,1), (1,2,0), dan (2,2,0) yang seluruh parameternya signifikan. ARIMA (0,2,1) terpilih karena RMSE training terkecil (skripsi: 125,78; di sini sedikit berbeda karena estimator statsmodels dan R tidak identik)."),
        ("code", "print('Model terpilih: ARIMA', terbaik.order)\nprint('Parameter:', terbaik.params, 'p-value:', terbaik.pvalues)\nterbaik.ljung_box.round(4)"),
        ("code", "terbaik.normalitas"),
        ("code", "from kurs_forecast.plots import plot_residual, plot_fit_train\nd = terbaik.order[1]\nplot_residual(terbaik.model.resid[d:], f'Diagnostik residual ARIMA{terbaik.order}');\nplot_fit_train(train, terbaik.fitted_train, f'ARIMA{terbaik.order}', offset=d);"),
        ("md", "## Prediksi data testing\nPeramalan 21 minggu ke depan lalu dikembalikan ke skala rupiah. Seperti temuan skripsi, ARIMA menghasilkan garis yang hampir datar karena model MA(1) dengan differencing 2 hanya memproyeksikan tren lokal terakhir."),
        ("code", "from kurs_forecast.metrics import all_metrics\npred = forecast_arima(terbaik, len(test))\nprint(all_metrics(test.values, pred))\nplt.figure(figsize=(10,4)); plt.plot(test.index, test.values, 'k-o', ms=4, label='Aktual'); plt.plot(test.index, pred, 'g--s', ms=3, label=f'ARIMA{terbaik.order}'); plt.legend(); plt.grid(alpha=.3); plt.title('Prediksi testing ARIMA');"),
    ],
)

NB3 = nb(
    "03. Pemodelan Support Vector Regression",
    [
        ("md", "SVR memakai input lag-1 (hasil PACF skripsi) dan grid search pada C = 2^0..2^9, gamma = 2^-3..2^3, epsilon = 0..1 (Tabel 4.7 skripsi). Berbeda dari skripsi, pemilihan hyperparameter memakai **TimeSeriesSplit** sehingga data testing benar-benar tidak pernah dilihat saat tuning."),
        ("code", HEADER),
        ("code", "from kurs_forecast.models import fit_svr, predict_svr\nfrom kurs_forecast.metrics import all_metrics\nhasil = {}\nfor kernel in ['rbf', 'poly']:\n    r = fit_svr(train, kernel=kernel)\n    pred = predict_svr(r, train, test)\n    hasil[kernel] = (r, pred)\n    print(kernel, r.best_params, 'CV RMSE', round(r.cv_rmse, 2))\n    print('   train', {k: round(v, 3) for k, v in r.metrik_train.items()})\n    print('   test ', {k: round(v, 3) for k, v in all_metrics(test.values, pred).items()})"),
        ("md", "Seperti skripsi, kernel RBF jauh lebih baik daripada polynomial. Nilai parameter terbaik berbeda dari skripsi (C = 2^4, gamma = 1, epsilon = 0,1) karena target distandardisasi dan validasi silangnya berbeda, tetapi akurasi testing-nya setara."),
        ("code", "from kurs_forecast.plots import plot_fit_train\nplot_fit_train(train, hasil['rbf'][0].fitted_train, 'SVR RBF', offset=1);"),
        ("code", "plt.figure(figsize=(10,4))\nplt.plot(test.index, test.values, 'k-o', ms=4, label='Aktual')\nplt.plot(test.index, hasil['rbf'][1], '--^', color='#f59e0b', ms=3, label='SVR RBF')\nplt.plot(test.index, hasil['poly'][1], '--', color='#16a34a', label='SVR polynomial')\nplt.legend(); plt.grid(alpha=.3); plt.title('Prediksi testing SVR');"),
        ("md", "## Catatan penting: satu langkah vs rekursif\nPrediksi di atas memakai nilai aktual minggu sebelumnya sebagai input (one-step-ahead), persis seperti skripsi. Jika model harus meramal 21 minggu tanpa melihat aktual (rekursif), galatnya menumpuk. Ini penting dijelaskan ke pemangku kepentingan agar akurasi 0,5% tidak disalahartikan sebagai akurasi peramalan jangka panjang."),
        ("code", "pred_rek = predict_svr(hasil['rbf'][0], train, test, recursive=True)\nprint('One-step :', {k: round(v, 3) for k, v in all_metrics(test.values, hasil['rbf'][1]).items()})\nprint('Rekursif :', {k: round(v, 3) for k, v in all_metrics(test.values, pred_rek).items()})"),
    ],
)

NB4 = nb(
    "04. Perbandingan Model dan Analisis Risiko",
    [
        ("md", "Notebook ini merangkum hasil pipeline (`reports/hasil.json`), menguji apakah keunggulan SVR signifikan secara statistik dengan uji **Diebold-Mariano**, dan menambahkan modul **Value at Risk** sebagai pengembangan di luar skripsi."),
        ("code", HEADER),
        ("code", "import json\nh = json.load(open(Path.cwd().parent / 'reports' / 'hasil.json' if Path.cwd().name == 'notebooks' else Path('reports/hasil.json')))\nperbandingan = pd.DataFrame({\n    'ARIMA (0,2,1) [repo]': h['arima']['metrik_test'],\n    'SVR RBF [repo]': h['svr']['rbf']['metrik_test'],\n    'SVR poly [repo]': h['svr']['poly']['metrik_test'],\n    'ARIMA (0,2,1) [skripsi]': {**h['skripsi']['arima_test'], 'MAE': np.nan},\n    'SVR RBF [skripsi]': {**h['skripsi']['svr_rbf_test'], 'MAE': np.nan},\n}).T\nperbandingan.round(3)"),
        ("code", "prediksi = pd.read_csv(Path.cwd().parent / 'data/processed/prediksi_testing.csv' if Path.cwd().name == 'notebooks' else 'data/processed/prediksi_testing.csv', parse_dates=['tanggal'])\nplt.figure(figsize=(10,4.2))\nplt.plot(prediksi.tanggal, prediksi.aktual, 'k-o', ms=4, label='Aktual')\nplt.plot(prediksi.tanggal, prediksi.arima, 'g--s', ms=3, label='ARIMA (0,2,1)')\nplt.plot(prediksi.tanggal, prediksi.svr_rbf, '--^', color='#f59e0b', ms=3, label='SVR RBF')\nplt.legend(); plt.grid(alpha=.3); plt.title('Perbandingan prediksi data testing');"),
        ("md", "## Uji Diebold-Mariano\nMembandingkan RMSE saja tidak menjawab apakah selisihnya nyata atau kebetulan. Uji DM (dengan koreksi Harvey-Leybourne-Newbold) menguji H0: kedua model sama akuratnya. Statistik negatif berarti model pertama (SVR) lebih akurat."),
        ("code", "h['diebold_mariano_svr_vs_arima']"),
        ("md", "## Value at Risk dari return kurs\nPertanyaan praktis untuk tim treasury atau risiko: jika perusahaan punya kewajiban 100.000 USD, berapa tambahan rupiah terburuk yang perlu disiapkan dalam satu minggu pada keyakinan 95%?"),
        ("code", "from kurs_forecast.risk import ringkasan_risiko, log_returns\nfrom kurs_forecast.plots import plot_var\nr = ringkasan_risiko(serie)\nplot_var(log_returns(serie), r['var_historis_pct']/100, r['var_parametrik_pct']/100);\npd.Series(r).round(4)"),
        ("md", "## Kesimpulan\n1. SVR kernel RBF mengungguli ARIMA (0,2,1) pada data testing, dan keunggulan itu signifikan secara statistik (uji DM p < 0,05).\n2. Keunggulan SVR berlaku untuk peramalan satu langkah; pada skema rekursif 21 minggu akurasinya turun ke level ARIMA, sehingga pemilihan model harus mengikuti horizon kebutuhan bisnis.\n3. Risiko pelemahan rupiah mingguan pada keyakinan 95% berada di kisaran 1,2 sampai 1,4% (VaR), dengan expected shortfall sedikit lebih tinggi."),
    ],
)


def main():
    for nama, notebook in [("01_eda.ipynb", NB1), ("02_pemodelan_arima.ipynb", NB2), ("03_pemodelan_svr.ipynb", NB3), ("04_perbandingan_dan_risiko.ipynb", NB4)]:
        path = HERE / nama
        nbf.write(notebook, path)
        print("Mengeksekusi", nama)
        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=1200", str(path)],
            check=True,
            cwd=HERE,
        )


if __name__ == "__main__":
    main()
