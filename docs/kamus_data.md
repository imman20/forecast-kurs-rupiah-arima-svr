# Kamus Data

## `data/raw/kurs_idr_usd_mingguan.csv`

| Kolom | Tipe | Keterangan |
|---|---|---|
| `index` | integer | Nomor urut 1 sampai 105, sama dengan Lampiran 1 skripsi |
| `tanggal` | date (YYYY-MM-DD) | Hari Minggu yang menandai akhir pekan pengamatan |
| `kurs` | float | Rupiah per 1 dolar AS pada penutupan minggu tersebut |

Sumber: id.investing.com (USD/IDR historical data), ditranskripsi dari Lampiran 1 skripsi dan diverifikasi terhadap Lampiran 4 (nilai Box-Cox).

## `data/processed/prediksi_testing.csv`

| Kolom | Keterangan |
|---|---|
| `tanggal` | Tanggal minggu testing (14 Mei 2023 sampai 1 Oktober 2023) |
| `aktual` | Kurs aktual |
| `arima` | Prediksi ARIMA (0,2,1) 21 langkah ke depan |
| `svr_rbf` | Prediksi SVR RBF satu langkah ke depan |
| `svr_rbf_rekursif` | Prediksi SVR RBF rekursif |
| `svr_poly` | Prediksi SVR polynomial satu langkah ke depan |

## `reports/hasil.json`

| Kunci | Isi |
|---|---|
| `karakteristik` | Jumlah data, rentang, uji nonlinieritas, lambda Box-Cox, uji ADF dan KPSS |
| `arima` | Orde, parameter, p-value, AIC, Ljung-Box, normalitas, metrik train/test, tabel kandidat |
| `svr` | Per kernel: parameter terbaik, RMSE CV, metrik train, test, dan test rekursif |
| `diebold_mariano_svr_vs_arima` | Statistik DM, p-value, keputusan |
| `risiko` | Statistik return, VaR historis/parametrik, Expected Shortfall, konversi rupiah |
| `skripsi` | Angka pembanding dari naskah skripsi untuk transparansi |
