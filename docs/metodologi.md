# Metodologi

Dokumen ini merinci setiap tahap analisis dan padanannya dengan skripsi.

## 1. Pembagian data

Data dibagi secara kronologis 84 : 21 (80 % : 20 %). Tidak ada pengacakan karena urutan waktu adalah informasi inti pada runtun waktu; pengacakan akan membocorkan masa depan ke data training.

## 2. Uji karakteristik

| Tujuan | Skripsi (R) | Repo ini (Python) |
|---|---|---|
| Linieritas | `tseries::terasvirta.test` | Regresi Volterra: uji F penambahan suku kuadrat dan kubik dari prediksi linier (`diagnostics.terasvirta_proxy`) |
| Stasioneritas | Pembacaan visual ACF/PACF | ACF/PACF ditambah uji ADF (`adfuller`) dan KPSS |
| Varians stasioner | `forecast::BoxCox.lambda` (batas -1 sampai 1) | MLE `scipy.stats.boxcox_llf` dengan batas yang sama (`transform.estimate_lambda`) |

## 3. ARIMA

1. Transformasi Box-Cox pada data training, lalu standardisasi (affine, tidak mengubah model).
2. Differencing hingga stasioner. Pada differencing 1 tidak ada lag yang signifikan di ACF/PACF (sama dengan skripsi), sehingga dilakukan differencing 2.
3. Kandidat dari ACF (memotong lag 1) dan PACF (memotong lag 1 sampai 2): (0,2,1), (1,2,0), (2,2,0), (1,2,1), (2,2,1).
4. Estimasi dengan `statsmodels.tsa.arima.model.ARIMA`, tanpa konstanta (trend "n") karena differencing dua kali.
5. Seleksi: semua parameter signifikan (p < 0,05), residual white noise (Ljung-Box lag 12, 24, 36, 48), normalitas (Kolmogorov-Smirnov dan Shapiro-Wilk), lalu RMSE training terkecil.
6. Peramalan 21 langkah, destandardisasi, invers Box-Cox.

## 4. SVR

1. Fitur: lag-1 kurs (PACF data level hanya signifikan di lag 1).
2. Standardisasi fitur (`StandardScaler`) dan target (manual, disimpan di pipeline) supaya epsilon pada grid 0 sampai 1 bermakna.
3. Grid search `C` = 2^0..2^9, `gamma` = 2^-3..2^3, `epsilon` = 0..1 (770 kombinasi) untuk kernel RBF dan polynomial derajat 2.
4. Validasi: `TimeSeriesSplit(n_splits=5)` dengan skor RMSE. Data testing tidak dipakai sama sekali saat tuning.
5. Prediksi testing dengan dua skema: satu langkah (input aktual t-1, seperti skripsi) dan rekursif (input prediksi sendiri).

## 5. Evaluasi

- RMSE, MAE, MAPE pada skala rupiah.
- Uji Diebold-Mariano (1995) dengan koreksi Harvey, Leybourne, dan Newbold (1997) pada fungsi galat kuadrat, horizon 1, untuk menguji H0 bahwa kedua model sama akuratnya.

## 6. Risiko

- Log return mingguan r_t = ln(kurs_t / kurs_{t-1}).
- VaR historis: kuantil empiris 95 % dari r_t (return positif berarti rupiah melemah).
- VaR parametrik: mu + z_{0,95} sigma dengan asumsi normal.
- Expected Shortfall: rata-rata return yang melebihi VaR historis.
- Konversi ke rupiah: kurs_akhir × (exp(r) - 1) × nominal USD.

## 7. Reproducibility

Seluruh hasil dihasilkan ulang oleh `python -m kurs_forecast.pipeline` dan disimpan di `reports/hasil.json`. Notebook dibangun dari `notebooks/buat_notebook.py` dan dieksekusi otomatis. CI menjalankan lint, unit test, dan pipeline pada setiap push.
