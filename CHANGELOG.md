# Changelog

Semua perubahan penting pada proyek ini dicatat di sini.
Format mengikuti [Keep a Changelog](https://keepachangelog.com/id/1.1.0/) dan versi mengikuti [Semantic Versioning](https://semver.org/lang/id/).

## [1.0.0] - 2025-11-05

### Ditambahkan
- Data kurs IDR/USD mingguan 105 observasi (Lampiran 1 skripsi) dan fungsi pengunduh pembaruan dari Yahoo Finance.
- Paket `kurs_forecast`: pemuatan data, Box-Cox dengan batas lambda [-1, 1], uji ADF/KPSS/Ljung-Box/normalitas, uji nonlinieritas ala Terasvirta.
- Pemodelan ARIMA dengan seleksi lima kandidat mengikuti kriteria skripsi.
- Pemodelan SVR (RBF dan polynomial) dengan grid search dan TimeSeriesSplit, skema prediksi satu langkah dan rekursif.
- Uji Diebold-Mariano untuk membandingkan akurasi kedua model secara statistik.
- Modul risiko: VaR historis, VaR parametrik, dan Expected Shortfall.
- Pipeline `python -m kurs_forecast.pipeline` yang menghasilkan `reports/hasil.json` dan seluruh gambar di `assets/`.
- Empat notebook teranotasi, aplikasi Streamlit, unit test, dan CI GitHub Actions.
