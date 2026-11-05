"""Paket kurs_forecast.

Implementasi ulang (dan pengembangan) skripsi "Perbandingan Model ARIMA dan
SVR untuk Prediksi Nilai Tukar Rupiah terhadap Dolar AS" dalam Python.

Modul:
- data        : memuat, memvalidasi, dan membagi data kurs mingguan
- transform   : transformasi Box-Cox dan differencing
- diagnostics : uji stasioneritas, white noise, dan normalitas residual
- models      : ARIMA dan SVR beserta pencarian hyperparameter
- metrics     : RMSE, MAE, MAPE, dan uji Diebold-Mariano
- risk        : Value at Risk dan Expected Shortfall dari return kurs
- plots       : fungsi visualisasi yang dipakai notebook, laporan, dan aplikasi
- pipeline    : menjalankan seluruh alur dan menyimpan hasil
"""

__version__ = "1.0.0"
