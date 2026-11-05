"""Pemuatan dan pembagian data kurs IDR/USD mingguan.

Data utama berasal dari Lampiran 1 skripsi (105 observasi mingguan,
3 Oktober 2021 sampai 1 Oktober 2023, sumber asli id.investing.com).
Fungsi `download_latest` disediakan untuk memperbarui data dari Yahoo Finance
jika koneksi internet tersedia, sehingga analisis dapat diulang pada periode lain.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Path default relatif terhadap root repo, supaya skrip bisa dijalankan dari mana saja.
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "kurs_idr_usd_mingguan.csv"

# Pembagian persis seperti skripsi: 84 minggu training, 21 minggu testing (80:20).
N_TRAIN_DEFAULT = 84


def load_kurs(path: Path | str = RAW_PATH) -> pd.Series:
    """Memuat data kurs sebagai Series berindeks tanggal.

    Validasi sederhana dilakukan di sini (bukan di tempat lain) supaya setiap
    pemakai data mendapat jaminan yang sama: tanggal urut, tidak ada nilai
    kosong, dan semua kurs positif.
    """
    df = pd.read_csv(path, parse_dates=["tanggal"])
    if df["kurs"].isna().any():
        raise ValueError("Terdapat nilai kurs kosong pada data mentah.")
    if (df["kurs"] <= 0).any():
        raise ValueError("Kurs harus bernilai positif.")
    if not df["tanggal"].is_monotonic_increasing:
        raise ValueError("Tanggal harus terurut naik.")
    serie = df.set_index("tanggal")["kurs"].astype(float)
    serie.index.name = "tanggal"
    serie.name = "kurs"
    return serie


def split_train_test(serie: pd.Series, n_train: int = N_TRAIN_DEFAULT) -> tuple[pd.Series, pd.Series]:
    """Membagi data secara kronologis (tanpa pengacakan).

    Pada runtun waktu, pengacakan akan membocorkan informasi masa depan ke
    data training, sehingga pembagian harus mengikuti urutan waktu.
    """
    if not 0 < n_train < len(serie):
        raise ValueError("n_train harus di antara 1 dan panjang data dikurangi 1.")
    return serie.iloc[:n_train], serie.iloc[n_train:]


def download_latest(start: str = "2021-10-01", end: str | None = None) -> pd.Series:
    """Mengunduh kurs USD/IDR harian dari Yahoo Finance lalu meresampel mingguan.

    Dipakai hanya jika pengguna ingin memperbarui data; pipeline utama tetap
    memakai CSV di repo agar hasil selalu dapat direproduksi tanpa internet.
    Resample memakai nilai hari Minggu (akhir pekan) agar sejalan dengan
    tanggal mingguan pada data skripsi.
    """
    import yfinance as yf  # impor lokal supaya paket ini opsional

    df = yf.download("IDR=X", start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError("Unduhan kosong; periksa koneksi internet atau ticker.")
    close = df["Close"].squeeze()
    weekly = close.resample("W-SUN").last().dropna()
    weekly.name = "kurs"
    weekly.index.name = "tanggal"
    return weekly.astype(float)
