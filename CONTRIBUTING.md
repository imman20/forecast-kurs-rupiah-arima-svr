# Panduan Kontribusi

Terima kasih sudah tertarik. Repo ini adalah proyek portofolio pribadi, tetapi saran dan perbaikan tetap diterima.

## Alur singkat

1. Fork repo lalu buat branch dari `main`, misalnya `feat/model-ets`.
2. Pasang dependensi: `pip install -r requirements.txt`.
3. Pastikan `ruff check src tests app` dan `pytest -q` lulus sebelum membuka pull request.
4. Jika mengubah logika pemodelan, jalankan ulang `make pipeline` dan `make notebooks` supaya angka di `reports/hasil.json` dan notebook tetap sinkron dengan kode.
5. Tulis pesan commit mengikuti [Conventional Commits](https://www.conventionalcommits.org/), contoh: `feat: tambah model ETS sebagai pembanding`.

## Gaya kode

- Komentar dan docstring dalam Bahasa Indonesia, menjelaskan alasan ("mengapa"), bukan hanya apa yang dilakukan.
- Nama fungsi, variabel, dan kolom dalam Bahasa Inggris atau istilah teknis yang sudah lazim.
- Jangan menulis angka hasil secara manual di README; semua angka harus bisa ditelusuri ke `reports/hasil.json`.
