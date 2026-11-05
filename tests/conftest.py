"""Menambahkan src/ ke sys.path supaya tes bisa mengimpor paket tanpa instalasi."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
