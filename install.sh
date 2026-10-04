#!/usr/bin/env bash
# FaceSwap installer (Linux / macOS) — Ahmet Gedik
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  FaceSwap kurulumu"
echo "  Author : Ahmet Gedik"
echo "  Insta  : https://www.instagram.com/ahmetgedik67"
echo "============================================================"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[HATA] python3 bulunamadı. Python 3.10+ kurun: https://www.python.org/downloads/"
    exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "[1/4] Sanal ortam oluşturuluyor..."
    python3 -m venv .venv
else
    echo "[1/4] Sanal ortam mevcut."
fi

echo "[2/4] Bağımlılıklar kuruluyor..."
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[3/4] FFmpeg kontrol ediliyor..."
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[UYARI] FFmpeg bulunamadı."
    echo "  macOS : brew install ffmpeg"
    echo "  Linux : sudo apt install ffmpeg"
fi

echo "[4/4] Modeller indiriliyor..."
python download_models.py || echo "[UYARI] Bazı modeller indirilemedi; app.py Kurulum sekmesinden tekrar deneyin."

echo
echo "============================================================"
echo "  Kurulum tamam! Web arayüzü başlatılıyor..."
echo "  Tarayıcıda: http://127.0.0.1:7860"
echo "============================================================"
python app.py
