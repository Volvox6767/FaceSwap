@echo off
setlocal
chcp 65001 >nul
title FaceSwap - Ahmet Gedik

echo ============================================================
echo   FaceSwap kurulumu
echo   Author : Ahmet Gedik
echo   Insta  : https://www.instagram.com/ahmetgedik67
echo ============================================================
echo.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [HATA] Python bulunamadi. https://www.python.org/downloads/ adresinden
    echo        Python 3.10 veya 3.11 kurun ve "Add to PATH" isaretleyin.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Sanal ortam olusturuluyor...
    python -m venv .venv
) else (
    echo [1/4] Sanal ortam mevcut.
)

echo [2/4] Bağımlılıklar kuruluyor...
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [HATA] Bağımlılık kurulumu başarısız. Yukarıdaki hatayı kontrol edin.
    pause
    exit /b 1
)

echo [3/4] FFmpeg kontrol ediliyor...
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [UYARI] FFmpeg bulunamadi. Kurmak icin:  winget install Gyan.FFmpeg
    echo         Kurduktan sonra bu pencereyi yeniden acin.
)

echo [4/4] Modeller indiriliyor...
python download_models.py
if errorlevel 1 (
    echo [UYARI] Bazı modeller indirilemedi. Web UI "Kurulum" sekmesinden tekrar deneyin.
)

echo.
echo ============================================================
echo   Kurulum tamam! Web arayuzu baslatiliyor...
echo   Tarayicida: http://127.0.0.1:7860
echo ============================================================
python app.py
pause
