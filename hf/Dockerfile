# FaceSwap — Hugging Face Docker Space (CPU)
# Runtime'da modeller indirilir, image'a gömülmez (~950 MB).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FACESWAP_MODELS=/app/models \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# ffmpeg: video birleştirme / ses kopyalama
# libgl1 + libglib2.0-0: opencv ihtiyaç duyuyor
# build-essential/cmake: insightface Cython uzantıları derleniyor
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Önce requirements (katman önbelleği)
COPY requirements.txt ./
# insightface 0.7.3 Linux'ta kaynaktan derlenir; ARM64 (Ampere A1) dahil.
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY app.py swap_engine.py realism.py video_io.py face_parser.py \
     runtime_utils.py gpu_devices.py download_models.py face_quality.json ./

RUN mkdir -p /app/models /app/workspace

EXPOSE 7860

CMD ["python", "app.py"]
