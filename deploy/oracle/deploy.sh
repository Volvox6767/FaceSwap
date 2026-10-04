#!/usr/bin/env bash
# FaceSwap -> Oracle Cloud Always Free (Ampere A1 / ARM64) dağıtımı.
#
# Kullanim (VM üzerinde, repo klonlanmış durumda):
#   git clone https://github.com/Volvox6767/FaceSwap
#   cd FaceSwap && bash deploy/oracle/deploy.sh
#
# Docker kurulu değilse kendisi kurar. Modeller kalıcı volume'a yazılır,
# yeniden başlatmada yeniden indirilmez (~950 MB).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

PORT="${FACESWAP_PORT:-7860}"
IMAGE="faceswap:local"
VOL_MODELS="faceswap-models"
VOL_WORK="faceswap-workspace"

echo "== 1/5 Docker kontrolu"
if ! command -v docker >/dev/null 2>&1; then
  echo "   Docker kuruluyor..."
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker

echo "== 2/5 Volume'lar"
docker volume create "$VOL_MODELS" >/dev/null
docker volume create "$VOL_WORK" >/dev/null

echo "== 3/5 Image build (ilk seferde ~5-10 dk, insightface derleniyor)"
# Ampere A1 = aarch64; insightface Linux'ta kaynaktan derlenir (cython<3 + cmake).
docker build -f hf/Dockerfile -t "$IMAGE" .

echo "== 4/5 Container baslatiliyor (port $PORT)"
docker rm -f faceswap 2>/dev/null || true
docker run -d --name faceswap \
  --restart unless-stopped \
  -p "$PORT:7860" \
  -e FACESWAP_MODELS=/app/models \
  -e FACESWAP_WORK=/app/workspace \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -v "$VOL_MODELS":/app/models \
  -v "$VOL_WORK":/app/workspace \
  "$IMAGE" >/dev/null

echo "== 5/5 Durum"
sleep 5
docker ps --filter name=faceswap --format "   {{.Names}}  {{.Status}}  {{.Ports}}"
echo "   loglar: docker logs -f faceswap"

cat <<EOF

Tamam. Uygulama sunucunun IP adresinde:
   http://<SUNUCU-IP>:$PORT

Notlar:
 * Oracle Always Free A1 2026 itibarıyla 2 OCPU / 12 GB RAM (eskiden 4/24 idi).
   FaceSwap CPU'da ~1-3 sn/kare; 30 sn'lik klip ~1-2 dk.
 * Ilk acilista modeller indirilir (loglar: "modeller indiriliyor").
 * Kalici disk: $VOL_MODELS (yeniden baslatmada indirme tekrarlanmaz).
 * Guvenlik listesi / firewall'da TCP $PORT acik olmali (Oracle Console'dan).
 * HTTPS icin oneklendirici (nginx + Let's Encrypt) eklenebilir.
EOF
