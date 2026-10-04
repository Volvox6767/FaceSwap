#!/usr/bin/env bash
# FaceSwap -> Hugging Face Space (Docker, CPU) yayinlama.
# Modeller image'a gomulmez; Space ilk acilista indirir (~950 MB).
#
# Kullanim:
#   HF_TOKEN=hf_xxx bash publish_hf.sh [space-name]
#
# NOT: Hugging Face, ucretsiz hesaplarda Gradio ve Docker Space olusturulmasina
# izin vermiyor ("requires a PRO subscription"). Bu script once Space'i olusturmayi
# dener; 402 alirsa PRO gerektigini net mesajla bildirir.
set -euo pipefail
cd "$(dirname "$0")"

SPACE_NAME="${1:-FaceSwap}"
HF_USER="${HF_USER:-volvox67}"
ROOT="$(pwd)"
OUT="$ROOT/../FaceSwap-space"
SPACE_URL="https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN gerekli. huggingface.co/settings/tokens adresinden 'Write' token olusturun."
  exit 1
fi

echo "[1/5] Space klasoru hazirlaniyor: $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
for f in app.py swap_engine.py realism.py video_io.py face_parser.py \
         runtime_utils.py gpu_devices.py download_models.py face_quality.json; do
  cp "$ROOT/$f" "$OUT/$f"
done
cp "$ROOT/hf/Dockerfile" "$OUT/Dockerfile"
cp "$ROOT/hf/README.md" "$OUT/README.md"
cp "$ROOT/hf/requirements.txt" "$OUT/requirements.txt"
printf 'workspace/\nmodels/\n__pycache__/\n*.pyc\n.pylibs/\n' > "$OUT/.gitignore"

echo "[2/5] Kod Space'e gonderiliyor"
cd "$OUT"
git init -q -b main 2>/dev/null || true
git config user.name "Ahmet Gedik"
git config user.email "$HF_USER@users.noreply.huggingface.co"
git add -A
git commit -q -m "FaceSwap Space: Docker + runtime model download" || echo "  (degisiklik yok)"
git remote remove hf 2>/dev/null || true
git remote add hf "https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$SPACE_NAME"

echo "[3/5] Hugging Face bos Space olusturuluyor (docker / cpu-basic)"
RESP="$(curl -s -w '\n%{http_code}' -X POST "https://huggingface.co/api/repos/create" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$SPACE_NAME\",\"organization\":\"$HF_USER\",\"type\":\"space\",\"sdk\":\"docker\",\"hardware\":\"cpu-basic\",\"private\":false}")"
BODY="$(printf '%s' "$RESP" | sed '$d')"
CODE="$(printf '%s' "$RESP" | tail -n1)"

case "$CODE" in
  200|201)
    echo "  Space olusturuldu: $SPACE_URL" ;;
  409|422)
    echo "  Space zaten var, devam ediliyor." ;;
  *)
    echo "  HTTP $CODE"
    printf '  %s\n' "$BODY" | head -c 600; echo
    echo ""
    echo "  >> Hugging Face ucretsiz hesaplarda Gradio/Docker Space icin PRO"
    echo "     aboneliği istiyor. Secenekler:"
    echo "       a) huggingface.co/pro ile PRO al (Space aynen bu haliyle yayinlanir)"
    echo "       b) publish_hf.sh --no-create ile sadece kodu Space repo'suna gonder"
    exit 1 ;;
esac

echo "[4/5] Push: $SPACE_URL"
GIT_TERMINAL_PROMPT=0 git push -q hf main && echo "  push: tamam"

echo "[5/5] Kaynak repo GitHub'a guncelleniyor"
git -C "$ROOT" add -A
git -C "$ROOT" commit -q -m "Add Docker Space support (Hugging Face)" || echo "  (degisiklik yok)"
git -C "$ROOT" push -q origin main && echo "  github: gonderildi"

echo "Tamamlandi -> $SPACE_URL"
