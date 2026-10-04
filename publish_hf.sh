#!/usr/bin/env bash
# Publish FaceSwap to a Hugging Face Space (runtime model download).
# Usage: HF_TOKEN=hf_xxx bash publish_hf.sh [space-name]
set -euo pipefail
cd "$(dirname "$0")"

SPACE_NAME="${1:-FaceSwap}"
HF_USER="${HF_USER:-volvox67}"
ROOT="$(pwd)"
OUT="$ROOT/../FaceSwap-space"

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
cp "$ROOT/hf/README.md" "$OUT/README.md"
cp "$ROOT/hf/requirements.txt" "$OUT/requirements.txt"
printf 'workspace/\nmodels/\n__pycache__/\n*.pyc\n.pylibs/\n' > "$OUT/.gitignore"

echo "[2/5] GitHub'a gonderiliyor (kaynak repo guncel kalsin)"
git -C "$ROOT" add -A
git -C "$ROOT" commit -q -m "Add Hugging Face Space support (runtime model download)" || echo "  (degisiklik yok)"
git -C "$ROOT" push -q origin main && echo "  github: gonderildi"

echo "[3/5] Hugging Face bos Space olusturuluyor"
curl -s -X PUT "https://huggingface.co/api/repos/$HF_USER/$SPACE_NAME" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"type\":\"space\",\"sdk\":\"gradio\",\"hardware\":\"cpu-basic\"}" \
  | python -c "import sys,json;d=json.load(sys.stdin);print('  ',d.get('id') or d.get('error'))"

echo "[4/5] Kod Space'e gonderiliyor"
cd "$OUT"
git init -q -b main
git config user.name "Ahmet Gedik"
git config user.email "$HF_USER@users.noreply.huggingface.co"
git add -A
git commit -q -m "FaceSwap Space: Gradio UI + runtime model download"
git remote remove hf 2>/dev/null || true
git remote add hf "https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$SPACE_NAME"
GIT_TERMINAL_PROMPT=0 git push -q hf main && echo "  push: tamam"

echo "[5/5] Tamamlandi -> https://huggingface.co/spaces/$HF_USER/$SPACE_NAME"