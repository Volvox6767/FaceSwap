"""FaceSwap model downloader — Ahmet Gedik.

Downloads the three ONNX models into ./models:
  inswapper_128.onnx   (identity swap, ~554 MB)
  gfpgan_1.4.onnx      (face restoration, ~60 MB)
  face_parsing.onnx    (semantic mask, ~340 MB)
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

AUTHOR = "Ahmet Gedik"
AUTHOR_URL = "https://www.instagram.com/ahmetgedik67"

MODELS = Path(__file__).resolve().parent / "models"

URLS = {
    "inswapper_128.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
    "gfpgan_1.4.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx",
    "face_parsing.onnx": "https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model.onnx",
}


def progress(block: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(1.0, block * block_size / total)
    bar = "#" * int(done * 30)
    sys.stdout.write(f"\r  [{bar:<30}] {done:5.1%}")
    sys.stdout.flush()


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"\nFaceSwap model downloader — by {AUTHOR} ({AUTHOR_URL})\n")
    failed = []
    for name, url in URLS.items():
        dst = MODELS / name
        if dst.exists() and dst.stat().st_size > 1_000_000:
            print(f"[skip] {name} zaten var ({dst.stat().st_size // (1024 * 1024)} MB)")
            continue
        print(f"[indiriliyor] {name}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=progress)  # nosec - pinned asset URLs
            tmp.replace(dst)
            print(f"\n[ok] {name} -> {dst}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[HATA] {name}: {exc}")
            tmp.unlink(missing_ok=True)
            failed.append(name)
    if failed:
        print(f"\nTamamlanamayanlar: {', '.join(failed)}")
        print("Elle indirebilirsiniz — adresler README içinde.")
        return 1
    print("\nTüm modeller hazır ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
