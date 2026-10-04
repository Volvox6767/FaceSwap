"""FaceSwap model downloader — Ahmet Gedik.

Downloads the three ONNX models into ./models:
  inswapper_128.onnx   (identity swap, ~554 MB)
  gfpgan_1.4.onnx      (face restoration, ~60 MB)
  face_parsing.onnx    (semantic mask, ~340 MB)

`ensure_models()` is reused by app.py (local UI and Hugging Face Spaces, where
the models are fetched on first start instead of living in the repo).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

AUTHOR = "Ahmet Gedik"
AUTHOR_URL = "https://www.instagram.com/ahmetgedik67"

ROOT = Path(__file__).resolve().parent

URLS = {
    "inswapper_128.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
    "gfpgan_1.4.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx",
    "face_parsing.onnx": "https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model.onnx",
}


def _progress(label: str):
    def hook(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(1.0, block * block_size / total)
        bar = "#" * int(done * 30)
        sys.stdout.write(f"\r  {label} [{bar:<30}] {done:5.1%}")
        sys.stdout.flush()
    return hook


def ensure_models(models_dir: Path | None = None, log=print) -> list[str]:
    """Download missing models. Returns the list of files that failed."""
    target = Path(models_dir or ROOT / "models")
    target.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    log(f"FaceSwap model downloader — by {AUTHOR} ({AUTHOR_URL})\n")
    for name, url in URLS.items():
        dst = target / name
        if dst.exists() and dst.stat().st_size > 1_000_000:
            log(f"[skip] {name} zaten var ({dst.stat().st_size // (1024 * 1024)} MB)")
            continue
        log(f"[indiriliyor] {name} (~{'-' if name != 'gfpgan_1.4.onnx' else '60'} MB)")
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=_progress(name))  # nosec - pinned asset URLs
            tmp.replace(dst)
            log(f"\n[ok] {name} -> {dst} ({dst.stat().st_size // (1024 * 1024)} MB)")
        except Exception as exc:  # noqa: BLE001
            log(f"\n[HATA] {name}: {exc}")
            tmp.unlink(missing_ok=True)
            failed.append(name)
    return failed


def main() -> int:
    failed = ensure_models()
    if failed:
        print(f"\nTamamlanamayanlar: {', '.join(failed)}")
        print("Elle indirebilirsiniz — adresler README içinde.")
        return 1
    print("\nTüm modeller hazır ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())