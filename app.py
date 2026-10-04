"""FaceSwap Web UI (Gradio).

Local web interface for the swap engine: upload a video + source photos,
analyze identities, preview the mapping and run the full high-quality swap.

By Ahmet Gedik — https://www.instagram.com/ahmetgedik67
"""
from __future__ import annotations

import contextlib
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2

try:
    import gradio as gr
except ImportError:
    print("\nGradio kurulu değil. Kurmak için:  pip install -r requirements.txt\n")
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("FACESWAP_WORK") or (ROOT / "workspace"))
UPLOADS = WORK / "uploads"
MODELS = Path(os.environ.get("FACESWAP_MODELS") or (ROOT / "models"))
for d in (WORK, UPLOADS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
import swap_engine as se  # noqa: E402
from download_models import ensure_models  # noqa: E402

CREDIT = "Made by **Ahmet Gedik** · [instagram.com/ahmetgedik67](https://www.instagram.com/ahmetgedik67)"


def models_ok() -> bool:
    return (MODELS / "inswapper_128.onnx").exists() and (MODELS / "gfpgan_1.4.onnx").exists()


def bootstrap_models() -> None:
    """Fetch models on first start (Hugging Face Spaces keep them out of the repo)."""
    if models_ok() and (MODELS / "face_parsing.onnx").exists():
        print("modeller hazir")
        return
    print("modeller bulunamadi, indiriliyor (ilk acilista birkac dakika surebilir)...")
    failed = ensure_models(MODELS)
    if failed:
        print("HATA: modeller indirilemedi:", ", ".join(failed))


class LogQueue:
    """Collect stdout lines from worker threads for streaming into the UI."""

    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()

    def write(self, s: str):
        for line in s.replace("\r", "\n").split("\n"):
            if line.strip():
                self.q.put(line)

    def flush(self):
        pass

    def drain(self) -> str:
        out = []
        try:
            while True:
                out.append(self.q.get_nowait())
        except queue.Empty:
            pass
        return "\n".join(out)


STOP = threading.Event()


def worker(log: LogQueue, fn, *args, **kwargs):
    """Run fn in a thread with stdout captured; return (ok, error)."""
    STOP.clear()
    result = {}

    def target():
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                fn(*args, **kwargs)
            result["ok"] = True
        except Exception as exc:  # noqa: BLE001
            result["ok"] = False
            result["err"] = str(exc)
        log.write("\n[done]" if result.get("ok") else f"\n[HATA] {result.get('err')}")

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def save_job(video, photos) -> Path:
    """Save the uploaded video + source photos into workspace/uploads/<stem>/."""
    if video is None:
        raise RuntimeError("Lütfen bir video yükleyin.")
    stem = Path(video).stem
    keep = "".join(c for c in stem if c.isalnum() or c in "-_ ").strip() or "video"
    jdir = UPLOADS / keep
    jdir.mkdir(parents=True, exist_ok=True)
    vdst = jdir / ("video.mp4")
    if Path(video).resolve() != vdst.resolve():
        shutil.copy2(video, vdst)
    if photos:
        for p in photos:
            src = Path(p)
            name = src.stem
            dst = jdir / f"{name}{src.suffix.lower()}"
            shutil.copy2(p, dst)
    return jdir


def job_video(jdir: Path) -> Path:
    return jdir / "video.mp4"


def pick_preview_frames(video: Path, n: int = 4) -> list[int]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    if total <= 0:
        return [0]
    return [max(0, int(total * f)) for f in (0.1, 0.35, 0.6, 0.85)][:n]


# ------------------------------------------------------------------ handlers
def ui_setup_check():
    lines = ["**Sistem Kontrolü**", ""]
    try:
        import onnxruntime as ort
        lines.append(f"- onnxruntime providers: `{', '.join(ort.get_available_providers())}`")
    except Exception as e:
        lines.append(f"- onnxruntime YOK: {e}")
    ff = shutil.which("ffmpeg")
    lines.append(f"- ffmpeg: `{'✓ ' + ff if ff else 'BULUNAMADI - PATH ekleyin'}`")
    lines.append(f"- inswapper_128.onnx: `{'✓' if (MODELS / 'inswapper_128.onnx').exists() else 'eksik'}`")
    lines.append(f"- gfpgan_1.4.onnx: `{'✓' if (MODELS / 'gfpgan_1.4.onnx').exists() else 'eksik'}`")
    lines.append(f"- face_parsing.onnx: `{'✓' if (MODELS / 'face_parsing.onnx').exists() else 'eksik (fast modda gerekmez)'}`")
    lines.append("")
    lines.append(CREDIT)
    return "\n".join(lines)


def ui_download_models(_log_text=None):
    if models_ok() and (MODELS / "face_parsing.onnx").exists():
        yield "Tüm modeller zaten indirilmiş ✓"
        return
    lines = []

    def log(msg):
        for line in str(msg).splitlines():
            if line.strip():
                lines.append(line)

    failed = ensure_models(MODELS, log=log)
    yield "\n".join(lines)
    yield ("HATA: " + ", ".join(failed)) if failed else "Tüm modeller hazır ✓"


def ui_analyze(video, photos, sample_every):
    jdir = save_job(video, photos)
    log = LogQueue()
    session = se.configure(str(jdir), str(job_video(jdir)), "")
    args = type("A", (), {"sample_every": int(sample_every)})()

    def run():
        app = se.FaceAnalysis(name="buffalo_l", providers=se.PROVIDERS, sess_options=se.session_options())
        app.prepare(ctx_id=0, det_size=(640, 640))
        session.app = app
        se.analyze(session, sample_every=args.sample_every)

    t = worker(log, run)
    while t.is_alive():
        time.sleep(0.7)
        yield log.drain(), None, gr.update()
    # results -> gallery + text
    gallery = []
    refs_dir = session.refs.with_suffix("")
    if refs_dir.exists():
        for img in sorted(refs_dir.glob("id*.jpg")):
            gallery.append((str(img), img.stem))
    yield log.drain(), gallery, None


def ui_preview(video, photos, map_by, min_count, match_thresh, quality):
    jdir = save_job(video, photos)
    names = [Path(p).stem for p in photos] if photos else []
    log = LogQueue()
    os.environ["FACESWAP_QUALITY"] = quality
    os.environ["FACESWAP_RESOLUTION"] = "source"
    args = se._build_ui_args(str(jdir), str(job_video(jdir)), map_by, int(min_count), float(match_thresh), names)
    frames = pick_preview_frames(job_video(jdir))

    def run():
        session = se.configure(str(jdir), str(job_video(jdir)), "")
        session.build_models()
        se.run_preview(session, args, frames, args.match_thresh)

    t = worker(log, run)
    while t.is_alive():
        time.sleep(0.7)
        yield log.drain(), None
    previews = jdir / "previews"
    gallery = []
    if previews.exists():
        for img in sorted(previews.glob("prev_*.jpg")):
            gallery.append((str(img), img.stem))
    yield log.drain(), gallery


def ui_run(video, photos, map_by, min_count, match_thresh, quality, resolution):
    jdir = save_job(video, photos)
    names = [Path(p).stem for p in photos] if photos else []
    log = LogQueue()
    os.environ["FACESWAP_QUALITY"] = quality
    os.environ["FACESWAP_RESOLUTION"] = resolution
    args = se._build_ui_args(str(jdir), str(job_video(jdir)), map_by, int(min_count), float(match_thresh), names)

    def run():
        session = se.configure(str(jdir), str(job_video(jdir)), "")
        session.build_models()
        session.stop_check = STOP.is_set  # graceful stop between frames
        se.run_video(session, args, args.match_thresh)

    t = worker(log, run)
    out = None
    while t.is_alive():
        time.sleep(1.0)
        yield log.drain(), out, gr.update(interactive=False)
    out_path = jdir / "video_swap_hq.mp4"
    out = str(out_path) if out_path.exists() else None
    yield log.drain(), out, gr.update(interactive=True)


def ui_stop():
    STOP.set()
    return "⏹ Durduruluyor... (mevcut kare bitince çıkar)"


# ------------------------------------------------------------------- layout
def build_ui():
    with gr.Blocks(title="FaceSwap — Ahmet Gedik") as demo:
        gr.Markdown(
            f"# 🎭 FaceSwap\n"
            f"Yüksek kaliteli video yüz değiştirme — inswapper + GFPGAN + anlamsal maske\n\n"
            f"{CREDIT}"
        )
        with gr.Tabs():
            with gr.Tab("⚙️ Kurulum"):
                check_btn = gr.Button("Sistem Kontrolü", variant="secondary")
                check_out = gr.Markdown()
                check_btn.click(ui_setup_check, None, check_out)
                dl_btn = gr.Button("⬇️ Modelleri İndir", variant="primary")
                dl_out = gr.Textbox(label="İndirme günlüğü", lines=8)
                dl_btn.click(ui_download_models, None, dl_out, show_progress="hidden")

            with gr.Tab("🎬 Yüz Değiştir"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_in = gr.File(label="Hedef video (.mp4)", file_types=[".mp4", ".mov", ".webm"])
                        photos_in = gr.File(
                            label="Kaynak yüz fotoğrafları (dosya adı = isim, ör. ahmet.jpeg)",
                            file_count="multiple", file_types=["image"],
                        )
                        map_by = gr.Dropdown(
                            ["top", "left", "first", "age"],
                            value="top",
                            label="Eşleme kuralı (map-by)",
                            info="top: en sık görünen ana konuşmacı · left: soldaki · first: ilk görünen · age: en yaşlı",
                        )
                        min_count = gr.Slider(1, 50, value=10, step=1, label="min-count (kimlik için min. örnek)")
                        match_thresh = gr.Slider(0.0, 0.6, value=0.30, step=0.05, label="match-thresh (eşik)")
                        quality = gr.Dropdown(["natural", "ultra", "fast"], value="natural", label="Kalite profili")
                        resolution = gr.Dropdown(["1080p", "720p", "source"], value="1080p", label="Çıktı çözünürlüğü")
                        with gr.Row():
                            analyze_btn = gr.Button("🔍 Analiz Et")
                            preview_btn = gr.Button("🖼 Önizle")
                            run_btn = gr.Button("▶️ Tam Swap", variant="primary")
                            stop_btn = gr.Button("⏹ Durdur", variant="stop")
                    with gr.Column(scale=2):
                        log_out = gr.Textbox(label="Günlük", lines=16)
                        gallery_out = gr.Gallery(label="Kimlikler / Önizleme", columns=4, height=320)
                        video_out = gr.Video(label="Sonuç video")

                analyze_btn.click(ui_analyze, [video_in, photos_in, min_count], [log_out, gallery_out, video_out])
                preview_btn.click(
                    ui_preview, [video_in, photos_in, map_by, min_count, match_thresh, quality],
                    [log_out, gallery_out],
                )
                run_btn.click(
                    ui_run, [video_in, photos_in, map_by, min_count, match_thresh, quality, resolution],
                    [log_out, video_out, run_btn],
                )
                stop_btn.click(ui_stop, None, log_out)

            with gr.Tab("⌨️ CLI (ileride)"):
                gr.Markdown(
                    "Terminal kullanmak isteyenler için örnek:\n\n"
                    "```bash\n"
                    "python swap_engine.py --stage analyze --clip-dir workspace/benim-klip\n"
                    "python swap_engine.py --stage preview --clip-dir workspace/benim-klip --sources ahmet\n"
                    "python swap_engine.py --stage run --clip-dir workspace/benim-klip --sources ahmet,erkan\n"
                    "```\n\n"
                    f"{CREDIT}"
                )
    return demo


def resolve_port(default: int = 7860) -> int:
    """Honour PORT / FACESWAP_PORT but ignore junk values such as PORT=0."""
    for key in ("PORT", "FACESWAP_PORT"):
        value = (os.environ.get(key) or "").strip()
        if value.isdigit() and 0 < int(value) < 65536:
            return int(value)
    return default


if __name__ == "__main__":
    print(se.banner())
    if os.environ.get("FACESWAP_AUTODOWNLOAD", "1") != "0":
        bootstrap_models()
    demo = build_ui()
    # Container ortaminda (HF Space, Docker, Oracle VM) 0.0.0.0'a baglanmali.
    on_space = bool(os.environ.get("SPACE_ID") or os.environ.get("HF_SPACE_ID"))
    in_container = Path("/.dockerenv").exists()
    server = os.environ.get("FACESWAP_SERVER") or os.environ.get("GRADIO_SERVER_NAME")
    if not server:
        server = "0.0.0.0" if (on_space or in_container) else "127.0.0.1"
    kwargs = {"server_name": server, "server_port": resolve_port(), "show_error": True}
    if int(gr.__version__.split(".")[0]) >= 6:  # Gradio 6 moved theme to launch()
        kwargs["theme"] = gr.themes.Soft()
    demo.queue(max_size=8).launch(**kwargs)
