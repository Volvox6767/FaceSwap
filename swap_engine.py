"""FaceSwap — High-quality video face swap engine.

Pipeline: inswapper_128 identity swap + pixel-boost multi-pass +
GFPGAN restoration + semantic face parsing + LAB skin color transfer,
with per-identity landmark smoothing and reference-embedding matching.

Stages:
  analyze  sample frames, cluster identities, print stats / co-occurrence,
           save reference embeddings + face crops
  preview  swap a few frames and write preview images (no video output)
  run      process the whole video, mux original audio back (default)

By Ahmet Gedik — https://www.instagram.com/ahmetgedik67
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Warm up the DLL chain early: on locked-down Windows systems (Smart App
# Control) late imports of scipy/skimage after heavy ONNX DLL loads can be
# blocked intermittently. Loading them up front is fast and deterministic.
try:
    import ctypes  # noqa: F401
    import scipy  # noqa: F401
    from skimage import transform  # noqa: F401
except Exception:
    pass

from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

from runtime_utils import available_providers, create_session, session_options
from realism import RealisticSwapper, Quality, FrameTracker
from video_io import HighQualityVideoWriter

AUTHOR = "Ahmet Gedik"
AUTHOR_URL = "https://www.instagram.com/ahmetgedik67"

PROVIDERS = available_providers()

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
WORK_DIR = ROOT / "workspace"

FFHQ_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


def banner() -> str:
    return (
        "\n"
        "  ============================================================\n"
        "   FaceSwap  —  high-quality video face swap\n"
        f"   Author : {AUTHOR}\n"
        f"   Insta  : {AUTHOR_URL}\n"
        "  ============================================================\n"
    )


# ---------------------------------------------------------------- io helpers
def load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def save_bgr(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, image, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    buf.tofile(str(path))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


class Session:
    """Per-video paths and the models shared by all stages."""

    def __init__(self, clip_dir: Path, video: Path, refs: Path, previews: Path, out: Path):
        self.clip_dir = clip_dir
        self.video = video
        self.refs = refs
        self.previews = previews
        self.out = out
        self.app = None
        self.swapper = None
        self.gfpgan = None
        self.sources: dict[str, object] = {}
        self.stop_check = None  # optional callable -> True aborts the run after the current frame

    def build_models(self) -> None:
        print("onnxruntime providers:", ort.get_available_providers())
        app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS, sess_options=session_options())
        app.prepare(ctx_id=0, det_size=(640, 640))
        model_swap = MODELS_DIR / "inswapper_128.onnx"
        model_gfp = MODELS_DIR / "gfpgan_1.4.onnx"
        for m in (model_swap, model_gfp):
            if not m.exists() or m.stat().st_size < 100_000_000:
                raise RuntimeError(
                    f"model missing or incomplete: {m}\n"
                    f"Run install_models.(bat|sh) or download it manually (see README)."
                )
        swapper = INSwapper(str(model_swap), create_session(model_swap))
        gfpgan = GFPGAN(model_gfp)
        self.app, self.swapper, self.gfpgan = app, swapper, gfpgan


# ------------------------------------------------------------ HQ swap pieces
class GFPGAN:
    def __init__(self, model_path: Path):
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.resolution = (int(shape[-1]), int(shape[-2]))
        print("gfpgan input", self.input_name, shape, "providers", self.session.get_providers())

    def enhance(self, img: np.ndarray) -> np.ndarray:
        x = cv2.resize(img, self.resolution, interpolation=cv2.INTER_CUBIC)
        x = x.astype(np.float32)[:, :, ::-1] / 255.0
        x = (x.transpose(2, 0, 1) - 0.5) / 0.5
        x = np.expand_dims(x, 0).astype(np.float32)
        y = self.session.run(None, {self.input_name: x})[0][0]
        y = (y.transpose(1, 2, 0).clip(-1, 1) + 1) * 0.5
        y = (y * 255)[:, :, ::-1]
        return np.clip(y, 0, 255).astype(np.uint8)


def swap_frame(swapper, gfpgan, frame, target_face, source_face, kps,
               enhance_weight=None, match_gray=False, track_id=None):
    engine = getattr(gfpgan, "_realistic_engine", None)
    if engine is None or engine.swapper is not swapper:
        engine = RealisticSwapper(swapper, gfpgan,
                                  parser_path=Path(swapper.model_file).with_name("face_parsing.onnx"))
        gfpgan._realistic_engine = engine
    return engine.apply(frame, target_face, source_face, kps, track_id,
                        restore_weight=enhance_weight, match_gray=match_gray)


def load_source_face(app: FaceAnalysis, path: Path):
    image = load_bgr(path)
    faces = [f for f in app.get(image) if f.det_score >= 0.4]
    if not faces:
        for scale in (1.5, 2.0):
            big = cv2.resize(image, None, fx=scale, fy=scale)
            faces = [f for f in app.get(big) if f.det_score >= 0.4]
            if faces:
                break
    if not faces:
        raise RuntimeError(f"No face in source photo: {path}")
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    face = faces[0]
    print(f"source {path.name}: det={face.det_score:.3f} age={face.age:.0f} sex={getattr(face, 'sex', '?')}")
    return face


def crop_face(image: np.ndarray, face, pad: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad * 0.7))
    return image[y1:y2, x1:x2].copy()


# -------------------------------------------------------------- analyze stage
def analyze(session: Session, sample_every: int = 12, sim_thresh: float = 0.35) -> dict:
    app = session.app
    video = session.video
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"video {width}x{height} fps={fps:.3f} frames={total}")

    clusters: list[dict] = []
    cooc: dict[int, list[int]] = {}
    idx = 0
    sampled = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            sampled += 1
            seen_here = set()
            for face in [f for f in app.get(frame) if f.det_score >= 0.3]:
                best_i, best_sim = -1, -2.0
                for i, c in enumerate(clusters):
                    if i in seen_here:
                        continue
                    s = cosine_sim(face.normed_embedding, c["emb"])
                    if s > best_sim:
                        best_sim, best_i = s, i
                if best_i < 0 or best_sim < sim_thresh:
                    clusters.append(
                        {
                            "emb": face.normed_embedding.copy(),
                            "n": 1,
                            "ages": [float(face.age)],
                            "sex": str(getattr(face, "sex", "?")),
                            "scores": [float(face.det_score)],
                            "xs": [float(face.bbox[0])],
                            "ws": [float(face.bbox[2] - face.bbox[0])],
                            "first": idx,
                            "sample": frame.copy(),
                            "sample_face": face,
                        }
                    )
                    cid = len(clusters) - 1
                    seen_here.add(cid)
                else:
                    c = clusters[best_i]
                    n = c["n"]
                    c["emb"] = (c["emb"] * n + face.normed_embedding) / (n + 1)
                    c["n"] = n + 1
                    c["ages"].append(float(face.age))
                    c["scores"].append(float(face.det_score))
                    c["xs"].append(float(face.bbox[0]))
                    c["ws"].append(float(face.bbox[2] - face.bbox[0]))
                    if n + 1 >= 6:
                        c["sample"] = frame.copy()
                        c["sample_face"] = face
                    seen_here.add(best_i)
            for cid in seen_here:
                cooc.setdefault(cid, []).append(idx)
        idx += 1
    cap.release()

    print(f"sampled {sampled} frames -> {len(clusters)} identities")
    for i, c in enumerate(clusters):
        print(
            f"id{i}: n={c['n']} avg_age={np.mean(c['ages']):.0f} max_age={max(c['ages']):.0f} "
            f"sex={c['sex']} det={np.mean(c['scores']):.3f} avg_x={np.mean(c['xs']):.0f} "
            f"avg_w={np.mean(c['ws']):.0f} first_frame={c['first']}"
        )

    if len(clusters) > 1:
        print("pairwise identity similarity:")
        for i in range(len(clusters)):
            row = " ".join(f"{cosine_sim(clusters[i]['emb'], clusters[j]['emb']):+.2f}" for j in range(len(clusters)))
            print(f"  id{i}: {row}")

    print("co-occurrence (frames where two identities appear together):")
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            both = sorted(set(cooc.get(i, [])) & set(cooc.get(j, [])))
            if both:
                print(f"  id{i}+id{j}: {len(both)} sampled frames, e.g. {both[:6]}")

    refs_dir = session.refs.with_suffix("")
    for i, c in enumerate(clusters):
        save_bgr(refs_dir / f"id{i}_n{c['n']}_age{int(np.mean(c['ages']))}.jpg", crop_face(c["sample"], c["sample_face"]))

    np.savez(
        session.refs,
        embs=np.stack([c["emb"] for c in clusters]),
        meta=json.dumps(
            [
                {
                    "id": i,
                    "n": c["n"],
                    "avg_age": float(np.mean(c["ages"])),
                    "sex": c["sex"],
                    "first": c["first"],
                    "avg_x": float(np.mean(c["xs"])),
                }
                for i, c in enumerate(clusters)
            ]
        ),
    )
    print(f"saved refs -> {session.refs}, crops -> {refs_dir}")
    return {"clusters": clusters, "fps": fps, "width": width, "height": height, "total": total}


# --------------------------------------------------------------- mapping stage
def load_refs(session: Session) -> tuple[np.ndarray, list[dict]]:
    if not session.refs.exists():
        raise RuntimeError(f"refs not found, run --stage analyze first: {session.refs}")
    data = np.load(session.refs, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return data["embs"], meta


def source_path(session: Session, name: str) -> Path:
    """Source photo: prefer the clip folder, fall back to the project root / sources dir."""
    for base in (session.clip_dir, WORK_DIR, ROOT, ROOT / "sources"):
        for ext in (".jpeg", ".jpg", ".png"):
            p = base / f"{name}{ext}"
            if p.exists():
                return p
    raise RuntimeError(
        f"source photo not found: '{name}' in {session.clip_dir}, {WORK_DIR} or {ROOT}\n"
        f"Put {name}.jpeg next to the video or into {WORK_DIR}."
    )


def configure(clip_dir: str, video: str, out: str) -> Session:
    """Resolve the clip folder, target video, refs path and output path."""
    if clip_dir:
        cd = Path(clip_dir)
        if not cd.is_absolute():
            cd = WORK_DIR / cd
    else:
        cd = WORK_DIR
    if not cd.exists():
        raise RuntimeError(f"clip dir not found: {cd}")

    if video:
        vid = Path(video)
        if not vid.is_absolute() and not vid.exists():
            vid = cd / video
        if not vid.exists():
            raise RuntimeError(f"video not found: {vid}")
    else:
        suffixes = ("_swap_hq", "_hq", "_nosound", "_sonuc", "_swap")
        clips = [
            p
            for p in sorted(cd.glob("*.mp4"))
            if p.stat().st_size > 100_000 and not p.stem.endswith(suffixes)
        ]
        if len(clips) != 1:
            raise RuntimeError(f"pass --video, could not pick one in {cd}: {[c.name for c in clips]}")
        vid = clips[0]

    if out:
        outp = Path(out)
        if not outp.is_absolute():
            outp = cd / outp
    else:
        outp = cd / f"{vid.stem}_swap_hq.mp4"

    refs = cd / "refs.npz"
    previews = cd / "previews"
    suffixes = ("_swap_hq", "_hq", "_nosound", "_sonuc", "_swap")
    clips = [p for p in sorted(cd.glob("*.mp4")) if p.stat().st_size > 100_000 and not p.stem.endswith(suffixes)]
    if len(clips) != 1:
        refs = cd / f"{vid.stem}_refs.npz"
        previews = cd / f"{vid.stem}_previews"

    print(f"clip dir : {cd}")
    print(f"video    : {vid}")
    print(f"output   : {outp}")
    return Session(cd, vid, refs, previews, outp)


def _build_ui_args(clip_dir: str, video: str, map_by: str, min_count: int,
                   match_thresh: float, source_names: list[str] | None = None) -> types.SimpleNamespace:
    """Argument namespace for the Gradio UI (mirrors the CLI parser defaults)."""
    return types.SimpleNamespace(
        clip_dir=clip_dir, video=video, out="",
        sample_every=12, match_thresh=match_thresh, min_count=int(min_count),
        map="", map_by=map_by, only_top=0, map_dict={},
        frames="0,180,360,540,720,900,1100,1300",
        quality=None, max_frames=0,
        resolution=os.environ.get("FACESWAP_RESOLUTION", "1080p"),
        source_names=source_names or [],
    )


def build_mapping(meta: list[dict], min_count: int, map_by: str = "age", only_top: int = 0,
                  explicit: dict | None = None, source_names: list[str] | None = None) -> dict[int, str]:
    """Map significant video identities -> source names.

    map_by="age"   : oldest identity first (first source = the old man).
    map_by="first" : earliest-seen identity first (later arrivals get the
                     next source), for clips where a new person shows up in
                     the later scenes.
    map_by="left"  : leftmost identity first (avg bbox x), for two-person
                     clips where the user specifies "soldaki X, sağdaki Y".
    map_by="top"   : most frequent identity first (main speaker gets the
                     first source, e.g. "ana konuşmacı Ahmet").
    only_top>0     : swap only the N most frequent identities, leave the rest.
    With a single source every selected identity becomes that person.
    """
    source_names = source_names or []
    significant = [m for m in meta if m["n"] >= min_count]
    if not significant:
        raise RuntimeError(f"no identity with >= {min_count} samples; lower --min-count")
    if only_top:
        significant = sorted(significant, key=lambda m: -m["n"])[:only_top]
    if explicit:
        # user-specified per-identity mapping, e.g. {0: "erkan", 5: "ahmet"}
        mapping: dict[int, str] = {}
        for m in significant:
            name = explicit.get(m["id"])
            if name is None:
                continue
            mapping[m["id"]] = name
            print(f"  id{m['id']} (n={m['n']}, avg_age={m['avg_age']:.0f}, first=f{m['first']}, x={m['avg_x']:.0f})"
                  f" -> {name.upper()} (explicit)")
        unmapped = [m["id"] for m in meta if m["id"] not in mapping]
        if unmapped:
            print(f"  identities left untouched (background): {unmapped}")
        return mapping
    if map_by == "first":
        significant.sort(key=lambda m: (m["first"], -m["n"]))
    elif map_by == "left":
        significant.sort(key=lambda m: (m["avg_x"], -m["n"]))
    elif map_by == "top":
        significant.sort(key=lambda m: (-m["n"], m["first"]))
    else:
        significant.sort(key=lambda m: m["avg_age"], reverse=True)
    mapping = {}
    for rank, m in enumerate(significant):
        if not source_names:
            break
        name = source_names[min(rank, len(source_names) - 1)]
        mapping[m["id"]] = name
        reason = "oldest" if (map_by == "age" and rank == 0) else (
            f"first seen f{m['first']}" if map_by == "first" else (
                "leftmost" if (map_by == "left" and rank == 0) else (
                    "main speaker" if (map_by == "top" and rank == 0) else ""
                )
            )
        )
        print(
            f"  id{m['id']} (n={m['n']}, avg_age={m['avg_age']:.0f}, first=f{m['first']}, x={m['avg_x']:.0f})"
            f" -> {name.upper()}{' (' + reason + ')' if reason else ''}"
        )
    skipped = [m["id"] for m in meta if m["id"] not in mapping]
    if skipped:
        print(f"  identities left untouched (background): {skipped}")
    return mapping


def match_identity(face, refs: np.ndarray, thresh: float) -> tuple[int | None, float]:
    best, best_sim = -1, -2.0
    for i in range(len(refs)):
        s = cosine_sim(face.normed_embedding, refs[i])
        if s > best_sim:
            best_sim, best = s, i
    if best_sim < thresh:
        return None, best_sim
    return best, best_sim


def build_match_set(refs: np.ndarray, mapping: dict[int, str]) -> tuple[np.ndarray, list]:
    """Keep background references in matching to prevent false swaps."""
    return refs, [mapping.get(i) for i in range(len(refs))]


# ------------------------------------------------------------ preview stage
def face_at(image: np.ndarray, app: FaceAnalysis, near: np.ndarray):
    """Detect faces and return the one closest to the given bbox center."""
    cx = float((near[0] + near[2]) / 2)
    cy = float((near[1] + near[3]) / 2)
    faces = [f for f in app.get(image) if f.det_score >= 0.2]
    if not faces:
        return None
    return min(faces, key=lambda f: abs(float((f.bbox[0] + f.bbox[2]) / 2) - cx) + abs(float((f.bbox[1] + f.bbox[3]) / 2) - cy))


def run_preview(session: Session, args, frames: list[int], thresh: float) -> None:
    app, swapper, gfpgan = session.app, session.swapper, session.gfpgan
    refs, meta = load_refs(session)
    mapping = build_mapping(meta, args.min_count, args.map_by, args.only_top, args.map_dict, args.source_names)
    mrefs, mnames = build_match_set(refs, mapping)
    sources = {name: load_source_face(app, source_path(session, name)) for name in set(mapping.values())}

    cap = cv2.VideoCapture(str(session.video))
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            print(f"frame {fi}: read failed")
            continue
        faces = [f for f in app.get(frame) if f.det_score >= 0.25]
        tracker = FrameTracker()
        tracker.begin_frame(frame)
        if hasattr(gfpgan, "_realistic_engine"):
            gfpgan._realistic_engine.reset()
        out = frame
        applied = []
        for face in faces:
            cid, sim = match_identity(face, mrefs, thresh)
            name = None if cid is None else mnames[cid]
            print(f"frame {fi}: det={face.det_score:.2f} -> {name} sim={sim:.3f}")
            if cid is None or name is None:
                continue
            tid, kps = tracker.assign(face, cid)
            out = swap_frame(swapper, gfpgan, out, face, sources[name], kps, track_id=tid)
            applied.append((face, name))
        path = session.previews / f"prev_{fi:05d}.jpg"
        save_bgr(path, out)
        # verification: the swapped face must now look like its source photo
        for face, name in applied:
            got = face_at(out, app, face.bbox)
            if got is None:
                print(f"  frame {fi}: verify failed, no face after swap")
                continue
            sims = {n: cosine_sim(got.normed_embedding, s.normed_embedding) for n, s in sources.items()}
            best = max(sims, key=sims.get)
            print(
                f"  frame {fi}: swapped->{name} verify sim {sims} "
                f"best={best} {'OK' if best == name and sims[name] > 0.35 else 'CHECK'}"
            )
        print(f"  wrote {path}")
    cap.release()


# --------------------------------------------------------------- full run
def run_video(session: Session, args, thresh: float) -> None:
    app, swapper, gfpgan = session.app, session.swapper, session.gfpgan
    refs, meta = load_refs(session)
    mapping = build_mapping(meta, args.min_count, args.map_by, args.only_top, args.map_dict, args.source_names)
    if not mapping:
        raise RuntimeError("empty mapping: provide --sources or --map")
    mrefs, mnames = build_match_set(refs, mapping)
    sources = {name: load_source_face(app, source_path(session, name)) for name in set(mapping.values())}

    cap = cv2.VideoCapture(str(session.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {session.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"video {width}x{height} @ {fps:.3f} fps frames={total}")

    session.out.parent.mkdir(parents=True, exist_ok=True)
    writer = HighQualityVideoWriter(session.out, fps, (width, height), session.video, crf=Quality.load().crf)
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")

    tracker = FrameTracker()
    idx = 0
    swapped_n = 0
    skipped_faces = 0
    missed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if tracker.begin_frame(frame) and hasattr(gfpgan, "_realistic_engine"):
                gfpgan._realistic_engine.reset()
            faces = [f for f in app.get(frame) if f.det_score >= 0.25]
            matched = []
            for face in faces:
                cid, sim = match_identity(face, mrefs, thresh)
                if cid is None or mnames[cid] is None:
                    skipped_faces += 1
                    continue
                matched.append((face, cid))
            for face, cid in matched:
                tid, kps = tracker.assign(face, cid)
                if kps is None:
                    missed += 1
                    continue
                try:
                    frame = swap_frame(swapper, gfpgan, frame, face, sources[mnames[cid]], kps, track_id=tid)
                    swapped_n += 1
                except Exception as exc:
                    raise RuntimeError(f"frame {idx} swap failed: {exc}") from exc
            writer.write(frame)
            idx += 1
            if session.stop_check and session.stop_check():
                print("stopped by user")
                break
            if args.max_frames and idx >= args.max_frames:
                break
            if idx % 30 == 0 or idx == total:
                print(f"  {idx}/{total} swaps={swapped_n} skipped_faces={skipped_faces} missed={missed}", flush=True)
        writer.release()
    finally:
        cap.release()
        writer.abort()
    print(f"output: {session.out} frames={idx} swaps={swapped_n} skipped={skipped_faces} missed={missed}")
    print(banner().rstrip())


def main() -> int:
    global args
    # Windows console (cp1254/cp1252) cannot encode emoji in file paths
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print(banner())
    parser = argparse.ArgumentParser(
        prog="faceswap",
        description=f"High-quality video face swap — by {AUTHOR} ({AUTHOR_URL})",
    )
    parser.add_argument("--stage", choices=["analyze", "preview", "run"], default="run")
    parser.add_argument("--clip-dir", default="", help="folder with the video + source photos (default: workspace/)")
    parser.add_argument("--video", default="", help="target video (default: the only .mp4 in --clip-dir)")
    parser.add_argument("--sources", default="", help="comma separated source faces, e.g. 'ahmet' or 'umit,erkan'")
    parser.add_argument("--out", default="", help="output mp4 (default: <video>_swap_hq.mp4 in --clip-dir)")
    parser.add_argument("--sample-every", type=int, default=12)
    parser.add_argument("--match-thresh", type=float, default=0.30)
    parser.add_argument("--min-count", type=int, default=10, help="samples needed to treat an identity as main")
    parser.add_argument("--map", default="",
                        help='explicit per-identity mapping overriding --map-by, e.g. "0=erkan,1=umit,5=ahmet"')
    parser.add_argument("--map-by", choices=["age", "first", "left", "top"], default="top",
                        help="age: oldest; first: earliest-seen; left: leftmost; top: most frequent (main speaker)")
    parser.add_argument("--only-top", type=int, default=0,
                        help="swap only the N most frequent identities (0 = all significant)")
    parser.add_argument("--frames", default="0,180,360,540,720,900,1100,1300")
    parser.add_argument("--quality", choices=["natural", "ultra", "fast"], default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = full video; positive = short validation clip")
    parser.add_argument("--resolution", choices=["source", "720p", "1080p"], default="1080p")
    args = parser.parse_args()
    os.environ["FACESWAP_RESOLUTION"] = args.resolution
    if args.sample_every < 1 or args.min_count < 1 or args.max_frames < 0:
        parser.error("sample-every/min-count must be positive; max-frames must be nonnegative")
    if args.quality:
        os.environ["FACESWAP_QUALITY"] = args.quality
    args.source_names = [s.strip() for s in args.sources.split(",") if s.strip()]
    args.map_dict = {}
    if args.map:
        for part in args.map.split(","):
            if "=" not in part:
                raise RuntimeError(f"bad --map part (want id=name): {part}")
            k, v = part.split("=", 1)
            args.map_dict[int(k.strip())] = v.strip()

    session = configure(args.clip_dir, args.video, args.out)

    if args.stage == "analyze":
        app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS, sess_options=session_options())
        app.prepare(ctx_id=0, det_size=(640, 640))
        session.app = app
        analyze(session, sample_every=args.sample_every)
        return 0

    session.build_models()
    if args.stage == "preview":
        frames = [int(x) for x in args.frames.split(",") if x.strip()]
        run_preview(session, args, frames, args.match_thresh)
        return 0

    run_video(session, args, args.match_thresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
