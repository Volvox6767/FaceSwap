"""Face parsing via SegFormer ONNX (jonathandinu/face-parsing, 19 classes).

Class ids (CelebAMask-HQ):
0 background, 1 skin, 2 nose, 3 eye_g, 4 l_eye, 5 r_eye, 6 l_brow, 7 r_brow,
8 l_ear, 9 r_ear, 10 mouth, 11 u_lip, 12 l_lip, 13 hair, 14 hat, 15 ear_r,
16 neck_l, 17 neck, 18 cloth

head mask = everything except background (and optionally except cloth/neck)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from runtime_utils import create_session, available_providers

PROVIDERS = available_providers()
MODEL = Path(__file__).with_name("models") / "face_parsing.onnx"
SIZE = 512
BACKGROUND = 0
HAIR = 13
HAT = 14


class FaceParser:
    def __init__(self, model_path: Path | None = None):
        path = model_path or MODEL
        # The local model is quantized SegFormer; CPU handles its integer
        # operators without fragmenting the graph into DirectML fallbacks.
        self.session = create_session(path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        print("face_parser providers:", self.session.get_providers())

    def parse(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return per-pixel class ids at the input image resolution."""
        x = cv2.resize(image_bgr, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        y = self.session.run(None, {self.input_name: x})[0]
        if y.ndim == 3:  # (1, H, W) logits already argmaxed upstream
            labels = y[0].astype(np.int32)
        else:  # (1, C, H, W)
            # Upsample logits BEFORE argmax; resizing labels gives blocky
            # hairlines and unstable contours at the low-resolution boundary.
            logits = cv2.resize(y[0].transpose(1, 2, 0),
                                (image_bgr.shape[1], image_bgr.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
            labels = logits.argmax(axis=2).astype(np.int32)
        return cv2.resize(labels, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    def head_mask(self, image_bgr: np.ndarray, include_shoulders: bool = True) -> np.ndarray:
        """Binary uint8 mask (255) of the whole head: hair, face, ears, hat,
        neck (and optionally shoulders/cloth)."""
        labels = self.parse(image_bgr)
        if include_shoulders:
            head = labels != BACKGROUND
        else:
            head = (labels != BACKGROUND) & (labels != 17) & (labels != 18)
        # smooth + fill
        head = (head * 255).astype(np.uint8)
        head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        head = cv2.morphologyEx(head, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return head

    def hair_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        labels = self.parse(image_bgr)
        hair = ((labels == HAIR) | (labels == HAT)).astype(np.uint8) * 255
        return cv2.morphologyEx(hair, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


if __name__ == "__main__":
    import sys

    parser = FaceParser()
    img = cv2.imdecode(np.fromfile(sys.argv[1], dtype=np.uint8), cv2.IMREAD_COLOR)
    labels = parser.parse(img)
    hist = np.bincount(labels.flatten(), minlength=19)
    for cid in range(19):
        if hist[cid]:
            print(f"class {cid:2d}: {hist[cid] / labels.size:.2%}")
    hm = parser.head_mask(img)
    vis = img.copy()
    vis[hm == 0] //= 4
    ok, buf = cv2.imencode(".jpg", vis)
    Path("output").mkdir(exist_ok=True)
    buf.tofile("output/parser_test.jpg")
    print("wrote output/parser_test.jpg")
