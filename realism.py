"""Aligned multi-pass swap, restrained restoration and skin-only compositing.

Pixel boost processes interleaved subgrids of the SAME aligned face. It does
not turn INSwapper into a natively higher resolution or newly trained model.
No previous rendered face is blended into a new frame (avoids motion trails).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FFHQ_512 = np.array([[192.98138, 239.94708], [318.90277, 240.1936],
                     [256.63416, 314.01935], [201.26117, 371.41043],
                     [313.08905, 371.15118]], np.float32)


@dataclass
class Quality:
    pixel_boost: int = 256
    restore_weight: float = .35
    color_strength: float = .75
    texture_weight: float = .05
    semantic_mask: bool = True
    crf: int = 16

    def __post_init__(self):
        if self.pixel_boost not in (128, 256, 384, 512):
            raise ValueError('pixel_boost: 128, 256, 384 veya 512 olmalı')
        for name in ('restore_weight', 'color_strength', 'texture_weight'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name}: 0..1 olmalı')
        if not 0 <= self.crf <= 51:
            raise ValueError('crf: 0..51 olmalı')

    @classmethod
    def load(cls, preset=None, path=None):
        path = Path(path or Path(__file__).with_name('face_quality.json'))
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding='utf-8'))
        preset = preset or os.environ.get('FACESWAP_QUALITY') or data['default_preset']
        if preset not in data['presets']:
            raise ValueError(f'Bilinmeyen kalite profili: {preset}')
        return cls(**data['presets'][preset])


class LandmarkSmoother:
    """Relative motion thresholds: steady on still faces, responsive in motion."""
    def __init__(self, alpha=.35):
        self.alpha = alpha
        self.prev = None
        self.missed = 0

    def reset(self):
        self.prev = None
        self.missed = 0

    def update(self, kps):
        if kps is None:
            self.missed += 1
            self.prev = None
            return None  # never reuse stale coordinates over a hand or a cut
        kps = np.asarray(kps, np.float32)
        if kps.shape != (5, 2) or not np.isfinite(kps).all():
            self.reset()
            return None
        self.missed = 0
        if self.prev is None:
            self.prev = kps.copy()
            return self.prev.copy()
        scale = max(float(np.linalg.norm(kps[0] - kps[1])), 1.)
        old_scale = max(float(np.linalg.norm(self.prev[0] - self.prev[1])), 1.)
        motion = float(np.linalg.norm(kps - self.prev, axis=1).mean()) / scale
        if motion > .6 or not .65 < scale / old_scale < 1.55:
            alpha = 1.
        else:
            alpha = float(np.clip(self.alpha + motion * 3.5, self.alpha, .95))
        self.prev = self.prev * (1 - alpha) + kps * alpha
        return self.prev.copy()


class FrameTracker:
    """Separate tracks for simultaneous faces, reset at scene changes."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.index = 0
        self.thumbnail = None

    def begin_frame(self, frame):
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
        cut = False
        if self.thumbnail is not None:
            delta = np.mean(np.abs(thumb.astype(np.float32) - self.thumbnail))
            hist_a = cv2.calcHist([thumb], [0], None, [32], [0, 256])
            hist_b = cv2.calcHist([self.thumbnail], [0], None, [32], [0, 256])
            cut = delta > 50 or (delta > 28 and cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL) < .55)
        if cut:
            self.tracks.clear()
        self.thumbnail = thumb
        self.index += 1
        self.used = set()
        self.tracks = {k: v for k, v in self.tracks.items() if self.index - v['last'] <= 1}
        return cut

    def assign(self, face, identity):
        bbox = np.asarray(face.bbox, np.float32)
        center = (bbox[:2] + bbox[2:]) * .5
        scale = max(float(np.linalg.norm(bbox[2:] - bbox[:2])), 1.)
        candidates = []
        for key, track in self.tracks.items():
            if key in self.used or track['identity'] != identity:
                continue
            old_center = (track['bbox'][:2] + track['bbox'][2:]) * .5
            distance = float(np.linalg.norm(center - old_center)) / scale
            if distance < .65:
                candidates.append((distance, key))
        if candidates:
            key = min(candidates)[1]
        else:
            key = self.next_id
            self.next_id += 1
            self.tracks[key] = {'identity': identity, 'smoother': LandmarkSmoother(alpha=.35)}
        track = self.tracks[key]
        track.update(bbox=bbox.copy(), last=self.index)
        self.used.add(key)
        return key, track['smoother'].update(face.kps)


def align_matrix(kps, dst=FFHQ_512):
    from skimage.transform import SimilarityTransform
    if hasattr(SimilarityTransform, 'from_estimate'):
        transform = SimilarityTransform.from_estimate(np.asarray(kps, np.float32), dst)
    else:
        transform = SimilarityTransform()
        if not transform.estimate(np.asarray(kps, np.float32), dst):
            raise ValueError('Geçersiz yüz noktaları')
    if not np.isfinite(transform.params).all():
        raise ValueError('Geçersiz yüz noktaları')
    return transform.params[:2].astype(np.float32)


def warp(image, matrix, size=512):
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def soft_mask(binary, erosion=3, sigma=3.):
    binary = np.asarray(binary, np.uint8)
    if erosion:
        binary = cv2.erode(binary, np.ones((erosion, erosion), np.uint8))
    return np.clip(cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigma), 0, 1)


def geometric_mask(size=512):
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (size // 2, int(size * .57)),
                (int(size * .29), int(size * .35)), 0, 0, 360, 1, -1)
    return soft_mask(mask, erosion=5, sigma=6.)


def feature_protection(size=512):
    """Reduce GFPGAN influence around eyelids, lips and teeth."""
    mask = np.zeros((size, size), np.float32)
    for center, axes in (((.38, .47), (.085, .04)), ((.62, .47), (.085, .04)),
                         ((.5, .73), (.17, .085))):
        cv2.ellipse(mask, tuple(int(v * size) for v in center),
                    tuple(int(v * size) for v in axes), 0, 0, 360, 1., -1)
    return cv2.GaussianBlur(mask, (0, 0), size / 80)


def match_skin(image, reference, skin_mask, strength=.75, previous=None):
    """Bounded LAB correction using skin pixels only, never the background."""
    select = skin_mask > .8
    if np.count_nonzero(select) < 128 or strength == 0:
        return image, previous
    source = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    s, r = source[select], target[select]
    # Trim out glare/deep shadows. Clamp contrast to avoid washed-out faces.
    slo, shi = np.percentile(s[:, 0], [10, 90])
    rlo, rhi = np.percentile(r[:, 0], [10, 90])
    s = s[(s[:, 0] >= slo) & (s[:, 0] <= shi)]
    r = r[(r[:, 0] >= rlo) & (r[:, 0] <= rhi)]
    gain = np.clip(r.std(0) / np.maximum(s.std(0), 4.), .8, 1.2)
    # LAB chroma is centered on 128 in OpenCV. Scaling absolute A/B values
    # around zero introduces a visible blue/green cast when gain < 1.
    offset = np.clip(r.mean(0) - ((s.mean(0) - 128) * gain + 128),
                     [-24, -16, -16], [24, 16, 16])
    stats = np.stack([gain, offset])
    if previous is not None:
        # Keep large illumination changes responsive, damp small flicker.
        alpha = .8 if np.max(np.abs(previous[1] - offset)) > 10 else .4
        stats = previous * (1 - alpha) + stats * alpha
    corrected = (source - 128) * stats[0] + 128 + stats[1]
    corrected = source * (1 - strength) + corrected * strength
    return cv2.cvtColor(np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR), stats


def paste_face(frame, image, matrix, mask):
    height, width = frame.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    face = cv2.warpAffine(image, inverse, (width, height), flags=cv2.INTER_CUBIC)
    alpha = cv2.warpAffine(mask, inverse, (width, height), flags=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0, 1)[..., None]
    blended = np.clip(face.astype(np.float32) * alpha + frame * (1 - alpha), 0, 255).astype(np.uint8)
    blended[alpha[..., 0] <= 1e-6] = frame[alpha[..., 0] <= 1e-6]
    return blended


class RealisticSwapper:
    def __init__(self, swapper, gfpgan=None, quality=None, parser_path=None):
        self.swapper = swapper
        self.gfpgan = gfpgan
        self.quality = quality or Quality.load()
        self.parser = None
        self.color_history = {}
        self.latent_cache = {}
        if self.quality.semantic_mask:
            from face_parser import FaceParser, MODEL
            parser_path = Path(parser_path or MODEL)
            if not parser_path.exists():
                raise FileNotFoundError(f'Yüz maskesi modeli bulunamadı: {parser_path}; fast profili kullanılabilir')
            self.parser = FaceParser(parser_path)

    def reset(self):
        self.color_history.clear()

    def swap_crop(self, frame, kps, source_face):
        from insightface.utils.face_align import norm_crop2
        size = self.quality.pixel_boost
        native = self.swapper.input_size[0]
        if size % native or self.swapper.input_size[1] != native:
            raise ValueError('pixel_boost model çözünürlüğünün katı olmalı')
        _, matrix = norm_crop2(frame, kps, native)
        matrix = matrix.astype(np.float32) * (size / native)
        crop = warp(frame, matrix, size)
        embedding = np.asarray(source_face.normed_embedding, np.float32)
        key = embedding.tobytes()
        if key not in self.latent_cache:
            latent = np.dot(embedding.reshape(1, -1), self.swapper.emap)
            norm = np.linalg.norm(latent)
            if norm < 1e-8:
                raise ValueError('Geçersiz kaynak yüz vektörü')
            self.latent_cache[key] = (latent / norm).astype(np.float32)
        latent = self.latent_cache[key]
        factor = size // native
        swapped = np.empty_like(crop)
        for row in range(factor):
            for col in range(factor):
                tile = crop[row::factor, col::factor]
                blob = cv2.dnn.blobFromImage(tile, 1 / 255., (native, native), swapRB=True)
                output = self.swapper.session.run(self.swapper.output_names,
                          {self.swapper.input_names[0]: blob, self.swapper.input_names[1]: latent})[0][0]
                tile_result = np.clip(output.transpose(1, 2, 0)[:, :, ::-1] * 255., 0, 255).astype(np.uint8)
                swapped[row::factor, col::factor] = tile_result
        return swapped, matrix

    def masks(self, original):
        geometry = geometric_mask()
        if self.parser is None:
            return geometry, geometry, feature_protection()
        labels = self.parser.parse(original)
        face = np.isin(labels, [1, 2, 4, 5, 6, 7, 10, 11, 12]).astype(np.uint8)
        # If semantic segmentation fails, preserve the frame instead of painting
        # an ellipse over an uncertain face/occluder.
        if np.count_nonzero(face * (geometry > .5)) < 2000:
            return np.zeros_like(geometry), np.zeros_like(geometry), feature_protection()
        alpha = np.minimum(geometry, soft_mask(face, erosion=5, sigma=4.))
        # Keep original glasses, hair, ears, clothing and background intact.
        glasses = cv2.dilate((labels == 3).astype(np.uint8), np.ones((7, 7), np.uint8))
        alpha *= 1 - soft_mask(glasses, erosion=0, sigma=2.)
        skin = soft_mask(np.isin(labels, [1, 2]).astype(np.uint8), erosion=7, sigma=2.) * alpha
        features = soft_mask(np.isin(labels, [4, 5, 10, 11, 12]).astype(np.uint8), erosion=0, sigma=5.)
        return alpha, skin, np.maximum(features, feature_protection())

    def apply(self, frame, target_face, source_face, kps=None, track_id=None,
              restore_weight=None, match_gray=False):
        kps = np.asarray(target_face.kps if kps is None else kps, np.float32)
        matrix = align_matrix(kps)
        original = warp(frame, matrix)
        raw, swap_matrix = self.swap_crop(frame, kps, source_face)
        # Direct crop-to-crop transform; no low-res paste then recrop.
        transform = np.vstack([matrix, [0, 0, 1]]) @ np.vstack([cv2.invertAffineTransform(swap_matrix), [0, 0, 1]])
        raw = warp(raw, transform[:2].astype(np.float32))
        alpha, skin, protection = self.masks(original)
        if not np.any(alpha):
            return frame.copy()
        weight = self.quality.restore_weight if restore_weight is None else float(restore_weight)
        weight = float(np.clip(weight, 0, 1))
        enhanced = raw.astype(np.float32)
        if self.gfpgan is not None and weight > 0:
            restored = cv2.resize(self.gfpgan.enhance(raw), (512, 512), interpolation=cv2.INTER_CUBIC)
            restore_alpha = (weight * (1 - .85 * protection))[..., None]
            enhanced = enhanced * (1 - restore_alpha) + restored * restore_alpha
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        previous = self.color_history.get(track_id) if track_id is not None else None
        enhanced, stats = match_skin(enhanced, original, skin, self.quality.color_strength, previous)
        if track_id is not None and stats is not None:
            self.color_history[track_id] = stats
            if len(self.color_history) > 128:
                self.color_history.pop(next(iter(self.color_history)))
        # Match broad scene lighting without copying the actor's features.
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
        lighting = cv2.GaussianBlur(target_lab[..., 0] - lab[..., 0], (0, 0), 24.)
        lab[..., 0] += np.clip(lighting, -12, 12) * skin * .35
        enhanced = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        # Very weak scene texture only on skin, excluded from eyes/mouth.
        noise = original.astype(np.float32) - cv2.GaussianBlur(original.astype(np.float32), (0, 0), .7)
        texture_mask = (skin * (1 - protection))[..., None]
        enhanced = np.clip(enhanced + np.clip(noise, -6, 6) * texture_mask * self.quality.texture_weight, 0, 255).astype(np.uint8)
        if match_gray:
            enhanced = cv2.cvtColor(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return paste_face(frame, enhanced, matrix, alpha)
