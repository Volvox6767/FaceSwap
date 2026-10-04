"""Single video encode from raw frames, original audio, atomic output."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np


def find_ffmpeg():
    binary = shutil.which('ffmpeg')
    if not binary:
        raise RuntimeError('FFmpeg bulunamadı; FFmpeg bin klasörünü PATH içine ekleyin')
    return binary


def output_filter(width, height, resolution='1080p'):
    if resolution == 'source':
        return 'pad=ceil(iw/2)*2:ceil(ih/2)*2,setsar=1'
    if resolution not in ('720p', '1080p'):
        raise ValueError('Çözünürlük: source, 720p veya 1080p olmalı')
    long_side, short_side = (1920, 1080) if resolution == '1080p' else (1280, 720)
    target_w, target_h = (long_side, short_side) if width >= height else (short_side, long_side)
    # Preserve already higher resolutions instead of discarding their detail.
    if width >= target_w and height >= target_h:
        return 'pad=ceil(iw/2)*2:ceil(ih/2)*2,setsar=1'
    return (f'scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,'
            f'pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,'
            'unsharp=5:5:0.25:5:5:0')


def source_rate(source, fallback):
    """Prefer the source rational timebase over an approximate OpenCV float."""
    probe = shutil.which('ffprobe')
    if not probe:
        return str(Fraction(float(fallback)).limit_denominator(1001))
    result = subprocess.run([probe, '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
                             '-of', 'json', str(source)], capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)['streams']
    if not streams:
        raise RuntimeError(f'Video akışı bulunamadı: {source}')
    data = streams[0]
    rate = Fraction(data.get('r_frame_rate', '0/1'))
    average = Fraction(data.get('avg_frame_rate', '0/1'))
    if rate <= 0:
        rate = Fraction(float(fallback)).limit_denominator(1001)
    if average > 0 and abs(float(average / rate) - 1) > .01:
        raise RuntimeError('Değişken kare hızı algılandı. Önce videoyu sabit kare hızına dönüştürün; ses zamanlaması korunmalı.')
    return str(rate)


class HighQualityVideoWriter:
    def __init__(self, output, fps, size, audio_source, crf=16, resolution=None, audio_start=0):
        self.output = Path(output)
        source = Path(audio_source)
        if self.output.resolve() == source.resolve():
            raise ValueError('Çıktı dosyası kaynak videonun üstüne yazılamaz')
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = (int(v) for v in size)
        if min(self.width, self.height) <= 0 or fps <= 0:
            raise ValueError('Geçersiz video boyutu veya kare hızı')
        self.count = 0
        self.closed = False
        self.log = tempfile.TemporaryFile()
        temp = tempfile.NamedTemporaryFile(prefix=self.output.stem + '_', suffix='.partial.mp4',
                                          dir=self.output.parent, delete=False)
        self.partial = Path(temp.name)
        temp.close()
        try:
            rate = source_rate(source, fps)
            command = [find_ffmpeg(), '-hide_banner', '-loglevel', 'error', '-y',
                       '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s:v', f'{self.width}x{self.height}',
                       '-r', rate, '-i', 'pipe:0']
            if audio_start:
                command += ['-ss', str(float(audio_start))]
            command += ['-i', str(source),
                       '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'libx264', '-preset', 'slow',
                       '-crf', str(crf), '-vf', output_filter(self.width, self.height, resolution or os.environ.get('FACESWAP_RESOLUTION', '1080p')),
                       '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                       '-af', 'apad', '-movflags', '+faststart', '-shortest', str(self.partial)]
            # apad makes -shortest stop at VIDEO duration, even if audio ends
            # early. Without any audio map, all video frames are still kept.
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=self.log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            self.partial.unlink(missing_ok=True)
            self.log.close()
            raise

    def isOpened(self):
        return not self.closed and self.process.poll() is None

    def error(self):
        self.log.seek(0)
        return self.log.read().decode('utf-8', errors='replace')[-3000:]

    def write(self, frame):
        if self.closed:
            raise RuntimeError('Video yazıcısı kapalı')
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError('Video karesi uint8 BGR ve sabit boyutta olmalı')
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
            self.count += 1
        except (BrokenPipeError, OSError) as exc:
            self.abort()
            raise RuntimeError(f'Video kodlama başarısız: {self.last_error}') from exc

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            code = self.process.wait(timeout=120)
            self.last_error = self.error()
            if code != 0 or self.count == 0:
                raise RuntimeError(f'Video kodlama başarısız ({code}): {self.last_error}')
            self.partial.replace(self.output)
        except Exception:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.partial.unlink(missing_ok=True)
            raise
        finally:
            self.log.close()

    def abort(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.last_error = self.error()
        self.process.stdin.close()
        self.partial.unlink(missing_ok=True)
        self.log.close()

    def __del__(self):
        if not getattr(self, 'closed', True) and hasattr(self, 'process'):
            try:
                self.abort()
            except Exception:
                pass
