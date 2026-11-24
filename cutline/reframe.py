"""Decide how to turn a landscape clip into 9:16: follow faces, or fit over a blurred background."""

from __future__ import annotations

import os
import statistics
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import ROOT
from .media import VideoInfo, ffmpeg_bin

TARGET_ASPECT = 9 / 16
SAMPLE_FPS = 2.0
SAMPLE_WIDTH = 480
MIN_FACE_HEIGHT = 0.09  # ignore faces smaller than this share of the frame (crowds, facecams)
KEYFRAMES_ONLY = True

_FACE_MODEL = ROOT / ".bin" / "face_detection_yunet_2023mar.onnx"
_FACE_MODEL_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
]


def _face_model() -> Path | None:
    """Download the small (230 KB) YuNet face detector once; None if offline."""
    if _FACE_MODEL.is_file() and _FACE_MODEL.stat().st_size > 100_000:
        return _FACE_MODEL
    _FACE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    for url in _FACE_MODEL_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cutline"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) > 100_000:
                tmp = _FACE_MODEL.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(_FACE_MODEL)
                return _FACE_MODEL
        except OSError:
            continue
    return None


@dataclass
class Layout:
    kind: str  # "crop" (follow subject), "fit" (letterbox over blur) or "vertical" (already portrait)
    crop_w: int = 0
    crop_h: int = 0
    keyframes: list[tuple[float, int]] = field(default_factory=list)  # (time in clip, crop x)
    face_ratio: float = 0.0

    def x_expr(self) -> str:
        """ffmpeg expression for the crop x position: hard cuts between camera positions."""
        kf = self.keyframes
        expr = str(kf[-1][1])
        for i in range(len(kf) - 2, -1, -1):
            expr = f"if(lt(t,{kf[i + 1][0]:.2f}),{kf[i][1]},{expr})"
        return expr


def _even(v: float) -> int:
    return int(v) // 2 * 2


def sample_faces(info: VideoInfo, start: float, duration: float) -> list[float | None]:
    """Horizontal centre (0..1) of the main face in frames sampled at SAMPLE_FPS (None = no face).

    Returns an empty list if the face model is unavailable.
    """
    model = _face_model()
    if model is None:
        return []
    w = SAMPLE_WIDTH
    h = _even(w * info.height / info.width)
    detector = cv2.FaceDetectorYN.create(str(model), "", (w, h), score_threshold=0.75)
    # Decoding only keyframes is several times faster than decoding every HD frame; the fps
    # filter repeats them to keep a steady sample rate. Plenty for following a face.
    skip = ["-skip_frame", "nokey"] if KEYFRAMES_ONLY else []
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-loglevel", "error", *skip,
        "-ss", f"{start:.3f}", "-i", str(info.path), "-t", f"{duration:.3f}",
        "-vf", f"fps={SAMPLE_FPS},scale={w}:{h}", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    frame_size = w * h * 3
    centres: list[float | None] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            _, faces = detector.detect(frame)
            best = None
            if faces is not None:
                big = [f for f in faces if f[3] >= h * MIN_FACE_HEIGHT]
                if big:
                    best = max(big, key=lambda f: f[2] * f[3])  # largest face = main subject
            centres.append(None if best is None else float(best[0] + best[2] / 2) / w)
    finally:
        proc.stdout.close()
        proc.wait()
    return centres


def plan_layout(info: VideoInfo, start: float, duration: float, mode: str = "auto") -> Layout:
    if info.width / info.height <= TARGET_ASPECT * 1.15:
        return Layout("vertical")
    if mode == "fit":
        return Layout("fit")

    crop_h = _even(info.height)
    crop_w = min(_even(crop_h * TARGET_ASPECT), _even(info.width))
    centres = sample_faces(info, start, duration)
    found = [c for c in centres if c is not None]
    ratio = len(found) / max(1, len(centres))

    # Few faces (screen recordings, gameplay, b-roll): a centre crop would lose the content.
    if mode == "auto" and ratio < 0.35:
        return Layout("fit", face_ratio=ratio)

    if not found:
        centres = [0.5] * max(1, len(centres))
    else:
        # Fill gaps with the last seen position (or the first one for leading gaps).
        last = found[0]
        filled = []
        for c in centres:
            last = c if c is not None else last
            filled.append(last)
        centres = filled

    max_x = info.width - crop_w
    xs = [min(max_x, max(0, int(c * info.width - crop_w / 2))) for c in centres]

    # Hold the camera still; only cut to a new position when the subject has clearly
    # moved there for at least `hold` samples (avoids jitter between two faces).
    hold = max(2, int(1.5 * SAMPLE_FPS))
    threshold = crop_w * 0.35
    current = int(statistics.median(xs[:hold]))
    keyframes = [(0.0, current)]
    pending = None
    for i, x in enumerate(xs):
        if abs(x - current) > threshold:
            pending = i if pending is None else pending
            if i - pending + 1 >= hold:
                new = int(statistics.median(xs[pending:i + 1]))
                if abs(new - current) > threshold:
                    current = new
                    keyframes.append((pending / SAMPLE_FPS, current))
                pending = None
        else:
            pending = None
    return Layout("crop", crop_w=crop_w, crop_h=crop_h, keyframes=keyframes, face_ratio=ratio)
