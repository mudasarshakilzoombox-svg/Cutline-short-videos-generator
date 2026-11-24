"""Speech-to-text with word timestamps using faster-whisper (runs locally, free)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from .media import ffmpeg_bin

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SAMPLE_RATE = 16000


def load_audio(media_path: Path) -> np.ndarray:
    """Decode the audio track to 16 kHz mono float32 with ffmpeg."""
    proc = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-i", str(media_path), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"Could not read the audio track:\n{proc.stderr.decode(errors='replace')[-1500:]}")
    return np.frombuffer(proc.stdout, np.float32)


_MODELS: dict = {}


def _load_model(model_size: str):
    """Load a whisper model once per process (fast mode transcribes several short parts)."""
    if model_size in _MODELS:
        return _MODELS[model_size]
    from faster_whisper import WhisperModel

    model = None
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception:
        pass  # no usable GPU / CUDA libraries: fall back to CPU
    if model is None:
        model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    _MODELS.clear()  # keep at most one model in memory
    _MODELS[model_size] = model
    return model


def transcribe(
    media_path: Path,
    *,
    model_size: str = "small",
    language: str | None = None,
    on_progress: Callable[[float, float, float], None] | None = None,
) -> dict:
    """Transcribe a video/audio file.

    on_progress(fraction, seconds_done, total_seconds) is called after each segment.
    """
    audio = load_audio(media_path)
    model = _load_model(model_size)
    segments, info = model.transcribe(
        audio,
        language=language or None,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,  # avoids repetition loops on long videos
        # Fewer, cheaper retries when a window decodes badly (music under speech, noise).
        # The default (6 temperatures x 5 samples) can make such videos 5x slower.
        temperature=(0.0, 0.2, 0.4),
        best_of=2,
    )
    total = float(info.duration or 0)
    out = []
    for seg in segments:
        words = [
            {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip()}
            for w in (seg.words or [])
            if w.word.strip()
        ]
        text = seg.text.strip()
        if text:
            out.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text, "words": words})
        if on_progress and total:
            on_progress(min(1.0, seg.end / total), seg.end, total)
    return {
        "language": info.language,
        "duration": total,
        "model": model_size,
        "segments": out,
    }
