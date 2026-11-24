"""ffmpeg discovery, probing, frame grabbing and URL downloads."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import ROOT

_FFMPEG: str | None = None


def _supports_required_features(binary: str) -> bool:
    """True if this ffmpeg can burn ASS subtitles (libass) and encode H.264 (libx264)."""
    try:
        filters = subprocess.run(
            [binary, "-hide_banner", "-filters"], capture_output=True, timeout=30
        ).stdout.decode(errors="replace")
        encoders = subprocess.run(
            [binary, "-hide_banner", "-encoders"], capture_output=True, timeout=30
        ).stdout.decode(errors="replace")
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(r"\bass\s+V->V", filters) is not None and "libx264" in encoders


def _bundled_ffmpeg() -> str:
    """Copy the ffmpeg binary shipped with imageio-ffmpeg to .bin/ffmpeg(.exe).

    The copy gets the plain name "ffmpeg" so yt-dlp can find and use it too.
    """
    import imageio_ffmpeg

    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    local = ROOT / ".bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not local.exists() or local.stat().st_size != bundled.stat().st_size:
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundled, local)
    return str(local)


def ffmpeg_bin() -> str:
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    candidates = []
    system = shutil.which("ffmpeg")
    if system:
        candidates.append(system)
    try:
        candidates.append(_bundled_ffmpeg())
    except Exception:  # imageio-ffmpeg missing or broken
        pass
    if not candidates:
        raise RuntimeError("ffmpeg not found. Run: pip install imageio-ffmpeg (or install ffmpeg).")
    _FFMPEG = next((c for c in candidates if _supports_required_features(c)), candidates[0])
    return _FFMPEG


def run_ffmpeg(
    args: list[str],
    *,
    cwd: Path | None = None,
    duration: float | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """Run ffmpeg, optionally reporting progress (0..1) based on output time."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y"]
    if on_progress and duration:
        cmd += ["-progress", "pipe:1", "-nostats"]
    cmd += args
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL
        )
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                if not (on_progress and duration):
                    continue
                line = raw.decode(errors="replace").strip()
                if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                    try:
                        us = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    on_progress(max(0.0, min(1.0, us / 1_000_000 / duration)))
        except BaseException:
            proc.kill()  # e.g. the user pressed Stop (raised from on_progress)
            proc.wait()
            raise
        proc.wait()
        if proc.returncode != 0:
            err.seek(0)
            msg = err.read().decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}):\n{msg[-3000:]}")


@dataclass
class VideoInfo:
    path: Path
    duration: float
    width: int  # displayed width (after rotation metadata is applied)
    height: int
    has_audio: bool


def grab_frame(path: Path, t: float = 0.0, width: int | None = None) -> np.ndarray:
    """Decode one frame at time t as a BGR image (ffmpeg applies rotation metadata)."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(path),
           "-frames:v", "1"]
    if width:
        cmd += ["-vf", f"scale={width}:-2"]
    cmd += ["-f", "image2pipe", "-c:v", "png", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
    img = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR) if proc.stdout else None
    if img is None:
        raise RuntimeError(f"Could not read a video frame from {path}:\n{proc.stderr.decode(errors='replace')[-1000:]}")
    return img


def _is_cover_art(stream) -> bool:
    """Album-art 'video' streams in audio files are not real video."""
    try:
        import av

        return bool(stream.disposition & av.stream.Disposition.attached_pic)
    except Exception:
        return False


def probe(path: Path) -> VideoInfo:
    import av

    with av.open(str(path)) as container:
        video = [s for s in container.streams if s.type == "video" and not _is_cover_art(s)]
        has_audio = any(s.type == "audio" for s in container.streams)
        duration = container.duration / av.time_base if container.duration else 0.0
        if not duration and video and video[0].duration:
            duration = float(video[0].duration * video[0].time_base)
    if not video:
        raise ValueError(f"{path.name} has no video track. Cutline needs a video file.")
    if duration <= 0:
        raise ValueError(f"Could not determine the duration of {path.name}.")
    frame = grab_frame(path, t=min(1.0, duration / 2))
    h, w = frame.shape[:2]
    return VideoInfo(path=path, duration=duration, width=w, height=h, has_audio=has_audio)


def is_url(text: str) -> bool:
    return re.match(r"^https?://", text.strip(), re.IGNORECASE) is not None


COOKIES_FILE = ROOT / "cookies.txt"

COOKIE_STEPS = """   a. In Chrome, install the "Get cookies.txt LOCALLY" extension, and in chrome://extensions open
      its Details and turn on "Allow in Incognito".
   b. Open a new Incognito window, sign in to YouTube there, then go to youtube.com/robots.txt.
   c. Click the extension and Export, then CLOSE the Incognito window right away
      (that stops YouTube from replacing these cookies, so they keep working).
   d. In Cutline, load the file under Advanced > "YouTube cookies file"."""

BOT_CHECK_HELP = f"""YouTube blocked the download ("Sign in to confirm you're not a bot").
This is YouTube limiting downloads from your internet connection, not a Cutline bug. Fixes:

1. Download the video yourself and use "Upload a video" instead of the link.
2. Let Cutline use a YouTube login:
{COOKIE_STEPS}
3. Wait a few hours (the block is often temporary) and turn off any VPN."""

STALE_COOKIES_HELP = f"""YouTube blocked the download ("Sign in to confirm you're not a bot") even with your cookies file.
YouTube has most likely replaced those cookies (it does this when you keep using YouTube in the
browser you exported them from). Export a fresh set the way that keeps working:
{COOKIE_STEPS}
Or download the video yourself and use "Upload a video"."""


def save_cookies(src: Path) -> str:
    """Validate an exported cookies.txt and install it as <project>/cookies.txt. Returns a status message."""
    text = Path(src).read_text(encoding="utf-8", errors="replace")
    rows = [l for l in text.splitlines() if l.strip() and not l.startswith("#") and l.count("\t") >= 6]
    if not rows:
        raise ValueError("That file isn't a cookies.txt export (Netscape format). Export it with the "
                         '"Get cookies.txt LOCALLY" extension while on youtube.com.')
    if not any("youtube.com" in l for l in rows):
        raise ValueError("That cookies file has no youtube.com cookies. Open youtube.com (signed in) "
                         "and export again from that tab.")
    COOKIES_FILE.write_text(text, encoding="utf-8")
    return f"YouTube cookies saved ({len(rows)} cookies). Links will now download with your login."


class _QuietLogger:
    """Keep yt-dlp's own messages out of the console; errors still surface as exceptions."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


def _js_runtimes() -> dict:
    """YouTube needs a JavaScript runtime to solve its download challenge.

    Prefer the Deno binary installed with yt-dlp[deno] (works regardless of the system's
    Node version; yt-dlp needs Node >= 23.5), then anything on PATH.
    """
    try:
        import deno

        path = deno.find_deno_bin()
        if path and Path(path).is_file():
            return {"deno": {"path": str(path)}}
    except Exception:
        pass
    for name in ("deno", "node", "bun"):
        path = shutil.which(name)
        if path:
            return {name: {"path": path}}
    return {"deno": {}}


# Best quality up to 1080p, preferring H.264: YouTube's AV1 and VP9 streams take 5x longer to
# decode on a laptop CPU, which slows face tracking and rendering. m4a audio keeps merging a remux.
VIDEO_FORMAT = ("bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]"
                "/bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b")


def run_ytdlp(fn: Callable, cookies_file: Path | None = None, **extra_opts):
    """Run fn(ydl) with Cutline's yt-dlp settings and its fallbacks for YouTube trouble.

    Uses cookies_file, or <project>/cookies.txt if present, for sites that require a login.
    """
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    cookies = cookies_file or (COOKIES_FILE if COOKIES_FILE.is_file() else None)
    opts = {
        "noplaylist": True,
        "ffmpeg_location": ffmpeg_bin(),
        "quiet": True,
        "no_warnings": True,
        "logger": _QuietLogger(),
        "noprogress": True,
        "js_runtimes": _js_runtimes(),
        **extra_opts,
    }
    if cookies:
        opts["cookiefile"] = str(cookies)

    # When YouTube changes its main player script, yt-dlp's challenge solver can break and only
    # preview images come back ("Requested format is not available"). Other builds of the
    # player usually still work, so fall back to them.
    variants = [None, "es6", "tv"]
    attempt = 0
    locked_retries = 0
    while True:
        variant = variants[attempt]
        run_opts = dict(opts)
        if variant:
            run_opts["extractor_args"] = {"youtube": {"player_js_variant": [variant]}}
        try:
            with YoutubeDL(run_opts) as ydl:
                return fn(ydl)
        except DownloadError as e:
            message = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).replace("ERROR: ", "").strip()
            if re.search(r"not a bot|sign in to confirm", message, re.IGNORECASE):
                raise RuntimeError(STALE_COOKIES_HELP if cookies else BOT_CHECK_HELP) from e
            if "Requested format is not available" in message and attempt + 1 < len(variants):
                attempt += 1
                continue
            # Windows: antivirus (or another download of the same video) briefly locking the file.
            if re.search(r"WinError 32|being used by another process", message) and locked_retries < 3:
                locked_retries += 1
                time.sleep(10)
                continue
            if "WinError 32" in message or "being used by another process" in message:
                raise RuntimeError("Windows wouldn't let Cutline finish saving the download because another "
                                   "program is using the file (often antivirus scanning it, or the same video "
                                   "downloading twice). Wait a minute and click Make shorts again.") from e
            raise RuntimeError(f"Could not download the video: {message}\n\n"
                               "If the link is correct, try downloading the video yourself and uploading it."
                               ) from e


def download(
    url: str,
    dest_dir: Path,
    on_progress: Callable[[float], None] | None = None,
    cookies_file: Path | None = None,
) -> tuple[Path, dict]:
    """Download a whole video with yt-dlp. Returns (file path, yt-dlp info dict)."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    def hook(d: dict) -> None:
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                on_progress(min(1.0, d.get("downloaded_bytes", 0) / total))

    def fetch(ydl) -> tuple[Path, dict]:
        info = ydl.extract_info(url.strip(), download=True)
        if info.get("entries"):
            info = next(e for e in info["entries"] if e)
        downloads = info.get("requested_downloads") or []
        return (Path(downloads[0]["filepath"]) if downloads else Path(ydl.prepare_filename(info))), info

    path, info = run_ytdlp(
        fetch, cookies_file, format=VIDEO_FORMAT, merge_output_format="mp4",
        outtmpl=str(dest_dir / "%(title).80B [%(id)s].%(ext)s"), restrictfilenames=True,
        progress_hooks=[hook],
    )
    if not path.exists():
        raise RuntimeError(f"Download finished but the file was not found: {path}")
    return path, info
