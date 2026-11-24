"""Cut, reframe to 1080x1920, burn captions, normalise audio, export MP4 + thumbnail."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .captions import OUT_H, OUT_W
from .media import VideoInfo, run_ffmpeg
from .reframe import Layout


def video_filter(layout: Layout, ass_name: str | None) -> str:
    subs = f",ass={ass_name}" if ass_name else ""
    if layout.kind == "crop":
        return (
            f"[0:v]crop={layout.crop_w}:{layout.crop_h}:x='{layout.x_expr()}':y=0,"
            f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1{subs}[v]"
        )
    if layout.kind == "vertical":
        return (
            f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1{subs}[v]"
        )
    # "fit": full frame in the middle, blurred zoomed copy filling the background.
    return (
        "[0:v]split=2[bgsrc][fgsrc];"
        "[bgsrc]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=10:2,"
        f"scale={OUT_W}:{OUT_H},eq=brightness=-0.12,setsar=1[bg];"
        f"[fgsrc]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos,setsar=1[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2{subs}[v]"
    )


def render_clip(
    info: VideoInfo,
    start: float,
    end: float,
    out_path: Path,
    layout: Layout,
    *,
    ass_path: Path | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    duration = end - start
    workdir = ass_path.parent if ass_path else out_path.parent
    # ffmpeg filter arguments choke on Windows drive letters, so the subtitle file is
    # referenced by bare name and ffmpeg runs inside its folder.
    args = [
        "-ss", f"{start:.3f}", "-i", str(info.path.resolve()), "-t", f"{duration:.3f}",
        "-filter_complex", video_filter(layout, ass_path.name if ass_path else None),
        "-map", "[v]",
    ]
    if info.has_audio:
        fade_out = max(0.0, duration - 0.15)
        args += [
            "-map", "0:a:0",
            "-af", f"afade=t=in:st=0:d=0.04,afade=t=out:st={fade_out:.3f}:d=0.15,loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        ]
    args += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_path.resolve()),
    ]
    run_ffmpeg(args, cwd=workdir, duration=duration, on_progress=on_progress)


def make_thumbnail(clip_path: Path, out_path: Path, t: float = 1.0) -> None:
    run_ffmpeg(["-ss", f"{t:.2f}", "-i", str(clip_path.resolve()), "-frames:v", "1", "-q:v", "2",
                str(out_path.resolve())])
