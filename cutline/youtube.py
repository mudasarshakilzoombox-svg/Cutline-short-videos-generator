"""Fast mode for YouTube links: use YouTube's own subtitles and "Most replayed" data,
then download only the parts of the video that become shorts."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

from .analyze import fmt_ts
from .media import VIDEO_FORMAT, run_ffmpeg, run_ytdlp

_YOUTUBE = re.compile(r"^https?://([a-z0-9-]+\.)?(youtube\.com|youtu\.be|youtube-nocookie\.com)/", re.IGNORECASE)


def is_youtube(url: str) -> bool:
    return bool(_YOUTUBE.match(url.strip()))


def _pick_caption_track(info: dict) -> tuple[str, list[dict]] | None:
    """The best subtitle track in the video's own language (never a machine translation)."""
    subs = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
    autos = info.get("automatic_captions") or {}
    lang = (info.get("language") or "").split("-")[0].lower() or None
    if lang:  # subtitles the creator uploaded, in the video's language
        for key, formats in subs.items():
            if key.split("-")[0].lower() == lang:
                return key, formats
    for key, formats in autos.items():  # YouTube's speech recognition, untranslated
        if key.endswith("-orig"):
            return key, formats
    if lang and lang in autos:
        return lang, autos[lang]
    if subs:
        return next(iter(subs.items()))
    return None


def _has_word_timing(data: dict) -> bool:
    """True for YouTube's automatic captions, which time every word (uploaded subtitles time lines)."""
    segs = [s for e in data.get("events") or [] for s in (e.get("segs") or []) if (s.get("utf8") or "").strip()]
    timed = sum(1 for s in segs if s.get("tOffsetMs"))
    return bool(segs) and timed / len(segs) > 0.3


def _parse_json3(data: dict) -> list[dict]:
    """YouTube json3 captions -> transcript lines with word timings (exact for automatic
    captions, spread evenly across each line for uploaded subtitles)."""
    lines = []
    for event in data.get("events") or []:
        segs = event.get("segs")
        if not segs or "tStartMs" not in event:
            continue
        start = event["tStartMs"] / 1000
        end = start + event.get("dDurationMs", 0) / 1000
        pieces = []
        for i, seg in enumerate(segs):
            text = (seg.get("utf8") or "").replace("\n", " ").strip()
            if not text:
                continue
            seg_start = start + seg.get("tOffsetMs", 0) / 1000
            nxt = next((s for s in segs[i + 1:] if (s.get("utf8") or "").strip()), None)
            seg_end = start + nxt.get("tOffsetMs", 0) / 1000 if nxt and "tOffsetMs" in nxt else end
            pieces.append((seg_start, max(seg_end, seg_start + 0.05), text))
        words = []
        for seg_start, seg_end, text in pieces:
            # Uploaded subtitles come as whole lines: spread their words across the line's time.
            parts = text.split()
            step = (seg_end - seg_start) / len(parts)
            for j, part in enumerate(parts):
                words.append({"start": round(seg_start + j * step, 3), "end": round(seg_start + (j + 1) * step, 3),
                              "word": part})
        if words:
            lines.append({"start": round(start, 3), "end": round(end, 3), "words": words,
                          "text": " ".join(w["word"] for w in words)})

    # Auto-caption lines overlap while they scroll; end each line where the next one starts.
    for i, line in enumerate(lines[:-1]):
        nxt = lines[i + 1]["start"]
        if line["start"] < nxt < line["end"]:
            line["end"] = nxt
            if line["words"]:
                line["words"][-1]["end"] = min(line["words"][-1]["end"], nxt)
    return lines


def _whisper_language(code: str) -> str | None:
    base = code.replace("-orig", "").split("-")[0].lower()
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES

        return base if base in _LANGUAGE_CODES else None
    except ImportError:
        return None


RATE_LIMIT_HELP = """YouTube is temporarily limiting requests from your internet connection ("Too Many Requests").
This happens after many downloads in a short time and usually clears within an hour.

- Wait a while and click Make shorts again.
- Or download the video yourself and use "Upload a video" (that skips YouTube entirely)."""


def _fetch_text(ydl, url: str, on_status: Callable[[str], None] | None, waits=(15, 45)) -> str:
    """GET a YouTube URL, waiting and retrying when YouTube answers 429 Too Many Requests."""
    from yt_dlp.networking.exceptions import HTTPError

    for attempt in range(len(waits) + 1):
        try:
            return ydl.urlopen(url).read().decode("utf-8", errors="replace")
        except HTTPError as e:
            if e.status != 429:
                raise RuntimeError(f"YouTube refused the subtitle download (HTTP {e.status}).") from e
            if attempt == len(waits):
                raise RuntimeError(RATE_LIMIT_HELP) from e
            if on_status:
                on_status(f"YouTube says too many requests; waiting {waits[attempt]}s and trying again...")
            time.sleep(waits[attempt])
    raise AssertionError("unreachable")


def fetch_details(url: str, cookies_file: Path | None = None,
                  on_status: Callable[[str], None] | None = None) -> dict:
    """Title, duration, engagement graph and subtitle transcript, without downloading the video.

    The returned "transcript" is None when the video has no usable subtitles; "subtitle_problem"
    then says why ("none" or "rate_limited").
    """
    def fetch(ydl) -> dict:
        info = ydl.extract_info(url.strip(), download=False, process=False)
        if info.get("_type") == "playlist" or info.get("entries"):
            raise ValueError("That link is a playlist. Paste the link of a single video.")
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            raise ValueError("That video is a live stream that hasn't finished. Try again once it has ended.")
        transcript = None
        problem = "none"
        track = _pick_caption_track(info)
        if track:
            key, formats = track
            fmt = next((f for f in formats if f.get("ext") == "json3"), None)
            data = None
            if fmt and fmt.get("url"):
                try:
                    data = json.loads(_fetch_text(ydl, fmt["url"], on_status))
                except RuntimeError:
                    problem = "rate_limited"  # the caller can still use the Most replayed graph
            if data is not None:
                lines = _parse_json3(data)
                if lines:
                    transcript = {
                        "language": key.replace("-orig", ""),
                        "whisper_language": _whisper_language(key),
                        "duration": float(info.get("duration") or lines[-1]["end"]),
                        "model": f"youtube-captions:{key}",
                        "word_timing": "exact" if _has_word_timing(data) else "approx",
                        "segments": lines,
                    }
        return {
            "id": info.get("id"),
            "title": info.get("title") or "YouTube video",
            "duration": float(info.get("duration") or 0),
            "heatmap": info.get("heatmap") or None,
            "transcript": transcript,
            "subtitle_problem": None if transcript else problem,
        }

    return run_ytdlp(fetch, cookies_file)


def replay_windows(heatmap: list[dict], duration: float, count: int, width: float) -> list[tuple[float, float]]:
    """Non-overlapping windows around the most re-watched moments, earliest first.

    Each window starts a bit before its peak so the clip can include the setup, not just the payoff.
    """
    if not heatmap or duration <= 0:
        return []
    width = min(width, duration)
    points = sorted(heatmap, key=lambda h: -(h.get("value") or 0))
    windows: list[tuple[float, float]] = []
    for h in points:
        center = ((h.get("start_time") or 0) + (h.get("end_time") or 0)) / 2
        if center < 20 and len(points) > count:
            continue  # the very start is "replayed" by everyone pressing play
        start = max(0.0, center - 0.6 * width)
        end = min(duration, start + width)
        start = max(0.0, end - width)
        if any(start < e and s < end for s, e in windows):
            continue
        windows.append((round(start, 2), round(end, 2)))
        if len(windows) == count:
            break
    return sorted(windows)


def stream_urls(url: str, cookies_file: Path | None = None) -> list[tuple[str, dict]]:
    """Direct media URLs (video, then audio if separate) for reading parts of the video."""
    def fetch(ydl) -> list[tuple[str, dict]]:
        info = ydl.extract_info(url.strip(), download=False)
        formats = info.get("requested_formats") or [info]
        return [(f["url"], f.get("http_headers") or {}) for f in formats]

    return run_ytdlp(fetch, cookies_file, format=VIDEO_FORMAT)


def download_section(
    streams: list[tuple[str, dict]],
    start: float,
    end: float,
    dest: Path,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """Fetch only [start, end] of the video into `dest` (.mkv), without re-encoding.

    ffmpeg requests just the needed byte ranges and copies the streams as they are, which takes
    seconds. The copy begins at the keyframe before `start` and keeps the original timestamps,
    so use section_times() to map between video time and positions in the file.
    """
    duration = end - start
    args: list[str] = []
    for media_url, headers in streams:
        header_blob = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        if header_blob:
            args += ["-headers", header_blob]
        if media_url.startswith("http"):
            args += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
        args += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", media_url]
    if len(streams) > 1:
        args += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        args += ["-map", "0:v:0", "-map", "0:a:0?"]
    args += ["-c", "copy", "-copyts", str(dest)]
    # With -copyts ffmpeg reports progress in video time, so measure it from `start`.
    report = (lambda f: on_progress(max(0.0, min(1.0, (f * end - start) / duration)))) if on_progress else None
    run_ffmpeg(args, duration=end, on_progress=report)


def section_times(path: Path) -> tuple[float, float]:
    """(file start, audio start) in video time for a section made by download_section.

    Seek positions inside the file are relative to the file start; audio decoded from the file
    (e.g. for transcription) begins at the audio start.
    """
    import av

    with av.open(str(path)) as c:
        start = c.start_time / av.time_base if c.start_time is not None else 0.0
        audio = next((s for s in c.streams if s.type == "audio"), None)
        audio_start = float(audio.start_time * audio.time_base) if audio and audio.start_time is not None else start
    return start, audio_start


def engagement_text(heatmap: list[dict] | None) -> str | None:
    """Compact 'time value' list from YouTube's "Most replayed" graph (100 = most replayed)."""
    if not heatmap:
        return None
    top = max((h.get("value") or 0) for h in heatmap) or 1
    return ", ".join(f"{fmt_ts(h.get('start_time') or 0)} {round(100 * (h.get('value') or 0) / top)}"
                     for h in heatmap)
