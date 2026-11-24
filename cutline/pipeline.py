"""End to end: source -> transcript -> Gemini picks clips -> rendered shorts + metadata files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import ROOT, youtube
from . import analyze as az
from .captions import build_ass, build_srt, relative_words
from .media import download, is_url, probe
from .reframe import plan_layout
from .render import make_thumbnail, render_clip
from .transcribe import transcribe

ProgressFn = Callable[[float, str], None]

# Share of the overall progress bar each stage occupies.
_STAGES = {"download": (0.00, 0.08), "transcribe": (0.08, 0.55), "analyze": (0.55, 0.62), "render": (0.62, 1.00)}
_FAST_STAGES = {"fetch": (0.00, 0.05), "analyze": (0.05, 0.20), "render": (0.20, 1.00)}
_REPLAY_STAGES = {"fetch": (0.00, 0.03), "windows": (0.03, 0.55), "analyze": (0.55, 0.65), "render": (0.65, 1.00)}
SECTION_PAD = 2.5  # seconds fetched around each clip in fast mode
_GENERATED = re.compile(r"^\d{2} - .+\.(mp4|jpg|srt|txt)$")


@dataclass
class Options:
    num_clips: int = 5
    min_sec: float = 20
    max_sec: float = 60
    whisper_model: str = "small"
    language: str | None = None  # spoken language code; None = auto-detect
    meta_language: str | None = None  # language for titles etc.; None = same as video
    instructions: str | None = None  # extra guidance for Gemini
    layout: str = "auto"  # auto | crop | fit
    captions: bool = True
    hook: bool = True
    words_per_caption: int = 3
    highlight: str = "#FFE600"
    uppercase: bool = True
    model: str = az.DEFAULT_MODEL
    out_dir: Path = field(default_factory=lambda: ROOT / "output")
    reanalyze: bool = False
    render: bool = True
    cookies_file: Path | None = None  # for YouTube's "not a bot" check; default: <project>/cookies.txt
    fast: bool = True  # YouTube links: use YouTube's subtitles and download only the clips


def safe_name(text: str, max_len: int = 60) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f#%]', "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len].rstrip(" .") or "clip"


def clip_text(clip: dict) -> str:
    """Copy-paste-ready upload text for one clip."""
    alt = "\n".join(f"- {t}" for t in clip.get("alt_titles") or [])
    return f"""TITLE
{clip['title']}

ALTERNATIVE TITLES
{alt}

DESCRIPTION
{clip['description']}

{' '.join(clip.get('hashtags') or [])}

TAGS
{', '.join(clip.get('tags') or [])}

TIKTOK CAPTION
{clip['tiktok_caption']}

INSTAGRAM CAPTION
{clip['instagram_caption']}

---
On-screen hook: {clip['hook']}
Source: {az.fmt_ts(clip['start'])} - {az.fmt_ts(clip['end'])} ({clip['duration']:.0f}s)
Virality score: {clip['virality_score']}/100
Why it works: {clip['why_it_works']}
"""


def run(source: str, opts: Options, progress: ProgressFn | None = None) -> dict:
    """Make shorts from a file or link.

    YouTube links use fast mode when possible: YouTube's subtitles stand in for transcribing the
    whole video, and only the chosen parts are downloaded. Anything else, or a YouTube video
    without subtitles, goes through the full download + transcription.
    """
    source = source.strip().strip('"')
    out_root = Path(opts.out_dir).resolve()
    if opts.fast and is_url(source) and youtube.is_youtube(source):
        result = _run_fast(source, opts, out_root, progress)
        if result is not None:
            return result
    return _run_full(source, opts, out_root, progress)


# --------------------------------------------------------------------------- shared steps

def _reporter(progress: ProgressFn | None, stages: dict) -> Callable[[str, float, str], None]:
    def report(stage: str, frac: float, msg: str) -> None:
        if progress:
            lo, hi = stages[stage]
            progress(lo + (hi - lo) * max(0.0, min(1.0, frac)), msg)
    return report


def _check_length(duration: float, opts: Options) -> None:
    if duration < opts.min_sec:
        raise ValueError(f"This video is only {az.fmt_ts(duration)} long, shorter than the minimum clip "
                         f"length ({opts.min_sec:g}s). Lower the minimum length and try again.")


def _analysis_params(opts: Options, transcript_model: str, engagement: bool) -> dict:
    return {
        "num_clips": opts.num_clips, "min_sec": opts.min_sec, "max_sec": opts.max_sec,
        "meta_language": opts.meta_language, "instructions": opts.instructions,
        "model": opts.model, "transcript": transcript_model, "language": opts.language,
        "engagement": engagement,
    }


def _cached_analysis(project: Path, params: dict, opts: Options) -> dict | None:
    analysis_file = project / "analysis.json"
    if analysis_file.exists() and not opts.reanalyze:
        saved = json.loads(analysis_file.read_text(encoding="utf-8"))
        if saved.get("params") == params:
            return saved["result"]
    return None


def _analyze(project: Path, transcript: dict, opts: Options, title: str, engagement: str | None,
             params: dict, report: Callable) -> dict:
    cached = _cached_analysis(project, params, opts)
    if cached is not None:
        report("analyze", 1, "Using saved clip analysis (tick re-run to get new picks)")
        return cached
    report("analyze", 0, f"Gemini is finding the best moments ({opts.model})...")
    expected = max(1, opts.num_clips) * 1600

    def on_text(chars: int) -> None:
        report("analyze", min(0.95, chars / expected), "Gemini is writing titles and descriptions...")

    analysis = az.analyze(
        transcript, num_clips=opts.num_clips, min_sec=opts.min_sec, max_sec=opts.max_sec,
        meta_language=opts.meta_language, instructions=opts.instructions, title=title,
        model=opts.model, on_text=on_text, engagement=engagement,
        on_status=lambda msg: report("analyze", 0.02, msg),
    )
    (project / "analysis.json").write_text(
        json.dumps({"params": params, "result": analysis}, ensure_ascii=False, indent=2), encoding="utf-8")
    return analysis


def _pick_clips(analysis: dict, transcript: dict, opts: Options, duration: float, report: Callable) -> list[dict]:
    clips = az.finalize_clips(analysis, transcript, num_clips=opts.num_clips, min_sec=opts.min_sec,
                              max_sec=opts.max_sec, duration=duration)
    if not clips:
        why = (analysis.get("video_summary") or "").strip()
        raise az.AnalysisError("Gemini didn't find any moments that fit your settings."
                               + (f"\n\nGemini said: {why}" if why else "")
                               + "\n\nTry a lower minimum length, a higher maximum, or different instructions.")
    used = (analysis.get("_meta") or {}).get("model")
    report("analyze", 1, f"Found {len(clips)} clips" + (f" (model: {used})" if used else ""))
    return clips


def _number_clips(project: Path, clips: list[dict], opts: Options) -> None:
    if opts.render:
        for old in project.iterdir():
            if old.is_file() and _GENERATED.match(old.name):
                old.unlink()
    for i, clip in enumerate(clips, 1):
        base = f"{i:02d} - {safe_name(clip['title'])}"
        clip["number"] = i
        clip["files"] = {
            "video": str(project / f"{base}.mp4"), "thumbnail": str(project / f"{base}.jpg"),
            "captions_srt": str(project / f"{base}.srt"), "text": str(project / f"{base}.txt"),
        }


def _write_clip_texts(clip: dict) -> list[dict]:
    """Write the clip's .txt and .srt; returns its words relative to the clip start."""
    words = relative_words(clip.pop("_words"), clip["start"], clip["duration"])
    Path(clip["files"]["text"]).write_text(clip_text(clip), encoding="utf-8")
    Path(clip["files"]["captions_srt"]).write_text(build_srt(words, clip["duration"]), encoding="utf-8")
    return words


def _render(info, clip: dict, words: list[dict], offset: float, work: Path, opts: Options,
            report_clip: Callable[[float, str], None]) -> None:
    """Render one clip from `info` (a file in which the clip starts at clip['start'] - offset)."""
    start, end = clip["start"] - offset, clip["end"] - offset
    report_clip(0.0, "reframing...")
    layout = plan_layout(info, start, clip["duration"], opts.layout)
    clip["layout"] = layout.kind
    ass_path = None
    if opts.captions or opts.hook:
        ass_path = work / f"{clip['number']:02d}.ass"
        ass_path.write_text(
            build_ass(words, clip["duration"], hook=clip["hook"] if opts.hook else None,
                      max_words=opts.words_per_caption, highlight=opts.highlight,
                      uppercase=opts.uppercase, captions=opts.captions),
            encoding="utf-8",
        )
    video = Path(clip["files"]["video"])
    render_clip(info, start, end, video, layout, ass_path=ass_path,
                on_progress=lambda f: report_clip(f, f"rendering... {f:.0%}"))
    make_thumbnail(video, Path(clip["files"]["thumbnail"]))
    if ass_path:
        ass_path.unlink(missing_ok=True)


def _write_results(project: Path, source: str, title: str, analysis: dict, clips: list[dict],
                   mode: str) -> dict:
    result = {
        "source": source,
        "title": title,
        "mode": mode,
        "video_summary": analysis.get("video_summary", ""),
        "project_dir": str(project),
        "clips": clips,
    }
    (project / "clips.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# {title}\n", f"{result['video_summary']}\n"]
    for clip in clips:
        md.append(f"\n## {clip['number']:02d}. {clip['title']}\n\n```\n{clip_text(clip)}```\n")
    (project / "clips.md").write_text("\n".join(md), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- full mode

def _run_full(source: str, opts: Options, out_root: Path, progress: ProgressFn | None) -> dict:
    report = _reporter(progress, _STAGES)
    engagement = None
    if is_url(source):
        report("download", 0, "Downloading the whole video...")
        video_path, yt_info = download(
            source, out_root / "_downloads", cookies_file=opts.cookies_file,
            on_progress=lambda f: report("download", f, f"Downloading the whole video... {f:.0%}"))
        title = yt_info.get("title") or video_path.stem
        engagement = youtube.engagement_text(yt_info.get("heatmap"))
    else:
        video_path = Path(source).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        title = video_path.stem.replace("_", " ")
    info = probe(video_path)
    if not info.has_audio:
        raise ValueError(f"{video_path.name} has no audio track, so there is no speech to find clips in.")
    _check_length(info.duration, opts)
    project = out_root / safe_name(video_path.stem, 80)
    work = project / "_work"
    work.mkdir(parents=True, exist_ok=True)

    params = _analysis_params(opts, f"whisper:{opts.whisper_model}", bool(engagement))
    if _cached_analysis(project, params, opts) is None and not az.has_credentials():
        raise az.AnalysisError(az.NO_KEY_MESSAGE)  # fail before the slow transcription step

    # Transcribe (cached per video + whisper settings).
    transcript_file = project / "transcript.json"
    transcript = None
    if transcript_file.exists():
        saved = json.loads(transcript_file.read_text(encoding="utf-8"))
        if saved.get("model") == opts.whisper_model and saved.get("requested_language") == opts.language:
            transcript = saved
    if transcript is None:
        report("transcribe", 0, f"Preparing audio and loading the speech model ({opts.whisper_model})... "
                                f"long videos take a minute or two here")

        def on_tx(frac: float, done: float, total: float) -> None:
            report("transcribe", frac, f"Transcribing... {az.fmt_ts(done)} / {az.fmt_ts(total)}")

        transcript = transcribe(info.path, model_size=opts.whisper_model, language=opts.language, on_progress=on_tx)
        transcript["requested_language"] = opts.language
        transcript_file.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    report("transcribe", 1, f"Transcript ready ({len(transcript['segments'])} lines, language: {transcript['language']})")

    analysis = _analyze(project, transcript, opts, title, engagement, params, report)
    clips = _pick_clips(analysis, transcript, opts, info.duration, report)
    _number_clips(project, clips, opts)
    n = len(clips)
    for i, clip in enumerate(clips, 1):
        words = _write_clip_texts(clip)
        if opts.render:
            _render(info, clip, words, 0.0, work, opts,
                    lambda f, msg, i=i: report("render", (i - 1 + f) / n, f"Clip {i}/{n}: {msg}"))
    result = _write_results(project, str(info.path), title, analysis, clips, "full")
    report("render", 1, f"Done! {n} clips saved to {project}")
    return result


# --------------------------------------------------------------------------- fast mode (YouTube)

def _run_fast(url: str, opts: Options, out_root: Path, progress: ProgressFn | None) -> dict | None:
    """Returns None (caller falls back to full mode) when the video has no usable subtitles."""
    report = _reporter(progress, _FAST_STAGES)
    report("fetch", 0, "Fast mode: reading the video's details and subtitles from YouTube...")
    details = youtube.fetch_details(url, opts.cookies_file, on_status=lambda msg: report("fetch", 0.5, msg))
    transcript = details["transcript"]
    if transcript is None:
        if details["heatmap"]:
            return _run_replay(url, opts, out_root, progress, details)
        if details.get("subtitle_problem") == "rate_limited":
            # The subtitles exist; YouTube is only rate-limiting them for a while. Waiting is far
            # faster than transcribing the whole video, so don't fall back to the full process.
            raise RuntimeError(youtube.RATE_LIMIT_HELP + "\n- Or untick Fast mode to download and transcribe the "
                                                        "whole video (slow for long videos).")
        if progress:
            progress(0.0, "This video has no YouTube subtitles and no Most replayed graph, so Cutline will "
                          "download and transcribe all of it...")
        return None
    duration = details["duration"] or transcript["duration"]
    _check_length(duration, opts)
    title = details["title"]
    project = out_root / safe_name(f"{title} [{details['id']}]", 80)
    work = project / "_work"
    work.mkdir(parents=True, exist_ok=True)
    (project / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    engagement = youtube.engagement_text(details["heatmap"])
    report("fetch", 1, f"Got YouTube's subtitles ({len(transcript['segments'])} lines, {transcript['language']})"
                       + (" and its Most replayed graph" if engagement else ""))

    params = _analysis_params(opts, transcript["model"], bool(engagement))
    if _cached_analysis(project, params, opts) is None and not az.has_credentials():
        raise az.AnalysisError(az.NO_KEY_MESSAGE)
    analysis = _analyze(project, transcript, opts, title, engagement, params, report)
    clips = _pick_clips(analysis, transcript, opts, duration, report)
    _number_clips(project, clips, opts)

    streams = None
    if opts.render:
        report("render", 0, "Connecting to the video on YouTube...")
        streams = youtube.stream_urls(url, opts.cookies_file)
    whisper_lang = opts.language or transcript.get("whisper_language")
    exact_words = transcript.get("word_timing") == "exact"
    n = len(clips)
    for i, clip in enumerate(clips, 1):
        if not opts.render:
            _write_clip_texts(clip)
            continue

        def report_clip(frac: float, msg: str, i: int = i) -> None:
            report("render", (i - 1 + frac) / n, f"Clip {i}/{n}: {msg}")

        # YouTube's subtitle timing is approximate, so fetch a padded part of the video, then
        # transcribe just that part for exact word timing (clean cuts + accurate captions).
        sec_start = max(0.0, clip["start"] - SECTION_PAD)
        sec_end = min(duration, clip["end"] + SECTION_PAD)
        section = work / f"part_{i:02d}.mkv"
        report_clip(0.0, f"downloading just this part ({az.fmt_ts(sec_start)}-{az.fmt_ts(sec_end)})...")
        try:
            youtube.download_section(streams, sec_start, sec_end, section,
                                     on_progress=lambda f: report_clip(0.35 * f, f"downloading this part... {f:.0%}"))
        except RuntimeError as e:
            raise RuntimeError(f"Couldn't fetch part of the video from YouTube. Try again, or untick Fast mode "
                               f"to download the whole video instead.\n\n{e}") from e
        file_start, audio_start = youtube.section_times(section)
        if exact_words:
            # YouTube's automatic captions already time every word: no transcription needed.
            # Add a little lead-in/out, since their timing can be a fraction of a second off.
            start = max(sec_start, clip["start"] - 0.2)
            end = min(sec_end, clip["end"] + 0.2)
            clip.update(start=round(start, 3), end=round(end, 3), duration=round(end - start, 3))
        else:
            # Uploaded subtitles only time whole lines: transcribe just this part for exact timing.
            report_clip(0.35, "transcribing this part for exact captions...")
            part_tx = transcribe(section, model_size=opts.whisper_model, language=whisper_lang)
            _refine_with_part(clip, part_tx, audio_start, sec_end, opts)
        words = _write_clip_texts(clip)
        info = probe(section)
        _render(info, clip, words, file_start, work, opts,
                lambda f, msg: report_clip(0.4 + 0.6 * f, msg))
        section.unlink(missing_ok=True)

    result = _write_results(project, url, title, analysis, clips, "fast")
    report("render", 1, f"Done! {n} clips saved to {project}")
    return result


# --------------------------------------------------------------------------- Most replayed backup

def _run_replay(url: str, opts: Options, out_root: Path, progress: ProgressFn | None, details: dict) -> dict:
    """No subtitles available: transcribe only the windows around the most re-watched moments
    (from YouTube's "Most replayed" graph) and pick the clips from those."""
    report = _reporter(progress, _REPLAY_STAGES)
    duration = details["duration"]
    _check_length(duration, opts)
    why = ("YouTube is limiting subtitle downloads right now" if details.get("subtitle_problem") == "rate_limited"
           else "This video has no YouTube subtitles")
    report("fetch", 1, f"{why}, so Cutline is using the Most replayed graph instead...")
    title = details["title"]
    project = out_root / safe_name(f"{title} [{details['id']}]", 80)
    work = project / "_work"
    work.mkdir(parents=True, exist_ok=True)
    windows = youtube.replay_windows(details["heatmap"], duration, opts.num_clips + 1, max(opts.max_sec + 30, 75))
    engagement = youtube.engagement_text(details["heatmap"])
    parts = [work / f"replay_{i:02d}.mkv" for i in range(1, len(windows) + 1)]

    # Transcript of just those windows, cached so a retry (say after a Gemini hiccup) skips the slow part.
    cache_key = {"model": f"replay:{opts.whisper_model}", "windows": [list(w) for w in windows],
                 "language": opts.language}
    transcript_file = project / "transcript.json"
    transcript = None
    if transcript_file.exists():
        saved = json.loads(transcript_file.read_text(encoding="utf-8"))
        if saved.get("cache_key") == cache_key:
            transcript = saved
    params = _analysis_params(opts, f"{cache_key['model']}:{cache_key['windows']}", True)
    if _cached_analysis(project, params, opts) is None and not az.has_credentials():
        raise az.AnalysisError(az.NO_KEY_MESSAGE)

    n = len(windows)
    missing = [i for i, p in enumerate(parts) if not p.exists()]
    if missing and (transcript is None or opts.render):
        report("windows", 0, "Connecting to the video on YouTube...")
        streams = youtube.stream_urls(url, opts.cookies_file)
        for i in missing:
            start, end = windows[i]

            def on_dl(f: float, i: int = i) -> None:
                report("windows", (i + 0.3 * f) / n, f"Most replayed part {i + 1}/{n}: downloading... {f:.0%}")

            report("windows", i / n, f"Most replayed part {i + 1}/{n}: downloading {az.fmt_ts(start)}-{az.fmt_ts(end)}...")
            youtube.download_section(streams, start, end, parts[i], on_progress=on_dl)

    if transcript is None:
        segments: list[dict] = []
        language = opts.language
        for i, ((start, end), part) in enumerate(zip(windows, parts)):
            def on_tx(f: float, done: float, total: float, i: int = i) -> None:
                report("windows", (i + 0.3 + 0.7 * f) / n,
                       f"Most replayed part {i + 1}/{n}: transcribing... {az.fmt_ts(done)} / {az.fmt_ts(total)}")

            report("windows", (i + 0.3) / n, f"Most replayed part {i + 1}/{n}: transcribing...")
            tx = transcribe(part, model_size=opts.whisper_model, language=language, on_progress=on_tx)
            language = language or tx["language"]  # reuse the first part's language for the rest
            shift = youtube.section_times(part)[1]  # transcript times are relative to the part's audio
            for s in tx["segments"]:
                if s["end"] + shift < start or s["start"] + shift > end:
                    continue  # lead-in before the window (the copy starts at a keyframe)
                segments.append({
                    "start": round(s["start"] + shift, 3), "end": round(s["end"] + shift, 3), "text": s["text"],
                    "words": [{"start": round(w["start"] + shift, 3), "end": round(w["end"] + shift, 3),
                               "word": w["word"]} for w in s.get("words", [])],
                })
        transcript = {"language": language, "duration": duration, "model": cache_key["model"],
                      "windows": cache_key["windows"], "segments": segments, "cache_key": cache_key}
        transcript_file.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    if not transcript["segments"]:
        raise az.AnalysisError("There's no speech in the most re-watched parts of this video, so there's nothing "
                               "for Cutline to build clips around. Untick Fast mode to process the whole video.")
    report("windows", 1, f"Transcribed the {n} most re-watched parts")

    analysis = _analyze(project, transcript, opts, title, engagement, params, report)
    clips = _fit_to_windows(_pick_clips(analysis, transcript, opts, duration, report), windows, opts)
    if not clips:
        raise az.AnalysisError("Gemini's picks didn't fit inside the most re-watched parts. Try again with "
                               "re-run ticked, or untick Fast mode to process the whole video.")
    _number_clips(project, clips, opts)
    for i, clip in enumerate(clips, 1):
        window = clip.pop("_window")
        words = _write_clip_texts(clip)
        if opts.render:
            _render(probe(parts[window]), clip, words, youtube.section_times(parts[window])[0], work, opts,
                    lambda f, msg, i=i: report("render", (i - 1 + f) / len(clips), f"Clip {i}/{len(clips)}: {msg}"))
    result = _write_results(project, url, title, analysis, clips, "most-replayed")
    if opts.render:
        for part in parts:
            part.unlink(missing_ok=True)  # big files; the cached transcript is what saves time on a re-run
    report("render", 1, f"Done! {len(clips)} clips saved to {project}")
    return result


def _fit_to_windows(clips: list[dict], windows: list[tuple[float, float]], opts: Options) -> list[dict]:
    """Keep each clip inside the downloaded part it (mostly) falls in."""
    kept = []
    for clip in clips:
        k = max(range(len(windows)),
                key=lambda j: min(clip["end"], windows[j][1]) - max(clip["start"], windows[j][0]))
        lo, hi = windows[k]
        start, end = max(clip["start"], lo), min(clip["end"], hi)
        words = [w for w in clip["_words"] if w["start"] >= start - 0.01 and w["end"] <= end + 0.01]
        if words and (start, end) != (clip["start"], clip["end"]):
            start, end = max(start, words[0]["start"] - 0.15), min(end, words[-1]["end"] + 0.3)
        if end - start < max(5.0, opts.min_sec * 0.5):
            continue
        clip.update(start=round(start, 3), end=round(end, 3), duration=round(end - start, 3), _words=words,
                    _window=k)
        kept.append(clip)
    return kept


def _refine_with_part(clip: dict, part_tx: dict, sec_start: float, sec_end: float, opts: Options) -> None:
    """Re-snap a clip's boundaries and words using whisper's exact timing for its part."""
    segments = [{
        "start": s["start"] + sec_start, "end": s["end"] + sec_start, "text": s["text"],
        "words": [{"start": w["start"] + sec_start, "end": w["end"] + sec_start, "word": w["word"]}
                  for w in s.get("words", [])],
    } for s in part_tx["segments"]]
    if not any(s["words"] for s in segments):
        return  # nothing recognised: keep the subtitle-based timing
    raw = {k: v for k, v in clip.items() if not k.startswith("_")}
    refined = az.finalize_clips({"clips": [raw]}, {"segments": segments}, num_clips=1,
                                min_sec=opts.min_sec, max_sec=opts.max_sec, duration=sec_end)
    if refined:
        best = refined[0]
        start = max(best["start"], sec_start)
        clip.update(start=round(start, 3), end=best["end"], duration=round(best["end"] - start, 3),
                    _words=best["_words"])
