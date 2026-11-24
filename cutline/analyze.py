"""Ask Gemini to find the most viral moments and write the packaging for each clip."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Callable

from google import genai
from google.genai import errors, types

# Google's recommended free-tier model for new projects.
DEFAULT_MODEL = "gemini-4-hydrogen"
# Tried in order if the chosen model is unavailable to this key (paid-only, retired, overloaded...).
# All of these are on the Gemini API free tier.
FALLBACK_MODELS = ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]

NO_KEY_MESSAGE = (
    "No Gemini API key found. Get a free one at https://aistudio.google.com/apikey and either\n"
    "  - paste it into the API key box in the web app, or\n"
    "  - create a file named .env in the cutline folder containing: GEMINI_API_KEY=your-key\n"
    "  - or set the GEMINI_API_KEY environment variable."
)

_NETWORK_ERRORS: tuple[type[BaseException], ...] = (OSError,)
for _mod in ("httpx", "httpx2"):
    try:
        _NETWORK_ERRORS += (__import__(_mod).TransportError,)
    except (ImportError, AttributeError):
        pass


class AnalysisError(RuntimeError):
    pass


CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "video_summary": {"type": "string"},
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "clip start in seconds"},
                    "end": {"type": "number", "description": "clip end in seconds"},
                    "virality_score": {"type": "integer", "minimum": 1, "maximum": 100},
                    "hook": {"type": "string"},
                    "title": {"type": "string"},
                    "alt_titles": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "tiktok_caption": {"type": "string"},
                    "instagram_caption": {"type": "string"},
                    "why_it_works": {"type": "string"},
                },
                "required": [
                    "start", "end", "virality_score", "hook", "title", "alt_titles", "description",
                    "hashtags", "tags", "tiktok_caption", "instagram_caption", "why_it_works",
                ],
            },
        },
    },
    "required": ["video_summary", "clips"],
}

SYSTEM_PROMPT = (
    "You are a senior short-form video editor and growth strategist. You have cut hundreds of "
    "long-form videos, podcasts and streams into YouTube Shorts, TikToks and Instagram Reels that "
    "went viral. You can tell which moments stop a scroller and hold them to the end, and you write "
    "packaging (titles, captions, hashtags) that earns the click without misleading anyone."
)


def fmt_ts(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_transcript(transcript: dict) -> str:
    """One `[start-end] text` line per segment; for a partial transcript (only some windows of
    the video transcribed) each window gets a header line."""
    windows = transcript.get("windows")
    lines = []
    for s in transcript["segments"]:
        if not s["text"]:
            continue
        if windows:
            while windows and s["start"] >= windows[0][0] - 0.01:
                n = len(transcript["windows"]) - len(windows) + 1
                lines.append(f"--- part {n}: {fmt_ts(windows[0][0])} to {fmt_ts(windows[0][1])} ---")
                windows = windows[1:]
        lines.append(f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}")
    return "\n".join(lines)


def build_prompt(
    transcript: dict,
    *,
    num_clips: int,
    min_sec: float,
    max_sec: float,
    meta_language: str | None,
    instructions: str | None,
    title: str | None,
    engagement: str | None = None,
) -> str:
    language = meta_language or "the same language as the transcript"
    about = f' titled "{title}"' if title else ""
    extra = f"\nAdditional instructions from the creator:\n{instructions.strip()}\n" if instructions else ""
    if engagement:
        extra += (
            "\nViewer engagement: YouTube's \"Most replayed\" graph for this video, as `time value` pairs "
            "(value 0-100, 100 = the most re-watched moment):\n"
            f"{engagement}\n"
            "Strongly prefer moments where viewers re-watch the most, as long as they also meet the rules "
            "above. A replay peak can come from something visual (a gameplay highlight, a reaction) that "
            "the transcript only hints at, so look closely at what is said around each peak.\n"
        )
    if transcript.get("windows"):
        extra += (
            "\nImportant: to save time, only the most re-watched parts of this video were transcribed. They "
            "appear in the transcript as separate parts with gaps between them. Each clip must lie entirely "
            "inside one part. Choose the best moment from each part, and skip a part if nothing in it works "
            "as a standalone clip.\n"
        )
    return f"""<transcript>
{format_transcript(transcript)}
</transcript>

Above is the timestamped transcript of a {fmt_ts(transcript['duration'])} long video{about}. Each line is `[start-end] text`, times in seconds.

Find the {num_clips} best moments to publish as standalone vertical short videos. Return fewer if the video genuinely has fewer strong moments; never pad the list with weak clips.

What makes a moment work as a short:
- Hook: the first 1-3 seconds grab attention on their own - a bold claim, a surprising fact, a question, a conflict, strong emotion, or the setup of a punchline. Never start mid-sentence or on filler ("so", "um", "and yeah", "like I said").
- Self-contained: a viewer who has not seen the full video understands it without extra context.
- Payoff: it ends on a complete thought - the answer, punchline, reveal or takeaway - never mid-sentence.
- Value or emotion: insights, actionable tips, contrarian takes, funny moments, stories with a twist, quotable lines.

Timing rules:
- Each clip is one continuous range, between {min_sec:g} and {max_sec:g} seconds long.
- `start` is the start time of the transcript line where the clip begins; `end` is the end time of the line where it ends.
- Clips must not overlap.

Packaging for each clip, written in {language}:
- hook: on-screen text for the first seconds, at most 7 words, punchy, no hashtags or emojis.
- title: YouTube Shorts title, at most 70 characters, curiosity-driven but truthful to what the clip delivers. Never invent facts, numbers, "rules" or claims that the speaker does not actually say in the clip.
- alt_titles: 2 alternative titles that take different angles.
- description: 2-3 sentences of context for YouTube, ending with a call to action.
- hashtags: 3-6 hashtags for YouTube, each starting with #, including #shorts.
- tags: 8-15 YouTube search tags (plain keywords, no #).
- tiktok_caption: at most 150 characters, with 3-5 hashtags inline.
- instagram_caption: 1-3 short lines for Reels, at most 2 emojis, ending with 5-8 hashtags.
- virality_score: integer from 1 to 100, your honest estimate of this clip's viral potential.
- why_it_works: one sentence on why this moment will perform.

Also return video_summary: 1-2 sentences describing the whole video.
{extra}"""


def _api_key() -> str | None:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None


def has_credentials() -> bool:
    return _api_key() is not None


def analyze(
    transcript: dict,
    *,
    num_clips: int,
    min_sec: float,
    max_sec: float,
    meta_language: str | None = None,
    instructions: str | None = None,
    title: str | None = None,
    model: str = DEFAULT_MODEL,
    on_text: Callable[[int], None] | None = None,
    engagement: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict:
    """on_text(chars_received) is called as Gemini's answer streams in.

    If `model` is unavailable to this API key, the FALLBACK_MODELS are tried in turn.
    """
    if not transcript["segments"]:
        raise AnalysisError("The transcript is empty - the video seems to have no speech to work with.")
    if not has_credentials():
        raise AnalysisError(NO_KEY_MESSAGE)

    prompt = build_prompt(
        transcript, num_clips=num_clips, min_sec=min_sec, max_sec=max_sec,
        meta_language=meta_language, instructions=instructions, title=title, engagement=engagement,
    )
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_json_schema=CLIP_SCHEMA,
        max_output_tokens=65536,
        # "high" (the default on gemini-4-hydrogen) took ~4 minutes per video; medium is much
        # faster and still picks well. All FALLBACK_MODELS accept this level.
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MEDIUM),
    )
    client = genai.Client(
        api_key=_api_key(),
        http_options=types.HttpOptions(
            timeout=6 * 60 * 1000,  # ms; a normal answer takes 1-3 minutes, a stall shouldn't hang longer
            retry_options=types.HttpRetryOptions(attempts=3),
        ),
    )

    models = list(dict.fromkeys([model, *FALLBACK_MODELS]))  # keeps order, drops duplicates
    waits = [30, 60, 90]  # Google's "high demand" 503s usually clear within a few minutes
    for attempt in range(len(waits) + 1):
        failures: list[tuple[str, int, str]] = []
        for name in models:
            try:
                return _generate(client, name, prompt, config, on_text)
            except errors.APIError as e:
                message = (e.message or str(e)).strip()
                if e.code in (400, 401, 403) and re.search(
                        r"API[_ ]KEY|api key|unauthenticated|PERMISSION_DENIED.*key", message, re.IGNORECASE):
                    raise AnalysisError("Your Gemini API key was rejected. Check that it is correct.\n\n"
                                        + NO_KEY_MESSAGE) from e
                # Model not available to this key (paid-only, retired), quota exhausted, or overloaded:
                # try the next model.
                failures.append((name, e.code, f"{e.status or ''} {message[:300]}"))
            except _NETWORK_ERRORS as e:
                raise AnalysisError(f"Could not reach the Gemini API. Check your internet connection. ({e})") from e

        codes = {code for _, code, _ in failures}
        overloaded = codes <= {500, 502, 503, 504}
        # Retry while some model is merely overloaded (others may be out of free quota).
        retryable = codes <= {429, 500, 502, 503, 504} and bool(codes & {500, 502, 503, 504})
        if retryable and attempt < len(waits):
            for left in range(waits[attempt], 0, -5):
                if on_status:
                    on_status(f"Google's Gemini servers are overloaded right now; trying again in {left}s "
                              f"(attempt {attempt + 2} of {len(waits) + 1})...")
                time.sleep(5)
            continue
        break

    details = "\n".join(f"  {name}: {code} {text}" for name, code, text in failures)
    if overloaded:
        raise AnalysisError(
            "Google's Gemini servers are overloaded right now (\"high demand\"), and stayed that way for a few "
            "minutes. This is temporary and on Google's side. Wait a little and click Make shorts again; "
            "the slow steps are saved, so it continues from here.\n\n" + details)
    if codes == {429}:
        raise AnalysisError(
            "You've hit the Gemini free-tier limit for now. Wait a while (limits reset daily) and try again; "
            "the slow steps are saved, so it continues from here.\n\n" + details)
    if retryable:
        raise AnalysisError(
            "Gemini is busy right now: some models are overloaded (\"high demand\") and your free daily quota "
            "for the others is used up. Both are temporary. Try again later; the slow steps are saved, so it "
            "continues from here. (A Gemini key with billing turned on avoids the free-tier limits.)\n\n"
            + details)
    raise AnalysisError("None of the Gemini models could take this request:\n" + details)


def _generate(client: genai.Client, model: str, prompt: str, config: types.GenerateContentConfig,
              on_text: Callable[[int], None] | None) -> dict:
    parts: list[str] = []
    received = 0
    last = None
    for chunk in client.models.generate_content_stream(model=model, contents=prompt, config=config):
        last = chunk
        if not chunk.candidates or not chunk.candidates[0].content:
            continue
        for part in chunk.candidates[0].content.parts or []:
            if part.text and not part.thought:
                parts.append(part.text)
                received += len(part.text)
                if on_text:
                    on_text(received)

    if last is None:
        raise AnalysisError(f"Gemini ({model}) returned an empty response.")
    feedback = last.prompt_feedback
    if not parts and feedback and feedback.block_reason:
        raise AnalysisError(f"Gemini refused to analyze this video ({feedback.block_reason}).")
    finish = last.candidates[0].finish_reason if last.candidates else None
    finish_name = getattr(finish, "name", str(finish or ""))
    if finish_name == "MAX_TOKENS":
        raise AnalysisError("Gemini's answer was cut off. Try asking for fewer clips.")
    if finish_name in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"):
        raise AnalysisError(f"Gemini stopped because of its content filters ({finish_name}).")

    text = "".join(parts).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)  # tolerate a fenced reply
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AnalysisError(f"Gemini ({model}) returned malformed JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
        raise AnalysisError(f"Gemini ({model}) returned an unexpected answer format.")

    usage = last.usage_metadata
    data["_meta"] = {
        "model": last.model_version or model,
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "thinking_tokens": getattr(usage, "thoughts_token_count", None),
    }
    return data


def _hashtags(raw) -> list[str]:
    """Normalise to '#word' (models sometimes drop the '#' or include spaces); keep order, drop duplicates."""
    out: list[str] = []
    for tag in raw or []:
        word = re.sub(r"[^\w]", "", str(tag), flags=re.UNICODE)
        if word and f"#{word}".lower() not in (t.lower() for t in out):
            out.append(f"#{word}")
    return out


def finalize_clips(
    data: dict,
    transcript: dict,
    *,
    num_clips: int,
    min_sec: float,
    max_sec: float,
    duration: float,
) -> list[dict]:
    """Snap the model's ranges to word boundaries, enforce lengths, drop overlaps, rank by score."""
    words = [w for s in transcript["segments"] for w in s.get("words", [])]
    words.sort(key=lambda w: w["start"])
    results = []

    for raw in data.get("clips", []):
        try:
            start, end = float(raw["start"]), float(raw["end"])
        except (KeyError, TypeError, ValueError):
            continue
        start, end = max(0.0, start), min(duration, end)
        if end - start < 3:
            continue

        idx = [i for i, w in enumerate(words) if w["end"] > start + 0.05 and w["start"] < end - 0.05]
        if idx:
            first, last = idx[0], idx[-1]
            start, end = words[first]["start"], words[last]["end"]

            # Too long: cut at the last sentence end that fits, else the last word that fits.
            if end - start > max_sec * 1.15:
                fit = [i for i in idx if words[i]["end"] - start <= max_sec]
                sentence = [i for i in fit if words[i]["word"].endswith((".", "!", "?"))
                            and words[i]["end"] - start >= min(min_sec, max_sec / 2)]
                pick = sentence or fit
                if pick:
                    last = pick[-1]
                    end = words[last]["end"]

            # Small pads that never swallow part of a neighbouring word.
            prev_end = words[first - 1]["end"] if first > 0 else 0.0
            next_start = words[last + 1]["start"] if last + 1 < len(words) else duration
            start = max(0.0, start - min(0.15, max(0.0, start - prev_end - 0.02)))
            end = min(duration, end + max(0.05, min(0.45, next_start - end - 0.02)))
        if end - start < 5:
            continue

        clip_words = [
            {"start": w["start"], "end": w["end"], "word": w["word"]}
            for w in words if w["start"] >= start - 0.01 and w["end"] <= end + 0.01
        ]
        clip = {k: raw.get(k) for k in CLIP_SCHEMA["properties"]["clips"]["items"]["properties"]}
        clip.update(start=round(start, 3), end=round(end, 3), duration=round(end - start, 3))
        clip["virality_score"] = int(clip.get("virality_score") or 0)
        clip["hashtags"] = _hashtags(clip.get("hashtags"))
        clip["tags"] = [t.strip().lstrip("#") for t in clip.get("tags") or [] if t and t.strip()]
        for key in ("alt_titles",):
            clip[key] = [t.strip() for t in clip.get(key) or [] if t and t.strip()]
        for key in ("hook", "title", "description", "tiktok_caption", "instagram_caption", "why_it_works"):
            clip[key] = (clip.get(key) or "").strip()
        clip["_words"] = clip_words
        results.append(clip)

    results.sort(key=lambda c: -c["virality_score"])
    kept: list[dict] = []
    for clip in results:
        if all(min(clip["end"], k["end"]) - max(clip["start"], k["start"]) < 1.0 for k in kept):
            kept.append(clip)
    return kept[:num_clips]
