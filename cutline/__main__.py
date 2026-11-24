"""Command line: python -m cutline <video file or URL> [options]"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import ROOT
from .analyze import DEFAULT_MODEL, AnalysisError, fmt_ts
from .pipeline import Options, run


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser(
        prog="cutline",
        description="Turn a long video into short vertical clips with captions, titles, descriptions and hashtags.",
    )
    p.add_argument("source", help="path to a video file, or a URL (YouTube, Vimeo, X, ...)")
    p.add_argument("-n", "--clips", type=int, default=5, help="how many clips to make (default 5)")
    p.add_argument("--min", dest="min_sec", type=float, default=20, help="minimum clip length in seconds (default 20)")
    p.add_argument("--max", dest="max_sec", type=float, default=60, help="maximum clip length in seconds (default 60)")
    p.add_argument("-o", "--out", default=str(ROOT / "output"), help="output folder (default ./output)")
    p.add_argument("--whisper", default="small",
                   help="speech model: tiny, base, small, medium, large-v3, turbo (default small)")
    p.add_argument("--lang", default=None, help="spoken language code, e.g. en, hi, ur, es (default auto-detect)")
    p.add_argument("--meta-lang", default=None,
                   help='language for titles/descriptions, e.g. "English" (default: same as the video)')
    p.add_argument("--instructions", default=None,
                   help='extra guidance for picking clips, e.g. "focus on the money advice"')
    p.add_argument("--layout", choices=["auto", "crop", "fit"], default="auto",
                   help="auto (default): follow faces, or fit over blur when there are none; "
                        "crop: always follow faces; fit: whole frame over a blurred background")
    p.add_argument("--no-captions", action="store_true", help="don't burn in captions")
    p.add_argument("--no-hook", action="store_true", help="don't show the hook text at the start")
    p.add_argument("--words", type=int, default=3, help="max words per caption (default 3)")
    p.add_argument("--highlight", default="#FFE600", help="caption highlight colour (default #FFE600)")
    p.add_argument("--no-upper", action="store_true", help="keep caption case instead of UPPERCASE")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Gemini model (default {DEFAULT_MODEL}; falls back to other free models if unavailable)")
    p.add_argument("--reanalyze", action="store_true", help="ask Gemini again instead of reusing saved results")
    p.add_argument("--no-render", action="store_true", help="only pick clips and write text, skip video rendering")
    p.add_argument("--full", action="store_true",
                   help="YouTube links: download and transcribe the whole video instead of fast mode "
                        "(fast mode uses YouTube's subtitles and downloads only the chosen parts)")
    p.add_argument("--cookies", default=None,
                   help="cookies.txt exported from youtube.com, for when YouTube says \"confirm you're not a bot\" "
                        "(default: cookies.txt in the Cutline folder, if present)")
    a = p.parse_args(argv)

    if a.min_sec >= a.max_sec:
        p.error("--min must be smaller than --max")

    opts = Options(
        num_clips=max(1, a.clips), min_sec=a.min_sec, max_sec=a.max_sec, whisper_model=a.whisper,
        language=a.lang, meta_language=a.meta_lang, instructions=a.instructions, layout=a.layout,
        captions=not a.no_captions, hook=not a.no_hook, words_per_caption=max(1, a.words),
        highlight=a.highlight, uppercase=not a.no_upper, model=a.model, out_dir=Path(a.out),
        reanalyze=a.reanalyze, render=not a.no_render,
        cookies_file=Path(a.cookies).expanduser().resolve() if a.cookies else None,
        fast=not a.full,
    )

    tty = sys.stdout.isatty()
    last = {"msg": "", "t": 0.0}

    def progress(frac: float, msg: str) -> None:
        now = time.monotonic()
        stage = msg.split("...")[0]
        if tty:
            if last["msg"] and stage != last["msg"].split("...")[0]:
                sys.stdout.write("\n")  # keep the finished stage's line, start a new one
            sys.stdout.write(f"\r[{frac:4.0%}] {msg}".ljust(100))
            sys.stdout.flush()
        elif stage != last["msg"].split("...")[0] or now - last["t"] > 10:
            print(f"[{frac:4.0%}] {msg}", flush=True)
            last["t"] = now
        last["msg"] = msg

    try:
        result = run(a.source, opts, progress)
    except (AnalysisError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    if tty:
        print()

    print(f"\n{result['video_summary']}\n")
    for c in result["clips"]:
        print(f"{c['number']:02d}. [{c['virality_score']:>3}/100] {c['title']}")
        print(f"    {fmt_ts(c['start'])}-{fmt_ts(c['end'])} ({c['duration']:.0f}s)  hook: {c['hook']}")
    print(f"\nSaved to: {result['project_dir']}")
    print("Each clip has an .mp4, a .jpg thumbnail, an .srt caption file and a .txt with title/description/hashtags.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
