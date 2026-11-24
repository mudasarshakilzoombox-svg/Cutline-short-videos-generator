"""Word-by-word animated captions (ASS, burned in by ffmpeg) plus plain SRT files."""

from __future__ import annotations

OUT_W, OUT_H = 1080, 1920
BREAK_PUNCT = (".", ",", "?", "!", ";", ":")


def _ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _srt_time(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """'#RRGGBB' -> ASS '&HAABBGGRR'."""
    v = hex_rgb.strip().lstrip("#")
    if len(v) != 6:
        v = "FFE600"
    r, g, b = v[0:2], v[2:4], v[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def relative_words(words: list[dict], clip_start: float, clip_dur: float) -> list[dict]:
    out = []
    for w in words:
        s = max(0.0, w["start"] - clip_start)
        e = min(clip_dur, w["end"] - clip_start)
        if e > s and w["word"]:
            out.append({"start": s, "end": e, "word": w["word"]})
    return out


def chunk_words(words: list[dict], max_words: int = 3, max_chars: int = 18) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur:
            chars = sum(len(x["word"]) + 1 for x in cur) + len(w["word"])
            gap = w["start"] - cur[-1]["end"]
            if (len(cur) >= max_words or chars > max_chars or gap > 0.6
                    or cur[-1]["word"].endswith(BREAK_PUNCT)):
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def build_ass(
    words: list[dict],
    clip_dur: float,
    *,
    hook: str | None = None,
    hook_dur: float = 3.2,
    max_words: int = 3,
    highlight: str = "#FFE600",
    uppercase: bool = True,
    font: str = "Arial Black",
    captions: bool = True,
) -> str:
    hl = _ass_color(highlight)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,7,3,2,70,70,540,1
Style: Hook,{font},62,&H00111111,&H00111111,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,22,0,8,90,90,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    if hook:
        text = _escape(hook.upper() if uppercase else hook)
        events.append(
            f"Dialogue: 1,{_ass_time(0)},{_ass_time(min(hook_dur, clip_dur))},Hook,,0,0,0,,{{\\fad(120,250)}}{text}"
        )

    if captions:
        chunks = chunk_words(words, max_words=max_words)
        for ci, chunk in enumerate(chunks):
            tokens = [_escape(w["word"].upper() if uppercase else w["word"]) for w in chunk]
            next_start = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else clip_dur
            chunk_end = min(next_start, chunk[-1]["end"] + 0.5, clip_dur)
            for wi, w in enumerate(chunk):
                ev_start = chunk[0]["start"] if wi == 0 else w["start"]
                ev_end = chunk_end if wi == len(chunk) - 1 else chunk[wi + 1]["start"]
                if ev_end - ev_start < 0.01:
                    continue
                parts = [
                    f"{{\\c{hl}&\\fscx108\\fscy108}}{t}{{\\c&HFFFFFF&\\fscx100\\fscy100}}" if j == wi else t
                    for j, t in enumerate(tokens)
                ]
                events.append(
                    f"Dialogue: 0,{_ass_time(ev_start)},{_ass_time(ev_end)},Caption,,0,0,0,,{' '.join(parts)}"
                )
    return header + "\n".join(events) + "\n"


def build_srt(words: list[dict], clip_dur: float, max_words: int = 7) -> str:
    lines = []
    for i, chunk in enumerate(chunk_words(words, max_words=max_words, max_chars=42), 1):
        end = min(chunk[-1]["end"] + 0.3, clip_dur)
        lines.append(f"{i}\n{_srt_time(chunk[0]['start'])} --> {_srt_time(end)}\n"
                     + " ".join(w["word"] for w in chunk) + "\n")
    return "\n".join(lines)
