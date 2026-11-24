"""Cutline web app. Run: python app.py  (or double-click "Start Cutline.bat")."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import gradio as gr

from cutline import save_api_key
from cutline.analyze import AnalysisError, fmt_ts, has_credentials
from cutline.media import COOKIES_FILE, save_cookies
from cutline.pipeline import Options, run

MAX_SLOTS = 15
COPY = {"buttons": ["copy"]} if int(gr.__version__.split(".")[0]) >= 6 else {"show_copy_button": True}


def key_status() -> str:
    return "✅ API key is set." if has_credentials() else "⚠️ No API key yet. Paste it below and click Save."


def on_save_key(key: str):
    key = (key or "").strip()
    if len(key) < 20 or " " in key:
        raise gr.Error("That doesn't look like a Gemini API key. Copy it from https://aistudio.google.com/apikey")
    save_api_key(key)
    return "", key_status()


def cookies_status() -> str:
    if COOKIES_FILE.is_file():
        return "✅ YouTube cookies loaded: links download with your YouTube login."
    return ("Only needed if YouTube says *\"confirm you're not a bot\"*.\n"
            "1. In Chrome, install the **Get cookies.txt LOCALLY** extension and allow it in Incognito "
            "(chrome://extensions, then Details).\n"
            "2. Open an **Incognito** window, sign in to YouTube, then go to youtube.com/robots.txt.\n"
            "3. Click the extension and **Export**, then **close the Incognito window** (so YouTube "
            "doesn't replace these cookies).\n"
            "4. Drop the downloaded file here.")


def on_cookies(path):
    if not path:
        return gr.skip(), cookies_status()
    try:
        message = save_cookies(Path(path))
    except (OSError, ValueError) as e:
        raise gr.Error(str(e)) from e
    gr.Info(message)
    return None, cookies_status()


def open_folder(path: str):
    if not path or not Path(path).exists():
        raise gr.Error("No output folder yet. Make some shorts first.")
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - opening the user's own output folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


IDLE_STATUS = "Paste a link or upload a video, then click **Make shorts**. Progress will show here."
KNOWN_ERRORS = (AnalysisError, FileNotFoundError, ValueError, RuntimeError)


class JobStopped(Exception):
    """Raised inside a job (from its progress callback) when the user clicks Stop."""


class Job:
    """One shorts job, run on the server independently of any browser page.

    The job outlives the page that started it, so a refreshed (or second) tab can
    reconnect and keep showing its progress and, at the end, its results.
    """

    def __init__(self, source: str, opts: Options) -> None:
        self.source, self.opts = source, opts
        self.label = source if "://" in source else Path(source).name
        self.frac, self.msg = 0.0, "Starting..."
        self.started = time.monotonic()
        self.elapsed_final: float | None = None
        self.result: dict | None = None
        self.error: BaseException | None = None
        self.done = threading.Event()
        self.stop_requested = threading.Event()

    @property
    def elapsed(self) -> float:
        return self.elapsed_final if self.elapsed_final is not None else time.monotonic() - self.started

    def _progress(self, frac: float, msg: str) -> None:
        if self.stop_requested.is_set():
            raise JobStopped()  # unwinds the pipeline; run_ffmpeg kills any running ffmpeg
        self.frac, self.msg = frac, msg

    def run(self) -> None:
        try:
            self.result = run(self.source, self.opts, self._progress)
        except BaseException as e:  # reported to the page by follow()
            # After Stop, whatever unwound the job (e.g. the downloader wrapping our signal) means "stopped".
            self.error = JobStopped() if self.stop_requested.is_set() else e
            if not isinstance(self.error, (*KNOWN_ERRORS, JobStopped)):
                traceback.print_exception(e)  # unexpected: keep the details in the console
        finally:
            self.elapsed_final = time.monotonic() - self.started
            if self.result is not None:
                save_last_job(self)
            self.done.set()

    @classmethod
    def finished(cls, label: str, result: dict, elapsed: float) -> "Job":
        job = cls(label, Options())
        job.label, job.result, job.elapsed_final, job.frac = label, result, elapsed, 1.0
        job.done.set()
        return job


LAST_JOB_FILE = Path(os.environ.get("CUTLINE_LAST_JOB")
                     or Path(__file__).resolve().parent / "output" / "_last_job.json")


def save_last_job(job: Job) -> None:
    """Remember the last finished job so a restarted app can show its results again."""
    try:
        LAST_JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_JOB_FILE.write_text(json.dumps({
            "label": job.label, "elapsed": job.elapsed, "project_dir": job.result["project_dir"],
        }, ensure_ascii=False), encoding="utf-8")
    except (OSError, KeyError, TypeError):
        pass  # only a convenience


def load_last_job() -> Job | None:
    try:
        saved = json.loads(LAST_JOB_FILE.read_text(encoding="utf-8"))
        project = Path(saved["project_dir"])
        result = json.loads((project / "clips.json").read_text(encoding="utf-8"))
        if not all(Path(c["files"]["video"]).is_file() for c in result["clips"]):
            return None  # clips were moved or deleted
        return Job.finished(saved["label"], result, float(saved.get("elapsed") or 0))
    except (OSError, ValueError, KeyError, TypeError):
        return None


_jobs_lock = threading.Lock()
_current_job: Job | None = load_last_job()  # the running job, or the last finished one


def start_job(source: str, opts: Options) -> tuple[Job, bool]:
    """Start a job unless one is already running. Returns (job, started_new)."""
    global _current_job
    with _jobs_lock:
        if _current_job is not None and not _current_job.done.is_set():
            return _current_job, False
        _current_job = Job(source, opts)
        threading.Thread(target=_current_job.run, daemon=True).start()
        return _current_job, True


def job_running() -> bool:
    job = _current_job
    return job is not None and not job.done.is_set()


def progress_text(job: Job) -> str:
    width = 30
    filled = int(round(max(0.0, min(1.0, job.frac)) * width))
    mins, secs = divmod(int(job.elapsed), 60)
    return (f"### ⏳ Working on it... {job.frac:.0%}\n"
            f"Video: `{job.label}`  \n"
            f"`{'█' * filled}{'░' * (width - filled)}`  \n"
            f"**{job.msg}**  \n"
            f"<small>Elapsed {mins}:{secs:02d}. You can refresh or close this tab; the job keeps running "
            f"and this page reconnects to it.</small>")


def _unchanged() -> list:
    return [gr.skip()] * (2 + len(slot_outputs))  # project_dir, folder button, clip slots


def follow(job: Job):
    """Generator: stream a job's progress to this page, then show its outcome."""
    hide_slots = [gr.update(visible=False) if i % 9 == 0 else gr.skip() for i in range(len(slot_outputs))]
    if not job.done.is_set():
        yield [progress_text(job), gr.skip(), gr.update(visible=False), *hide_slots]
    while not job.done.wait(timeout=1.0):
        yield [progress_text(job), *_unchanged()]
    yield finished_outputs(job)


def finished_outputs(job: Job) -> list:
    if isinstance(job.error, JobStopped):
        return [f"### ⏹ Stopped\nVideo: `{job.label}`\n\nYou can start another video now.", *_unchanged()]
    if job.error is not None:
        # Report without raising: a raised error would stop the chained step
        # that re-enables the "Make shorts" button.
        return [f"### ❌ Something went wrong\nVideo: `{job.label}`\n\n{job.error}", *_unchanged()]

    result = job.result
    clips = result["clips"]
    mins, secs = divmod(int(job.elapsed), 60)
    summary = (f"### ✅ {len(clips)} short{'s' if len(clips) != 1 else ''} ready (took {mins}:{secs:02d})\n"
               f"{result['video_summary']}\n\nSaved in `{result['project_dir']}`")
    updates = [summary, result["project_dir"], gr.update(visible=True)]
    for i in range(MAX_SLOTS):
        if i < len(clips):
            c = clips[i]
            desc = f"{c['description']}\n\n{' '.join(c.get('hashtags') or [])}"
            info = (f"**#{c['number']} · Virality {c['virality_score']}/100** · "
                    f"{fmt_ts(c['start'])}-{fmt_ts(c['end'])} ({c['duration']:.0f}s) · "
                    f"hook: *{c['hook']}*  \n{c['why_it_works']}")
            alts = "\n".join(c.get("alt_titles") or [])
            updates += [
                gr.update(visible=True), c["files"]["video"], info, c["title"], alts, desc,
                c["tiktok_caption"], c["instagram_caption"], ", ".join(c.get("tags") or []),
            ]
        else:
            updates += [gr.update(visible=False), None, "", "", "", "", "", "", ""]
    return updates


def make_shorts(upload, url, num_clips, min_sec, max_sec, layout, captions, hook, whisper_model,
                language, meta_language, instructions, words, highlight, uppercase, reanalyze, fast):
    """Start a job (or join the one already running) and stream it to this page."""
    running = _current_job if job_running() else None
    if running is None:
        source = (url or "").strip() or (upload if isinstance(upload, str) else getattr(upload, "name", None))
        if not source:
            gr.Warning("Upload a video or paste a video link first.")
            yield ["Upload a video or paste a video link first.", *_unchanged()]
            return
        if min_sec >= max_sec:
            gr.Warning("Minimum length must be shorter than maximum length.")
            yield ["Minimum length must be shorter than maximum length.", *_unchanged()]
            return
        opts = Options(
            num_clips=int(num_clips), min_sec=float(min_sec), max_sec=float(max_sec),
            whisper_model=whisper_model, language=(language or "").strip() or None,
            meta_language=(meta_language or "").strip() or None,
            instructions=(instructions or "").strip() or None,
            layout={"Auto": "auto", "Follow faces (crop)": "crop", "Full frame (blur bg)": "fit"}[layout],
            captions=captions, hook=hook, words_per_caption=int(words), highlight=highlight or "#FFE600",
            uppercase=uppercase, reanalyze=reanalyze, fast=bool(fast),
        )
        job, started = start_job(source, opts)
    else:
        job, started = running, False
    if not started:
        gr.Info(f"Still working on {job.label}. Showing its progress; start the next video when it's done.")
    yield from follow(job)
    if job.error is not None and not isinstance(job.error, JobStopped):
        gr.Warning(str(job.error)[:400])


def on_page_load():
    """When a page opens (or is refreshed), reconnect it to the running or last job."""
    job = _current_job
    if job is None or (job.done.is_set() and job.error is not None):
        yield [IDLE_STATUS, *_unchanged()]
        return
    yield from follow(job)  # running: live progress; finished: its results


def lock_button_if_running():
    if job_running():
        return gr.update(interactive=False, value=BUSY_LABEL), gr.update(visible=True, interactive=True)
    return gr.skip(), gr.skip()


def on_stop():
    job = _current_job
    if job is not None and not job.done.is_set():
        job.stop_requested.set()
        gr.Info("Stopping... this can take up to half a minute.")
    return gr.update(interactive=False, value="⏹ Stopping...")


GO_LABEL = "✂️ Make shorts"
BUSY_LABEL = "⏳ Working... (see progress below)"


with gr.Blocks(title="Cutline: long video to viral shorts") as demo:
    gr.Markdown("# ✂️ Cutline\nTurn a long video into vertical shorts with captions, titles, descriptions and hashtags.")

    with gr.Accordion("Gemini API key", open=not has_credentials()):
        key_md = gr.Markdown(key_status())
        with gr.Row():
            key_box = gr.Textbox(label="Gemini API key", type="password", placeholder="AIza...", scale=4)
            key_btn = gr.Button("Save", scale=1)
        gr.Markdown("Get a free key at https://aistudio.google.com/apikey. "
                    "It is saved in the `.env` file in the Cutline folder.")

    with gr.Row():
        with gr.Column(scale=3):
            upload = gr.File(label="Upload a video", file_types=["video"], type="filepath")
            url = gr.Textbox(label="...or paste a video link (YouTube, Vimeo, X, ...)",
                             placeholder="https://www.youtube.com/watch?v=...")
            instructions = gr.Textbox(label="What should the clips focus on? (optional)",
                                      placeholder="e.g. the funniest moments, or the money advice")
        with gr.Column(scale=2):
            num_clips = gr.Slider(1, MAX_SLOTS, value=5, step=1, label="Number of shorts")
            with gr.Row():
                min_sec = gr.Slider(5, 90, value=20, step=5, label="Min length (s)")
                max_sec = gr.Slider(15, 180, value=60, step=5, label="Max length (s)")
            layout = gr.Radio(["Auto", "Follow faces (crop)", "Full frame (blur bg)"], value="Auto",
                              label="Vertical framing")
            with gr.Row():
                captions = gr.Checkbox(value=True, label="Animated captions")
                hook = gr.Checkbox(value=True, label="Hook text at start")
            fast = gr.Checkbox(
                value=True, label="⚡ Fast mode for YouTube links",
                info="Uses YouTube's subtitles and Most replayed data, then downloads only the chosen parts. "
                     "Much faster; falls back to the full process if a video has no subtitles.")

    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            whisper_model = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3", "turbo"], value="small",
                                        label="Speech model (bigger = more accurate, slower)")
            language = gr.Textbox(label="Spoken language code (blank = auto)", placeholder="en, hi, ur, es...")
            meta_language = gr.Textbox(label="Write titles in (blank = same as video)", placeholder="English")
        with gr.Row():
            words = gr.Slider(1, 6, value=3, step=1, label="Words per caption")
            highlight = gr.ColorPicker(value="#FFE600", label="Caption highlight colour")
            uppercase = gr.Checkbox(value=True, label="UPPERCASE captions")
            reanalyze = gr.Checkbox(value=False, label="Re-run Gemini analysis (new picks)")
        with gr.Row():
            cookies_upload = gr.File(label="YouTube cookies file (cookies.txt)", file_types=[".txt"],
                                     type="filepath", scale=1)
            cookies_md = gr.Markdown(cookies_status())

    with gr.Row():
        go = gr.Button(GO_LABEL, variant="primary", size="lg", scale=4)
        stop_btn = gr.Button("⏹ Stop", variant="stop", size="lg", scale=1, visible=False)
    status = gr.Markdown(IDLE_STATUS)
    project_dir = gr.State("")
    folder_btn = gr.Button("📂 Open output folder", size="sm", visible=False)

    slot_outputs = []
    for i in range(MAX_SLOTS):
        with gr.Group(visible=False) as group:
            with gr.Row():
                with gr.Column(scale=2, min_width=260):
                    video = gr.Video(label=f"Short {i + 1}", height=560, interactive=False)
                with gr.Column(scale=3):
                    info = gr.Markdown()
                    title = gr.Textbox(label="Title", **COPY)
                    alts = gr.Textbox(label="Alternative titles", lines=2, **COPY)
                    desc = gr.Textbox(label="Description + hashtags (YouTube)", lines=4, **COPY)
                    tiktok = gr.Textbox(label="TikTok caption", lines=2, **COPY)
                    insta = gr.Textbox(label="Instagram caption", lines=3, **COPY)
                    tags = gr.Textbox(label="YouTube tags", lines=2, **COPY)
        slot_outputs += [group, video, info, title, alts, desc, tiktok, insta, tags]

    key_btn.click(on_save_key, [key_box], [key_box, key_md])
    cookies_upload.upload(on_cookies, [cookies_upload], [cookies_upload, cookies_md])
    job_outputs = [status, project_dir, folder_btn, *slot_outputs]
    unlock = dict(
        fn=lambda: (gr.update(interactive=True, value=GO_LABEL), gr.update(visible=False, value="⏹ Stop")),
        inputs=None, outputs=[go, stop_btn], queue=False)

    # Lock the button, stream the job into the page, then unlock (`.then` runs even on failure).
    # concurrency_limit=None: these handlers only watch the job (the heavy work runs in the job's
    # own thread, one job at a time), so several tabs or a refreshed tab must not queue behind
    # an earlier tab's watcher.
    go.click(
        lambda: (gr.update(interactive=False, value=BUSY_LABEL), gr.update(visible=True, interactive=True)),
        None, [go, stop_btn], queue=False,
    ).then(
        make_shorts,
        [upload, url, num_clips, min_sec, max_sec, layout, captions, hook, whisper_model, language,
         meta_language, instructions, words, highlight, uppercase, reanalyze, fast],
        job_outputs, show_progress="hidden", concurrency_limit=None,
    ).then(**unlock)

    # Opening or refreshing the page reconnects it to a running job (or shows the last results).
    demo.load(lock_button_if_running, None, [go, stop_btn], queue=False).then(
        on_page_load, None, job_outputs, show_progress="hidden", concurrency_limit=None,
    ).then(**unlock)
    stop_btn.click(on_stop, None, stop_btn, queue=False)
    folder_btn.click(open_folder, [project_dir], None)


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    demo.queue().launch(inbrowser=True, allowed_paths=[str(output_dir)])
