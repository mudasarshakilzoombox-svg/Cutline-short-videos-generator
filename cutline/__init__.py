"""Cutline: turn long videos into short, captioned, vertical clips with titles and descriptions."""

import os
from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # must be set before cv2 is imported


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from <project>/.env without overriding real env vars."""
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def save_api_key(key: str) -> None:
    """Persist a Gemini API key to <project>/.env and the current process."""
    key = key.strip()
    env_file = ROOT / ".env"
    lines = []
    if env_file.is_file():
        lines = [
            l for l in env_file.read_text(encoding="utf-8").splitlines()
            if not l.strip().startswith("GEMINI_API_KEY=")
        ]
    lines.append(f"GEMINI_API_KEY={key}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["GEMINI_API_KEY"] = key


_load_dotenv()
