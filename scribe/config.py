"""Settings and paths. Runtime state lives in %LOCALAPPDATA%\\Scribe, deliverables in Documents\\Scribe."""
from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime
from pathlib import Path

log = logging.getLogger("scribe.config")

APP_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Scribe"
SETTINGS_PATH = APP_DIR / "settings.json"
MEDIA_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}

DEFAULTS: dict = {
    "owner_name": "",
    "languages": ["en"],
    "vocabulary": ["Claude"],
    "dictation": {"model": "large-v3-turbo", "strip_fillers": True, "min_seconds": 0.4, "enroll_voice": True,
                  "cleanup": {"enabled": True, "model": "gemma4:e4b", "timeout_s": 2.5,
                              "keep_alive": "30m", "timeout_per_word_s": 0.01}},
    "meetings": {"model": "large-v3", "cluster_threshold": 0.5, "owner_match": 0.6},
    "vaults": {},
    "default_vault": "Local",
    "picker_timeout_s": 20,
    "claude": {"exe": "%USERPROFILE%\\.local\\bin\\claude.exe", "model": "claude-opus-5-5", "timeout_s": 7200},
    "ollama": {"url": "http://127.0.0.1:11434", "vision_model": "qwen3.6:27b-q4_K_M"},
    "output_root": "%USERPROFILE%\\Documents\\Scribe",
    "test_accept_injected_keys": False,
    "notetaker_hotkey": "ctrl+alt+n",
    "learn": {"enabled": True, "poll_s": 1.5, "max_s": 120.0, "settle_s": 4.0},
    "widget": {"idle_pill": True, "anchor": "right"},      # anchor: "right" (above the clock) or "centre"
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def expand(value: str) -> str:
    return os.path.expandvars(value)


def load_settings(path: Path | None = None) -> dict:
    """settings.json merged over DEFAULTS. Writes DEFAULTS on first run so the user has a file to edit."""
    path = path or SETTINGS_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
        return copy.deepcopy(DEFAULTS)
    try:
        return _merge(DEFAULTS, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        log.error("settings.json unreadable, using defaults: %r", exc)
        return copy.deepcopy(DEFAULTS)


def output_root(settings: dict | None = None) -> Path:
    settings = settings or load_settings()
    return Path(expand(settings["output_root"]))


def recordings_dir(settings: dict | None = None) -> Path:
    p = output_root(settings) / "Recordings"
    p.mkdir(parents=True, exist_ok=True)
    return p


def inbox_dir(vault: str, settings: dict | None = None) -> Path:
    p = output_root(settings) / "Inbox" / vault
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = APP_DIR / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = APP_DIR / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_job_folder(stamp: datetime | None = None, settings: dict | None = None) -> Path:
    """Recordings\\YYYY-MM-DD HHMM, then -2, -3 on a clash. Created on return."""
    base = recordings_dir(settings) / (stamp or datetime.now()).strftime("%Y-%m-%d %H%M")
    folder, n = base, 1
    while folder.exists():
        n += 1
        folder = base.with_name(f"{base.name}-{n}")
    folder.mkdir(parents=True)
    return folder


def vault_names(settings: dict) -> list[str]:
    return [*settings["vaults"].keys(), "Local"]
