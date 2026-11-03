"""Scribe for Claude Code sessions: an MCP server on stdio, so a session can see what Scribe is doing, read its
transcripts and drive the notetaker.

    .venv\\Scripts\\pythonw.exe -m scribe.mcp_server        (Claude Code starts it; install.ps1 registers it)

It talks to the running Scribe only through Scribe's own IPC (app.send_commands: the authenticated local socket
and key a second launch uses) and reads job folders directly; it opens no port of its own. Every tool reads,
except start_notetaker, stop_notetaker and send_to_vault, which act through the app as the tray does.
Transcript text goes back to the calling session only. It is never logged: the server's log
(%LOCALAPPDATA%\\Scribe\\logs\\mcp.log) holds ids, kinds and counts.

live_transcript transcribes a recording still in progress. It copies the tracks the recorder is writing (both
handles share the file, so the recorder never notices), mixes the copies, and transcribes them in a child process.
The child shares the GPU with whatever else runs on it: it waits for nothing, and a GPU out of memory is handled
as in every Scribe transcription (asr.transcribe_file: retried, then finished on the CPU).
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from multiprocessing import connection as mpc
from pathlib import Path
from typing import Callable, Literal

from . import app, config, draft, jobs

log = logging.getLogger("scribe.mcp")

SERVER_NAME = "scribe"
MODULE = "scribe.mcp_server"          # the child is started by name: run with -m, this module is __main__
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LIVE_MODEL = "large-v3-turbo"
KINDS = ("final", "draft", "notes", "speakers")
IPC_WAIT_S = 2.0              # a refused connection is retried this long (Scribe still starting up)
IPC_TIMEOUT_S = 10.0          # Scribe's answer to one command
TOGGLE_WAIT_S = 15.0          # start/stop: how long the new state is waited for
POLL_S = 0.25
MAX_CHARS = 40000             # one window of transcript text (about 10,000 tokens)
MIN_CHARS, LIMIT_CHARS = 1000, 200000
RECENT_MAX = 100
CHILD_START_S = 300.0         # the transcription child's first chunk (start-up, decoding, VAD, model load) ...
CHILD_STALL_S = 120.0         # ... and each next one must finish within this, or the child is ended as hung
TOO_SHORT_S = app.MIN_MEETING_S


class ScribeDown(RuntimeError):
    """Scribe is not running, or did not take the key: nothing was delivered."""


class NoAnswer(RuntimeError):
    """The command reached Scribe but no answer came back. It may have run, so it is never resent."""


class NothingYet(RuntimeError):
    """The folder holds no audio to transcribe yet."""


# ---------------------------------------------------------------- the running Scribe, over its own IPC

class Link:
    """The running Scribe through the socket, key and JSON messages a second launch uses (app.send_commands)."""

    def __init__(self, key_path: Path | None = None, port: int = app.IPC_PORT, wait_s: float = IPC_WAIT_S,
                 timeout: float = IPC_TIMEOUT_S):
        self.key_path = Path(key_path) if key_path else config.APP_DIR / app.KEY_FILE
        self.port, self.wait_s, self.timeout = port, wait_s, timeout

    def ask(self, commands: list[dict]) -> dict:
        """Scribe's reply. Raises ScribeDown before anything was delivered, NoAnswer after."""
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                reply = app.send_commands(commands, app.read_key(self.key_path), self.port, timeout=self.timeout)
            except app.NoReply as exc:
                raise NoAnswer(str(exc)) from None
            except (OSError, EOFError, mpc.AuthenticationError, ValueError) as exc:
                if time.monotonic() >= deadline:
                    raise ScribeDown(type(exc).__name__) from None
                time.sleep(POLL_S)
                continue
            return reply if isinstance(reply, dict) else {"ok": False, "error": "reply not an object"}

    def status(self) -> dict | None:
        """The app's status() document, or None when Scribe is not running."""
        try:
            reply = self.ask([{"cmd": "status"}])
        except ScribeDown:
            return None
        status = reply.get("status")
        return status if isinstance(status, dict) else None


# ---------------------------------------------------------------- text

def render_lines(lines: list[dict], offset: float = 0.0) -> str:
    """speakers.json lines or Whisper segments as "[mm:ss] Speaker: text", one per line ("[mm:ss] text" when
    a line has no speaker). offset moves every time, for a transcript of a recording's last minutes."""
    out = []
    for ln in lines or []:
        text = " ".join(str(ln.get("text") or "").split())
        if not text:
            continue
        who = str(ln.get("speaker") or "").strip()
        stamp = draft.fmt_ts(float(ln.get("start") or 0.0) + offset)
        out.append(f"[{stamp}] {who}: {text}" if who else f"[{stamp}] {text}")
    return "\n".join(out)


def page(text: str, offset: int = 0, max_chars: int = MAX_CHARS) -> str:
    """One window of a long text, cut at a line end, with a closing line naming the offset of the next window."""
    offset = max(0, int(offset))
    size = min(LIMIT_CHARS, max(MIN_CHARS, int(max_chars)))
    if offset >= len(text):
        return f"[end: the text is {len(text)} characters]"
    end = offset + size
    if end >= len(text):
        return text[offset:]
    cut = text.rfind("\n", offset, end)
    cut = end if cut <= offset else cut + 1
    chunk = text[offset:cut]
    return chunk + ("" if chunk.endswith("\n") else "\n") + \
        f"[{len(text) - cut} more characters: call again with offset={cut}]"


def tail(text: str, max_chars: int = MAX_CHARS) -> str:
    """The end of a live transcript when the whole would not fit: the newest lines matter most."""
    if len(text) <= max_chars:
        return text
    cut = text.find("\n", len(text) - max_chars)
    cut = len(text) - max_chars if cut < 0 else cut + 1
    lines = text.count("\n", 0, cut)
    return f"[{lines} earlier lines cut: ask with last_minutes]\n" + text[cut:]


# ---------------------------------------------------------------- live transcription

def snapshot(src: Path, dst: Path) -> int:
    """Copy what a track holds now; the bytes copied. The recorder keeps writing meanwhile: both handles share
    the file, the copy stops at the size seen when it opened, and a frame cut off there is dropped when the
    copy is decoded (audio._TrackReader reads an unfinished FLAC through ffmpeg)."""
    copied = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        left = os.fstat(fin.fileno()).st_size
        while left > 0:
            block = fin.read(min(left, 1 << 20))
            if not block:
                break
            fout.write(block)
            copied += len(block)
            left -= len(block)
    return copied


def _cut(src: Path, dst: Path, start_s: float) -> Path:
    """The audio from start_s to the end, as 16-bit FLAC."""
    import soundfile as sf

    with sf.SoundFile(str(src)) as f:
        rate = f.samplerate
        f.seek(min(f.frames, int(start_s * rate)))
        data = f.read(dtype="int16")
    sf.write(str(dst), data, rate, format="FLAC", subtype="PCM_16")
    return dst


def _child_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(PACKAGE_ROOT), env.get("PYTHONPATH", "")) if p)
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_OFFLINE"] = "1"
    return env


def _tail_lines(path: Path, n: int = 2) -> str:
    try:
        return " | ".join(path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-n:])
    except OSError:
        return ""


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def watch_child(proc, beats: Path) -> int:
    """The child's exit code, waited for while it shows progress: CHILD_START_S for its first chunk, then
    CHILD_STALL_S between chunks (it adds a line to `beats` for each, and at each GPU wait). A child quiet for
    longer is hung (a CUDA stall, a deadlock): it is ended and RuntimeError raised. A long recording that keeps
    going, on the CPU too, is never cut off."""
    seen, since, limit = 0, time.monotonic(), CHILD_START_S
    while True:
        try:
            return proc.wait(timeout=POLL_S)
        except subprocess.TimeoutExpired:
            pass
        size = _size(beats)
        if size > seen:
            seen, since, limit = size, time.monotonic(), CHILD_STALL_S
        elif time.monotonic() - since > limit:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeError(f"transcription made no progress for {limit:.0f} s; ended")


def transcribe_child(audio_path: Path, out_path: Path, model: str, settings: dict, workdir: Path) -> dict:
    """asr.transcribe_file in a child Python (no window), so CUDA, the model and its memory end with the child,
    not with this server, which lives as long as the Claude session. The child is ended if it stops making
    progress (watch_child)."""
    audio_path, out_path, workdir = Path(audio_path), Path(out_path), Path(workdir)
    request = workdir / "transcribe.json"
    beats = workdir / "progress.txt"
    jobs.write_json_atomic(request, {
        "audio": str(audio_path), "out": str(out_path), "model": model,
        "languages": list(jobs._setting(settings, "languages")),
        "vocabulary": list(jobs._setting(settings, "vocabulary")),
        "progress": str(beats),
    })
    out_path.unlink(missing_ok=True)
    log_path = workdir / "transcribe.log"
    cmd = [sys.executable, "-m", MODULE, "transcribe", "--request", str(request)]
    with open(log_path, "wb") as fh:
        proc = subprocess.Popen(cmd, cwd=str(PACKAGE_ROOT), env=_child_env(), stdin=subprocess.DEVNULL, stdout=fh,
                                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            returncode = watch_child(proc, beats)
        finally:
            if proc.poll() is None:         # the call itself was interrupted
                proc.kill()
    if returncode != 0 or not out_path.is_file():
        log.warning("transcription child exited %d", returncode)
        raise RuntimeError(f"transcription failed ({returncode}): {_tail_lines(log_path)}")
    return jobs.read_json(out_path)


def label_child(audio_path: Path, whisper_path: Path, out_path: Path, settings: dict, app_dir: Path,
                workdir: Path) -> dict:
    """Speaker labels as the pipeline's labelling step makes them (diarize.label_job: a child process, CPU)."""
    diarize = jobs._mod("diarize")
    models = Path(app_dir) / "models"
    diarize.segmentation_model(models)      # a clear "missing" before any child starts
    diarize.embedding_model(models)
    return diarize.label_job(
        audio_path, whisper_path, out_path, models,
        threshold=float(jobs._setting(settings, "meetings", "cluster_threshold")),
        owner_name=jobs._setting(settings, "owner_name"),
        owner_match=float(jobs._setting(settings, "meetings", "owner_match")),
        voiceprint_path=Path(app_dir) / "voiceprint.json",
        log_path=Path(workdir) / "labelling.log",
    )


class Live:
    """Transcribes what a job folder holds so far: a recording in progress, a finished one, or a dropped file.
    Everything it writes goes to a temp folder that is removed at the end; the job folder is only read."""

    def __init__(self, *, transcribe: Callable | None = None, label: Callable | None = None,
                 app_dir: Path | None = None):
        self.transcribe = transcribe or transcribe_child
        self.label = label or label_child
        self.app_dir = Path(app_dir) if app_dir else config.APP_DIR

    def audio(self, folder: Path, job: jobs.Job | None, workdir: Path, recording: bool) -> Path:
        """The audio to transcribe: the job's own prepared audio once it exists, else a mix of copies of the
        tracks as they stand, else the dropped file's audio track."""
        if job is not None and not recording and jobs._done(job, "audio"):
            return jobs._out(job, "audio")
        tracks = [folder / n for n in ("mic.flac", "system.flac") if (folder / n).is_file()]
        if tracks:
            for t in tracks:
                snapshot(t, workdir / t.name)
            audio = jobs._mod("audio")
            try:
                return Path(audio.mix_tracks(workdir))
            except audio.AudioError as exc:
                if not str(exc).startswith(("no tracks", "tracks in")):    # ffmpeg missing, not an empty recording
                    raise
                raise NothingYet("no audio recorded yet") from None
        if job is not None and job.kind == "file" and job.source and Path(job.source).is_file():
            return Path(jobs._mod("video").extract_audio(Path(job.source), workdir / "audio.flac"))
        raise NothingYet("no audio in this folder")

    def run(self, folder: Path, job: jobs.Job | None, settings: dict, *, recording: bool, speakers: bool = True,
            last_minutes: float | None = None) -> dict:
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="scribe-live-", ignore_cleanup_errors=True) as tmp:
            workdir = Path(tmp)
            src = self.audio(Path(folder), job, workdir, recording)
            duration = float(sf.info(str(src)).duration)
            offset = 0.0
            if last_minutes and last_minutes > 0 and duration > last_minutes * 60.0:
                offset = duration - last_minutes * 60.0
                src = _cut(src, workdir / "tail.flac", offset)
                duration = float(sf.info(str(src)).duration)
            whisper_path = workdir / "whisper.json"
            whisper = self.transcribe(src, whisper_path, LIVE_MODEL, settings, workdir)
            segments = whisper.get("segments") or []
            lines, note = None, "off"
            if speakers and segments:
                try:
                    doc = self.label(src, whisper_path, workdir / "speakers.json", settings, self.app_dir, workdir)
                    lines = doc.get("lines") or None
                    note = "owner matched" if doc.get("owner_matched") else "owner not matched"
                except Exception as exc:
                    log.warning("live speaker labels unavailable: %s", type(exc).__name__)
                    note = f"unavailable ({' '.join(str(exc).split())[:160] or type(exc).__name__})"
            return {"duration": duration, "offset": offset, "segments": segments, "lines": lines,
                    "speakers": note, "languages": whisper.get("languages") or {}}


# ---------------------------------------------------------------- what the tools do

def _same_path(a, b) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _read_json_quietly(path: Path):
    try:
        return jobs.read_json(path)
    except (OSError, ValueError):
        return None


def _when(folder: Path, job: jobs.Job | None) -> str | None:
    try:
        return datetime.strptime(folder.name[:15], "%Y-%m-%d %H%M").strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return job.created[:16].replace("T", " ") if job else None


class Scribe:
    """The tools' work, apart from the protocol: settings, job folders, the IPC link and live transcription."""

    def __init__(self, *, settings_fn: Callable[[], dict] = config.load_settings, link: Link | None = None,
                 live: Live | None = None):
        self.settings_fn = settings_fn
        self.link = link or Link()
        self.live_runner = live or Live()

    # -- helpers

    def folder(self, settings: dict, job_id: str) -> Path:
        """A recording's folder by its id (the folder's name under Recordings). Nothing outside it is reachable."""
        name = str(job_id or "").strip()
        if not name or name in (".", "..") or any(c in name for c in "\\/:"):
            raise ValueError(f"not a recording id: {job_id!r}")
        folder = config.recordings_dir(settings) / name
        if not folder.is_dir():
            raise ValueError(f"no recording {name!r}; recent_recordings lists them")
        return folder

    @staticmethod
    def job(folder: Path) -> jobs.Job | None:
        if not (folder / jobs.JOB_FILE).exists():
            return None
        try:
            return jobs.Job.load(folder)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{folder.name}: job.json unreadable ({type(exc).__name__})") from None

    def _status_quietly(self) -> dict | None:
        try:
            return self.link.status()
        except NoAnswer:
            return None

    @staticmethod
    def _recording_folder(status: dict | None) -> str | None:
        return status.get("recording_folder") if status and status.get("recording") else None

    @staticmethod
    def vault_file(job: jobs.Job, settings: dict, key: str) -> Path | None:
        """The final transcript or notes Claude shelved, from job.json outputs: only inside that vault."""
        rel = job.outputs.get(key)
        vault = (settings.get("vaults") or {}).get(job.vault)
        if not rel or not vault:
            return None
        path = jobs._in_vault(Path(config.expand(str(vault))), str(rel))
        return path if path is not None and path.is_file() else None

    def outputs(self, job: jobs.Job | None, settings: dict) -> list[str]:
        if job is None:
            return []
        have = {"final": self.vault_file(job, settings, "transcript") is not None,
                "draft": jobs._done(job, "draft"),
                "notes": self.vault_file(job, settings, "notes") is not None,
                "speakers": jobs._done(job, "speakers")}
        return [k for k in KINDS if have[k]]

    # -- status

    def status(self) -> dict:
        st = self.link.status()
        if st is None:
            return {"running": False}
        folder = self._recording_folder(st)
        elapsed = float(st.get("elapsed") or 0.0)
        return {
            "running": True,
            "recording": folder is not None,
            "recording_id": Path(folder).name if folder else None,
            "elapsed": draft.fmt_ts(elapsed) if folder else None,
            "elapsed_s": round(elapsed, 1) if folder else None,
            "dictation": st.get("dictation"),
            "jobs": [{k: j.get(k) for k in ("id", "kind", "vault", "status", "waiting", "error")}
                     for j in st.get("jobs") or [] if isinstance(j, dict)],
        }

    # -- recent recordings

    def recent(self, limit: int = 10) -> list[dict]:
        settings = self.settings_fn()
        root = config.recordings_dir(settings)
        current = self._recording_folder(self._status_quietly())
        limit = min(RECENT_MAX, max(1, int(limit)))
        folders = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)[:limit]
        return [self.entry(p, settings, current) for p in folders]

    def entry(self, folder: Path, settings: dict, current: str | None) -> dict:
        rec = _read_json_quietly(folder / jobs.RECORDING_FILE)
        rec = rec if isinstance(rec, dict) else None
        try:
            job = self.job(folder)
        except ValueError:
            return {"id": folder.name, "time": _when(folder, None), "status": "unreadable"}
        if current and _same_path(folder, current):
            status = "recording"
        elif job is not None:
            status = job.status
        elif rec is not None and rec.get("status") == "recording":
            status = "interrupted"      # recording.json was never closed and Scribe is not recording it
        elif rec is not None:
            status = "stopped"          # stopped before a vault was picked; Scribe queues it Local at start-up
        else:
            status = "empty"
        seconds = rec.get("seconds") if rec is not None else None
        if seconds is None and job is not None and jobs._done(job, "audio"):
            try:
                import soundfile as sf

                seconds = float(sf.info(str(jobs._out(job, "audio"))).duration)
            except Exception:
                seconds = None
        entry = {
            "id": folder.name,
            "time": _when(folder, job),
            "length": draft.fmt_ts(float(seconds)) if seconds is not None else None,
            "kind": job.kind if job is not None else ("meeting" if rec is not None else None),
            "vault": job.vault if job is not None else None,
            "status": status,
            "outputs": self.outputs(job, settings),
        }
        if job is not None and job.kind == "file" and job.source:
            entry["source"] = Path(job.source).name
        if job is not None and job.error:
            entry["error"] = job.error
        return entry

    # -- reading outputs

    def transcript(self, job_id: str, kind: str = "final", offset: int = 0, max_chars: int = MAX_CHARS) -> str:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        settings = self.settings_fn()
        folder = self.folder(settings, job_id)
        job = self.job(folder)
        if job is None:
            return f"{folder.name} has no job yet (still recording, or no vault picked); live_transcript reads it."
        if kind in ("final", "notes"):
            key = "transcript" if kind == "final" else "notes"
            path = self.vault_file(job, settings, key)
            if path is None:
                return self._gone(job, settings, kind, key) or self._missing(job, kind)
            text = path.read_text(encoding="utf-8", errors="replace")
        elif kind == "draft":
            if not jobs._done(job, "draft"):
                return self._missing(job, kind)
            text = jobs._out(job, "draft").read_text(encoding="utf-8", errors="replace")
        else:
            if not jobs._done(job, "speakers"):
                return self._missing(job, kind)
            doc = jobs.read_json(jobs._out(job, "speakers"))
            text = render_lines(doc.get("lines") or [] if isinstance(doc, dict) else [])
        log.info("transcript %s %s: %d characters", folder.name, kind, len(text))
        return page(text, offset, max_chars)

    @staticmethod
    def _gone(job: jobs.Job, settings: dict, kind: str, key: str) -> str | None:
        """The answer when job.json names a shelved file inside the vault that is no longer there; else None."""
        rel = job.outputs.get(key)
        vault = (settings.get("vaults") or {}).get(job.vault)
        if not rel or not vault or jobs._in_vault(Path(config.expand(str(vault))), str(rel)) is None:
            return None
        log.info("transcript %s %s: the shelved file is gone", job.id, kind)
        where = f"{job.vault}\\" + str(rel).replace("/", "\\")
        left = [k for k in ("draft", "speakers") if jobs._done(job, k)]
        still = f" Scribe still holds its {' and '.join(left)} (kind {left[0]})." if left else ""
        what, was, are, it = (("final transcript", "was", "is", "it") if kind == "final"
                              else ("notes", "were", "are", "they"))
        return (f"The {what} of {job.id} {was} shelved at {where} and {are} no longer there: {it} {was} moved or "
                f"deleted in the vault.{still}")

    @staticmethod
    def _missing(job: jobs.Job, kind: str) -> str:
        where = f"{job.id}: vault {job.vault}, status {job.status}"
        if kind in ("final", "notes") and job.vault == jobs.LOCAL:
            return f"No {kind} for {where}. A Local job's result is its draft (kind draft); send_to_vault sends it on."
        if job.status in jobs.TERMINAL:
            return f"No {kind} for {where}" + (f" ({job.error})." if job.error else ".")
        return f"No {kind} yet for {where}. live_transcript reads the audio now."

    # -- live transcript

    def live(self, job_id: str = "current", speakers: bool = True, last_minutes: float | None = None) -> str:
        settings = self.settings_fn()
        status = self._status_quietly()
        current = self._recording_folder(status)
        if str(job_id or "").strip().lower() in ("", "current"):
            if not current:
                return "Scribe is not recording. Pass a recording id (recent_recordings lists them)."
            folder = Path(current)
        else:
            folder = self.folder(settings, job_id)
        recording = current is not None and _same_path(folder, current)
        job = self.job(folder)
        began = time.monotonic()
        try:
            result = self.live_runner.run(folder, job, settings, recording=recording, speakers=speakers,
                                          last_minutes=last_minutes)
        except NothingYet as exc:
            return f"{folder.name}: {exc}."
        rows = result["lines"] or result["segments"]
        body = render_lines(rows, result["offset"])
        state = "recording" if recording else (job.status if job is not None else "stopped")
        span = draft.fmt_ts(result["duration"] + result["offset"])
        part = f", last {draft.fmt_ts(result['duration'])}" if result["offset"] else ""
        langs = ", ".join(f"{k} {v:.0f} s" for k, v in result["languages"].items() if v)
        header = (f"{folder.name} · {state} · {span} so far{part} · whisper {LIVE_MODEL}"
                  f" · speakers: {result['speakers']}" + (f" · {langs}" if langs else ""))
        log.info("live transcript %s: %.0f s of audio, %d lines, %.1f s", folder.name, result["duration"], len(rows),
                 time.monotonic() - began)
        return header + "\n\n" + (tail(body) if body else "(no speech yet)")

    # -- acting

    def _wait_status(self, done: Callable[[dict], bool], wait_s: float | None = None) -> dict | None:
        deadline = time.monotonic() + (TOGGLE_WAIT_S if wait_s is None else wait_s)
        while True:
            st = self._status_quietly()
            if st is not None and done(st):
                return st
            if time.monotonic() >= deadline:
                return st
            time.sleep(POLL_S)

    def start(self) -> str:
        st = self.link.status()
        if st is None:
            return "Scribe is not running."
        if st.get("recording"):
            folder = self._recording_folder(st)
            return (f"Already recording: {Path(folder).name if folder else '?'}, "
                    f"{draft.fmt_ts(float(st.get('elapsed') or 0))}. Nothing changed.")
        self.link.ask([{"cmd": "toggle"}])
        st = self._wait_status(lambda s: bool(s.get("recording")))
        folder = self._recording_folder(st)
        if folder:
            log.info("notetaker started: %s", Path(folder).name)
            return f"Recording: {Path(folder).name}. stop_notetaker ends it."
        return ("Scribe did not start recording. It ignores a toggle within 0.8 s of the last one or while a start "
                "or stop is under way, and a microphone that will not open shows ✕ on screen.")

    def stop(self) -> str:
        st = self.link.status()
        if st is None:
            return "Scribe is not running."
        folder = self._recording_folder(st)
        if not folder:
            return "Not recording. Nothing changed."
        self.link.ask([{"cmd": "toggle"}])
        st = self._wait_status(lambda s: not s.get("recording"))
        if st is not None and st.get("recording"):
            return "Scribe is still recording; the stop was not taken. Try again in a moment."
        rec = _read_json_quietly(Path(folder) / jobs.RECORDING_FILE)
        seconds = float(rec.get("seconds") or 0.0) if isinstance(rec, dict) else 0.0
        name = Path(folder).name
        log.info("notetaker stopped: %s, %.1f s", name, seconds)
        if seconds and seconds < TOO_SHORT_S:
            return f"Stopped: {name}, {draft.fmt_ts(seconds)}. Too short to transcribe; kept on disk, not queued."
        wait = int(self.settings_fn().get("picker_timeout_s", 20))
        return (f"Stopped: {name}, {draft.fmt_ts(seconds)}. Scribe's vault picker is on screen for {wait} s; "
                f"with no pick it stays Local, and send_to_vault sends it on later.")

    def send(self, job_id: str, vault: str) -> str:
        settings = self.settings_fn()
        vaults = list((settings.get("vaults") or {}).keys())
        if vault not in vaults:
            if not vaults:
                raise ValueError(f"unknown vault {vault!r}; no vaults are set in Scribe's settings, so recordings "
                                 f"stay Local")
            raise ValueError(f"unknown vault {vault!r}; vaults: {', '.join(vaults)}")
        folder = self.folder(settings, job_id)
        job = self.job(folder)
        if job is None:
            return f"{folder.name} has no job yet; it can be sent once it is done."
        if job.vault != jobs.LOCAL:
            return f"{job.id} already went to {job.vault} (status {job.status}). Nothing changed."
        if job.status != "done":
            return f"{job.id} is not finished (status {job.status}); send it once it is done."
        try:
            reply = self.link.ask([{"cmd": "enhance", "folders": [str(folder)], "vault": vault}])
        except ScribeDown:
            return "Scribe is not running; nothing was sent."
        if reply.get("ok") and reply.get("sent") == 1:
            log.info("sent %s to %s", job.id, vault)
            return (f"Queued: {job.id} to {vault}. A Claude run in {vault} corrects the draft and shelves the "
                    f"transcript, recording and notes there; status shows its progress.")
        return f"Scribe did not queue {job.id} (reply: {json.dumps(reply)[:200]})."


# ---------------------------------------------------------------- the MCP server

INSTRUCTIONS = (
    "Scribe is a local dictation and notetaker app on this PC. status, recent_recordings, transcript "
    "and live_transcript only read. start_notetaker, stop_notetaker and send_to_vault act through the app, as "
    "its tray does."
)


def _call(fn: Callable[[], object]) -> str:
    """A tool's work, as text: JSON for documents, plain text otherwise; Scribe's own states read in plain words."""
    try:
        out = fn()
    except ScribeDown:
        return "Scribe is not running."
    except NoAnswer:
        return "Scribe took the command but did not answer; status shows whether it ran."
    return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=1)


def build_server(scribe: Scribe | None = None):
    """The FastMCP server and its tools. Each tool runs on a worker thread, so the protocol loop stays live."""
    import anyio
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    scribe = scribe or Scribe()
    root = logging.getLogger()
    before = list(root.handlers), root.level
    server = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS, log_level="WARNING")
    # FastMCP gives a root logger with no handler one on stderr, which Claude Code keeps in its MCP log. This
    # server logs to mcp.log only (serve), so that handler is taken off again.
    for h in list(root.handlers):
        if h not in before[0]:
            root.removeHandler(h)
    root.setLevel(before[1])

    reads = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    async def run(fn: Callable[[], object]) -> str:
        return await anyio.to_thread.run_sync(lambda: _call(fn))

    @server.tool(annotations=reads, structured_output=False)
    async def status() -> str:
        """Scribe's state now: whether the notetaker is recording and for how long, the dictation model, and the
        jobs Scribe has seen since it started, with their status and vault. Read-only."""
        return await run(scribe.status)

    @server.tool(annotations=reads, structured_output=False)
    async def recent_recordings(limit: int = 10) -> str:
        """The newest recordings in Documents\\Scribe\\Recordings, newest first: id, time, length, kind, vault,
        status, and the outputs transcript() can read (final, draft, notes, speakers). Read-only."""
        return await run(lambda: scribe.recent(limit))

    @server.tool(annotations=reads, structured_output=False)
    async def transcript(id: str, kind: Literal["final", "draft", "notes", "speakers"] = "final", offset: int = 0,
                         max_chars: int = MAX_CHARS) -> str:
        """The text of one output of a recording. kind: final (the corrected transcript Claude shelved in the
        vault), draft (Scribe's own speaker-labelled draft; a Local job's result), notes (summary, decisions and
        to-do beside the final transcript), speakers (the speaker-labelled lines). A long text comes in windows of
        max_chars; its last line gives the offset of the next window. Read-only."""
        return await run(lambda: scribe.transcript(id, kind, offset, max_chars))

    @server.tool(annotations=reads, structured_output=False)
    async def live_transcript(id: str = "current", speakers: bool = True, last_minutes: float | None = None) -> str:
        """Transcribes what a recording holds so far, including the one being recorded now (id "current"), with
        Whisper large-v3-turbo on this PC's GPU and local speaker labels. The recording is not disturbed: its
        tracks are copied as they stand, and the copy is transcribed. A GPU out of memory adds up to 30 s of
        waiting, then the CPU finishes, more slowly. A long recording takes a minute or more; last_minutes
        transcribes only its end. For a finished job, transcript(id, "speakers") is quicker. Read-only."""
        return await run(lambda: scribe.live(id, speakers, last_minutes))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                                             openWorldHint=False), structured_output=False)
    async def start_notetaker() -> str:
        """Acts: starts Scribe's notetaker, which records the microphone and the computer's sound, as a click on
        the tray does. Refuses when it is already recording."""
        return await run(scribe.start)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                                             openWorldHint=False), structured_output=False)
    async def stop_notetaker() -> str:
        """Acts: stops the notetaker, as its stop button does. Scribe then shows its vault picker on screen; with
        no pick the recording stays Local. Refuses when it is not recording."""
        return await run(scribe.stop)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                                             openWorldHint=True), structured_output=False)
    async def send_to_vault(id: str, vault: str) -> str:
        """Acts: sends a finished Local recording to a vault named in Scribe's settings, as
        the tray's Send does. Scribe queues its enhance step: a Claude run in that vault corrects the draft and
        shelves the transcript, recording and notes there. Refuses a job that is not finished or already sent."""
        return await run(lambda: scribe.send(id, vault))

    return server


def root_errors_only(record: logging.LogRecord) -> bool:
    """mcp.log's filter: records logged on the root logger itself pass from ERROR up. SDK 1.30.0 (the newest 1.x)
    logs a request it does not know there with logging.warning, as ~96 lines of pydantic errors; Claude Code sends
    one ("server/discover") on every connection. Named loggers, the SDK's own and Scribe's, are not touched."""
    return record.name != "root" or record.levelno >= logging.ERROR


def setup_logging(level: int = logging.INFO) -> Path:
    """%LOCALAPPDATA%\\Scribe\\logs\\mcp.log for this process (scribe.log belongs to the app: two processes cannot
    share one rotating file on Windows). The root logger gets it too, at WARNING, so nothing reaches stderr."""
    path = config.logs_dir() / "mcp.log"
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8",
                                                   delay=True)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"))
    handler.addFilter(root_errors_only)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("scribe").setLevel(level)
    return path


def claim_stdio():
    """The protocol's stdin and stdout, as the SDK's stdio_server wraps them. stdout is the protocol channel, so
    fd 1 and sys.stdout are pointed at devnull and the protocol writes to a private copy of the handle: a stray
    print, in Python or native code, can never corrupt it. The copy is not inherited by child processes."""
    import anyio

    if sys.stdin is None or sys.stdout is None:
        raise OSError("no stdin/stdout: start this server from an MCP client")
    sys.stdout.flush()
    out_fd = os.dup(sys.stdout.fileno())
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, sys.stdout.fileno())
    os.close(null)
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    stdin = anyio.wrap_file(io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace"))
    stdout = anyio.wrap_file(io.TextIOWrapper(os.fdopen(out_fd, "wb"), encoding="utf-8"))
    return stdin, stdout


async def _run_stdio(server, stdin, stdout) -> None:
    from mcp.server.stdio import stdio_server

    low = server._mcp_server       # FastMCP.run_stdio_async does the same, with the default streams
    async with stdio_server(stdin, stdout) as (read_stream, write_stream):
        await low.run(read_stream, write_stream, low.create_initialization_options())


def serve(port: int = app.IPC_PORT) -> int:
    import anyio

    os.environ["HF_HUB_OFFLINE"] = "1"    # models come from the local cache only
    stdin, stdout = claim_stdio()
    setup_logging()
    jobs.kill_children_with_me()          # a transcription or labelling child ends with this server
    log.info("MCP server starting (pid %d, Scribe IPC port %d)", os.getpid(), port)
    try:
        anyio.run(_run_stdio, build_server(Scribe(link=Link(port=port))), stdin, stdout)
    except Exception:
        log.exception("MCP server failed")
        return 1
    log.info("MCP server stopped")
    return 0


# ---------------------------------------------------------------- the child half of transcribe_child

def transcribe_main(request_path: Path) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    logging.basicConfig(level=logging.INFO, stream=sys.stderr or open(os.devnull, "w"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    req = jobs.read_json(request_path)
    beats = Path(req["progress"]) if req.get("progress") else None

    def beat(*_) -> None:
        """A line per chunk done and at each GPU wait: the parent's sign that this child is not hung."""
        if beats is not None:
            with open(beats, "a", encoding="utf-8") as fh:
                fh.write(".\n")

    whisper = jobs._mod("asr").transcribe_file(Path(req["audio"]), req["model"], list(req["languages"]),
                                               list(req["vocabulary"]), progress=beat, on_wait=beat)
    jobs.write_json_atomic(Path(req["out"]), whisper)
    return 0


# ---------------------------------------------------------------- registration for Claude Code

def server_entry() -> dict:
    """The user-scope entry for this server in ~/.claude.json: the venv's windowless Python (pythonw.exe, so no
    console window can open whatever starts it) running this module, with the package on PYTHONPATH (Claude
    Code starts a server in the session's folder)."""
    exe = Path(sys.executable).with_name("pythonw.exe")
    return {"type": "stdio", "command": str(exe if exe.is_file() else Path(sys.executable)),
            "args": ["-m", MODULE], "env": {"PYTHONPATH": str(PACKAGE_ROOT)}}


def _same_entry(have, want: dict) -> bool:
    if not isinstance(have, dict):
        return False
    return (have.get("type", "stdio") == want["type"]
            and os.path.normcase(str(have.get("command", ""))) == os.path.normcase(want["command"])
            and list(have.get("args") or []) == want["args"]
            and dict(have.get("env") or {}) == want["env"])


def _registered(config_path: Path):
    doc = _read_json_quietly(config_path)
    servers = doc.get("mcpServers") if isinstance(doc, dict) else None
    return servers.get(SERVER_NAME) if isinstance(servers, dict) else None


def register(claude_exe: Path, config_path: Path | None = None, run: Callable = subprocess.run) -> str:
    """Register this server for Claude Code at user scope, once: "unchanged" when ~/.claude.json already holds
    this entry, else "added" or "updated" through `claude mcp add --scope user` (an older entry is removed first).
    Raises RuntimeError when the CLI fails or the entry is not there afterwards."""
    config_path = Path(config_path) if config_path else Path.home() / ".claude.json"
    want = server_entry()
    have = _registered(config_path)
    if _same_entry(have, want):
        return "unchanged"
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)                         # not a nested session
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

    def claude(*args: str) -> None:
        proc = run([str(claude_exe), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                   timeout=120, stdin=subprocess.DEVNULL, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        if proc.returncode != 0:
            raise RuntimeError(f"claude {args[0]} {args[1]} exited {proc.returncode}: "
                               f"{' '.join((proc.stderr or proc.stdout or '').split())[-300:]}")

    if have is not None:
        claude("mcp", "remove", "--scope", "user", SERVER_NAME)
    args = ["mcp", "add", "--scope", "user", SERVER_NAME]
    for key, value in want["env"].items():
        args += ["-e", f"{key}={value}"]
    claude(*args, "--", want["command"], *want["args"])
    if not _same_entry(_registered(config_path), want):
        raise RuntimeError(f"claude mcp add ran, but {config_path} does not hold the {SERVER_NAME} entry")
    return "updated" if have is not None else "added"


# ---------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scribe.mcp_server")
    parser.add_argument("--ipc-port", type=int, default=app.IPC_PORT, help=argparse.SUPPRESS)   # tests only
    sub = parser.add_subparsers(dest="cmd")
    tr = sub.add_parser("transcribe", help="the transcription child (started by live_transcript)")
    tr.add_argument("--request", type=Path, required=True)
    reg = sub.add_parser("register", help="register this server for Claude Code at user scope")
    reg.add_argument("--claude", type=Path, default=Path.home() / ".local" / "bin" / "claude.exe")
    args = parser.parse_args(argv)
    if args.cmd == "transcribe":
        return transcribe_main(args.request)
    if args.cmd == "register":
        try:
            result = register(args.claude)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            print(f"mcp      {SERVER_NAME} not registered: {exc}", flush=True)
            return 1
        print(f"mcp      {SERVER_NAME} {result} (Claude Code, user scope)", flush=True)
        return 0
    return serve(args.ipc_port)


if __name__ == "__main__":
    sys.exit(main())
