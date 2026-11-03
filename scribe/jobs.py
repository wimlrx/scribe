"""Job records and the one-at-a-time pipeline that turns a recording into a draft and hands it to Claude.

`job.json` in each job folder is the single source of truth. Every step is skipped when its output
exists, so a restart resumes where the last run stopped. Heavy modules (Whisper, sherpa, ffmpeg,
Claude) are imported inside their steps, so the queue loads even when one of them cannot.
"""
from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import gc
import importlib
import json
import logging
import msvcrt
import os
import queue
import re
import sys
import threading
import time
import uuid
from ctypes import wintypes
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from . import config

log = logging.getLogger("scribe.jobs")

JOB_FILE = "job.json"
RECORDING_FILE = "recording.json"
LOCAL = "Local"
STATUSES = ("queued", "transcribing", "labelling", "frames", "drafting", "enhancing", "done", "failed")
TERMINAL = frozenset({"done", "failed"})
KINDS = ("meeting", "file")
LOCK_FILE = "job.lock"
SHARING_ERRORS = (5, 32)                # ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION: antivirus, sync clients

Progress = Callable[["Job", "float | None"], None]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _transient(exc: OSError) -> bool:
    """A Windows sharing or access error: someone (antivirus, a sync client, the UI) holds the file briefly."""
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in SHARING_ERRORS


def _retrying(fn, tries: int = 10):
    for attempt in range(tries):
        try:
            return fn()
        except OSError as exc:
            if attempt == tries - 1 or not _transient(exc):
                raise
            time.sleep(0.05 * (attempt + 1))


def _replace(src: Path, dst: Path, tries: int = 10) -> None:
    """os.replace with retries: on Windows a reader (the UI, antivirus) can hold the target for a moment."""
    _retrying(lambda: os.replace(src, dst), tries)


def write_json_atomic(path: Path, data) -> None:
    """Write beside the target, fsync, then os.replace, so a crash never leaves half a JSON file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)

    def write() -> None:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())

    _retrying(write)
    _replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class Job:
    """One recording's trip through the pipeline. Persisted as job.json in its folder."""

    id: str
    kind: str                      # "meeting" | "file"
    source: Path | None            # the dropped file; None for a meeting
    folder: Path                   # Recordings\<YYYY-MM-DD HHMM[-n]>
    vault: str                     # a settings["vaults"] key, or "Local"
    status: str = "queued"
    error: str | None = None
    waiting: str | None = None     # "gpu" while transcription waits out a CUDA out-of-memory error
    created: str = field(default_factory=now_iso)
    updated: str = field(default_factory=now_iso)
    outputs: dict = field(default_factory=dict)

    @classmethod
    def new(cls, kind: str, folder: Path, vault: str, source: Path | None = None) -> "Job":
        """A fresh queued job; its id is the folder name, unique by construction."""
        if kind not in KINDS:
            raise ValueError(f"unknown job kind: {kind}")
        folder = Path(folder)
        return cls(id=folder.name, kind=kind, source=Path(source) if source else None, folder=folder, vault=vault)

    @property
    def path(self) -> Path:
        return self.folder / JOB_FILE

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["source"] = str(self.source) if self.source else None
        data["folder"] = str(self.folder)
        return data

    @classmethod
    def from_dict(cls, data: dict, folder: Path | None = None) -> "Job":
        names = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in data.items() if k in names}
        values["source"] = Path(values["source"]) if values.get("source") else None
        values["folder"] = Path(folder) if folder else Path(values["folder"])
        values["outputs"] = dict(values.get("outputs") or {})
        return cls(**values)

    def save(self) -> None:
        self.updated = now_iso()
        self.folder.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.path, self.to_dict())

    @classmethod
    def load(cls, folder: Path) -> "Job":
        """Read job.json; the folder on disk wins over the recorded one, so a moved Recordings tree still works."""
        folder = Path(folder)
        return cls.from_dict(read_json(folder / JOB_FILE), folder=folder)


# ---------------------------------------------------------------- helpers

def _mod(name: str):
    """Lazy import of a sibling module (scribe.asr, scribe.video, ...)."""
    return importlib.import_module(f"{__package__}.{name}")


def _setting(settings: dict, *keys: str):
    """settings[k1][k2]..., falling back to config.DEFAULTS for anything missing."""
    for source in (settings, config.DEFAULTS):
        value = source
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            continue
    raise KeyError(".".join(keys))


def _emit(progress: Progress | None, job: Job, fraction: float | None) -> None:
    """Report progress; a broken UI callback never fails a job."""
    if progress is None:
        return
    try:
        progress(job, fraction)
    except Exception:
        log.exception("progress callback failed")


def _store(job: Job, key: str, path: Path) -> None:
    """Record a finished output: a bare name when it sits in the job folder, else the full path."""
    path = Path(path)
    job.outputs[key] = path.name if path.parent == job.folder else str(path)


def _out(job: Job, key: str) -> Path | None:
    value = job.outputs.get(key)
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else job.folder / p


def _done(job: Job, key: str) -> bool:
    """A step's output counts only once recorded in job.json AND present on disk; deleting the file redoes the step."""
    p = _out(job, key)
    return p is not None and p.exists()


def _free_gpu() -> None:
    """Drop any Whisper model left alive before the vision model or Claude needs the memory."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _load_audio(path: Path):
    """16 kHz mono float32 for the voiceprint match (a 2 h file is ~460 MB; acceptable per DESIGN).
    Uses diarize.read_audio when that module offers it (it also resamples), else a plain soundfile read."""
    reader = getattr(_mod("diarize"), "read_audio", None)
    if reader is not None:
        return reader(path)
    import soundfile

    data, _rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    return data.mean(axis=1) if data.ndim > 1 else data


def _fresh(tmp: Path) -> Path:
    """A temp output path with any leftover from a crashed run removed (ffmpeg without -y would stall on it)."""
    tmp.unlink(missing_ok=True)
    return tmp


def _same_file(a: Path, b: Path | None) -> bool:
    """True when two paths name one file (NTFS ignores case: Recording.m4a is recording.m4a)."""
    if b is None:
        return False
    if os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b)):
        return True
    try:
        return a.exists() and Path(b).exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _beside_source(job: Job, path: Path) -> Path:
    """`path`, or a .scribe variant of it when it would be the job's own source: Scribe's outputs can never
    overwrite or delete the recording they are made from."""
    if not _same_file(path, job.source):
        return path
    other = path.with_name(f"{path.stem}.scribe{path.suffix}")
    log.warning("job %s: %s is the source itself; writing %s instead", job.id, path.name, other.name)
    return other


_STAMP = re.compile(r"(20\d{2})-?(\d{2})-?(\d{2})")


def _date_in_name(name: str) -> date | None:
    """The recording's own date stamp (Teams: 'Weekly sync-20261015_143000-Meeting Recording.mp4'), if any."""
    reader = getattr(_mod("draft"), "_date_from", None)
    text = reader(name) if reader is not None else None
    if text is None:
        m = _STAMP.search(name)
        text = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _title_of(source: Path) -> str:
    """The title the transcribe skill gives a file: the name without its stamp and 'Meeting Recording'."""
    reader = getattr(_mod("draft"), "_title_from", None)
    return reader(source.name) if reader is not None else source.stem


def _audio_path(job: Job) -> Path:
    p = _out(job, "audio")
    if p is None:
        raise RuntimeError("no audio prepared")
    return p


def _meeting_date(job: Job) -> str:
    """YYYY-MM-DD the meeting happened, fixed once in outputs so later steps and restarts agree."""
    if job.outputs.get("date"):
        return job.outputs["date"]
    day = None
    if job.kind == "meeting":
        try:
            day = datetime.strptime(job.folder.name[:10], "%Y-%m-%d").date()
        except ValueError:
            pass
    elif job.source:
        day = _date_in_name(job.source.name)       # the download time (mtime) is not the meeting's date
        if day is None and job.source.exists():
            day = datetime.fromtimestamp(job.source.stat().st_mtime).date()
    if day is None:
        try:
            day = datetime.fromisoformat(job.created).date()
        except ValueError:
            day = datetime.now().date()
    job.outputs["date"] = day.isoformat()
    return job.outputs["date"]


# ---------------------------------------------------------------- steps

def _step_audio(job: Job, settings: dict, progress: Progress | None, asr_lock) -> None:
    """meeting: mic + system -> mix.flac. file: audio straight from the source where it lies (no copy)."""
    if _done(job, "audio"):
        return
    if job.kind == "meeting":
        _store(job, "audio", _mod("audio").mix_tracks(job.folder))
        return
    if not job.source or not job.source.exists():
        raise FileNotFoundError(f"source missing: {job.source}")
    video = _mod("video")
    if "video" not in job.outputs:
        is_video = False
        if job.source.suffix.lower() in config.VIDEO_EXTS:
            info = video.probe(job.source)
            if not info.get("has_audio", True):
                raise RuntimeError("source has no audio track")
            is_video = bool(info.get("has_video"))
        job.outputs["video"] = is_video
    final = _beside_source(job, job.folder / "audio.flac")
    tmp = _fresh(_beside_source(job, job.folder / "audio.tmp.flac"))
    written = Path(video.extract_audio(job.source, tmp))
    _replace(written, final)
    _store(job, "audio", final)


def _step_transcribe(job: Job, settings: dict, progress: Progress | None, asr_lock) -> None:
    if _done(job, "whisper"):
        return
    asr = _mod("asr")

    def waiting(on: bool) -> None:
        job.waiting = "gpu" if on else None
        _emit(progress, job, None)

    try:
        with asr_lock if asr_lock is not None else contextlib.nullcontext():
            whisper = asr.transcribe_file(
                _audio_path(job),
                _setting(settings, "meetings", "model"),
                list(_setting(settings, "languages")),
                list(_setting(settings, "vocabulary")),
                progress=lambda f: _emit(progress, job, f),
                on_wait=waiting,
            )
    finally:
        job.waiting = None
    out = job.folder / "whisper.json"
    write_json_atomic(out, whisper)
    _store(job, "whisper", out)
    _free_gpu()


def _step_label(job: Job, settings: dict, progress: Progress | None, asr_lock) -> None:
    """Diarise and name speakers in a child process (diarize.label_job): sherpa-onnx holds the GIL for the
    whole run, which in Scribe's own process would starve the keyboard hook and a live recording."""
    if _done(job, "speakers"):
        return
    out = job.folder / "speakers.json"
    result = _mod("diarize").label_job(
        _audio_path(job), _out(job, "whisper"), out, config.models_dir(),
        threshold=float(_setting(settings, "meetings", "cluster_threshold")),
        owner_name=_setting(settings, "owner_name"),
        owner_match=float(_setting(settings, "meetings", "owner_match")),
        voiceprint_path=config.APP_DIR / "voiceprint.json",
        log_path=job.folder / "labelling.log",
    )
    write_json_atomic(out, result)
    _store(job, "speakers", out)


def _step_frames(job: Job, settings: dict, progress: Progress | None, asr_lock) -> None:
    if _done(job, "visuals"):
        return
    _free_gpu()
    video = _mod("video")
    frames = video.keyframes(job.source, job.folder / "frames")
    vision_error = getattr(video, "VideoError", None)
    job.outputs.pop("visuals_error", None)
    try:
        visuals = video.describe_frames(
            frames, _setting(settings, "ollama", "url"), _setting(settings, "ollama", "vision_model"),
            progress=lambda f: _emit(progress, job, f),
        )
    except Exception as exc:
        # Ollama down or the vision model missing: the transcript is the deliverable, so it goes ahead without
        # on-screen notes rather than failing the whole job. Deleting visuals.json redoes only this step.
        if vision_error is None or not isinstance(exc, vision_error):
            raise
        log.warning("job %s: no on-screen notes: %s", job.id, exc)
        job.outputs["visuals_error"] = str(exc)
        visuals = []
    out = job.folder / "visuals.json"
    write_json_atomic(out, visuals)
    _store(job, "visuals", out)


def _draft_meta(job: Job, settings: dict, speakers: dict) -> dict:
    if job.kind == "file" and job.source:
        title, source = _title_of(job.source), str(job.source)
    else:
        title, source = "Meeting", str(job.folder / "recording.m4a")
    status = "partial — Scribe stopped mid-recording; recovered at start-up" if job.outputs.get("interrupted") else "complete"
    owner = _setting(settings, "owner_name")
    return {
        "date": _meeting_date(job),
        "source": source,
        "title": title,
        "recording_status": status,
        "owner_name": owner,
        "owner_matched": bool(speakers.get("owner_matched")),
        "kind": job.kind,
    }


def _step_draft(job: Job, settings: dict, progress: Progress | None, asr_lock) -> None:
    if not _done(job, "draft"):
        whisper = read_json(_out(job, "whisper"))
        speakers = read_json(_out(job, "speakers")) if _done(job, "speakers") else {}
        lines = speakers.get("lines") or None
        visuals = read_json(_out(job, "visuals")) if job.outputs.get("video") and _done(job, "visuals") else None
        final = _beside_source(job, job.folder / "draft.md")
        tmp = _fresh(_beside_source(job, job.folder / "draft.tmp.md"))
        written = Path(_mod("draft").render_draft(whisper, lines, visuals, _draft_meta(job, settings, speakers), tmp))
        _replace(written, final)
        _store(job, "draft", final)
        job.save()
    if not _done(job, "recording"):
        final = _beside_source(job, job.folder / "recording.m4a")
        tmp = _fresh(_beside_source(job, job.folder / "recording.tmp.m4a"))
        written = Path(_mod("audio").to_m4a(_audio_path(job), tmp))
        _replace(written, final)
        _store(job, "recording", final)


class EnhanceInterrupted(RuntimeError):
    """A Claude run was cut off (Scribe quit, a reboot). It is never re-run into the vault on its own."""


def _process_alive(pid: int | None, created: float | None) -> bool:
    """True while the process that was started (same pid and start time) still runs."""
    if not pid:
        return False
    try:
        import psutil

        proc = psutil.Process(int(pid))
        if created is not None and abs(proc.create_time() - float(created)) > 1.0:
            return False
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _wait_process(pid: int, timeout: float) -> None:
    try:
        import psutil

        psutil.Process(int(pid)).wait(timeout=timeout)
    except Exception:
        pass


def _previous_run(job: Job, run: dict, log_path: Path, enhance, settings: dict) -> dict | None:
    """The result of a run started before this step was entered again: waited for while it still runs, then
    read from claude.log. None when it never finished."""
    if _process_alive(run.get("pid"), run.get("created")):
        log.warning("job %s: claude run (pid %s) still running; waiting for it", job.id, run.get("pid"))
        _wait_process(run["pid"], float(_setting(settings, "claude", "timeout_s")))
    reader = getattr(enhance, "last_result", None)
    return reader(log_path) if reader is not None else None


def _in_vault(vault_path: Path, rel: str) -> Path | None:
    """A vault-relative path Claude reported, or None when it points outside the vault."""
    path = (vault_path / str(rel).replace("\\", "/")).resolve()
    root = vault_path.resolve()
    return path if path == root or path.is_relative_to(root) else None


def _step_enhance(job: Job, settings: dict, progress: Progress | None, asr_lock) -> None:
    """The headless Claude run. Its session id and process are recorded before it starts, so a run cut off
    by a quit or a reboot is never started again behind the user's back: still running, it is waited for;
    finished unwatched, its result is read from claude.log; otherwise the job fails as interrupted and a
    retry resumes that same session. 'done' means the transcript and notes are on disk in the vault."""
    if _done(job, "enhance"):
        return
    vaults = _setting(settings, "vaults")
    if job.vault not in vaults:
        raise ValueError(f"unknown vault: {job.vault}")
    vault_path = Path(config.expand(str(vaults[job.vault])))
    _free_gpu()
    enhance = _mod("enhance")
    payload = {
        **job.to_dict(),
        "date": _meeting_date(job),
        "owner_name": _setting(settings, "owner_name"),
        "video": bool(job.outputs.get("video")),
        "recording": str(_out(job, "recording") or job.folder / "recording.m4a"),
    }
    log_path = job.folder / "claude.log"
    run = job.outputs.get("claude_run") or None
    resume = bool(job.outputs.pop("enhance_resume", False))
    result = _previous_run(job, run, log_path, enhance, settings) if run else None
    if result is None:
        if run and not resume:
            raise EnhanceInterrupted(f"interrupted during enhancement; retry resumes session {run.get('session')}")

        def started(info: dict) -> None:
            job.outputs["claude_run"] = dict(info)
            job.save()

        if run and run.get("session"):
            kwargs = {"session_id": run["session"], "resume": True}
        else:
            kwargs = {"session_id": str(uuid.uuid4())}
        result = enhance.run_claude(payload, vault_path, log_path, settings, on_start=started, **kwargs)
    job.outputs["log"] = "claude.log"
    if result.get("session_id"):
        job.outputs["session_id"] = result["session_id"]
    if result.get("returncode") != 0:
        raise RuntimeError(f"claude exited {result.get('returncode')}")
    if not result.get("transcript"):
        raise RuntimeError("claude gave no result line")
    missing = []
    for key in ("transcript", "notes"):
        rel = result.get(key)
        path = _in_vault(vault_path, rel) if rel else None
        if path is None or not path.is_file():
            missing.append(f"{key} {rel!r}")
    if missing:
        raise RuntimeError(f"claude reported files that are not in the vault: {', '.join(missing)}")
    out = job.folder / "enhance.json"
    write_json_atomic(out, result)
    _store(job, "enhance", out)
    job.outputs["transcript"] = result["transcript"]
    job.outputs["notes"] = result.get("notes")


# (status, step, applies) in pipeline order. The audio step runs under "queued", as DESIGN.md lists it.
_STEPS = (
    ("queued", _step_audio, lambda job: True),
    ("transcribing", _step_transcribe, lambda job: True),
    ("labelling", _step_label, lambda job: True),
    ("frames", _step_frames, lambda job: bool(job.outputs.get("video"))),
    ("drafting", _step_draft, lambda job: True),
    ("enhancing", _step_enhance, lambda job: job.vault != LOCAL),
)


class FolderLock:
    """An exclusive lock on one job folder (a byte-range lock on job.lock), held for a whole pipeline run.
    The app and `python -m scribe.run` both take it, so two processes never run one folder's steps at once.
    Windows drops the lock when the holder dies, so a crash never leaves a folder locked."""

    def __init__(self, folder: Path):
        self.path = Path(folder) / LOCK_FILE
        self._fd: int | None = None

    def acquire(self) -> bool:
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
        except OSError as exc:
            log.warning("job lock %s not opened: %r", self.path, exc)
            return False
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        os.close(fd)
        try:
            self.path.unlink()
        except OSError:
            pass


def folder_busy(folder: Path) -> bool:
    """True while another pipeline run (in this process or another) holds the folder."""
    lock = FolderLock(folder)
    if lock.acquire():
        lock.release()
        return False
    return True


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = [wintypes.HANDLE]
_CloseHandle.restype = wintypes.BOOL


def _save_quietly(job: Job) -> None:
    """Record a failure even when job.json is briefly held: retry once, then log."""
    for attempt in range(2):
        try:
            job.save()
            return
        except Exception:
            if attempt:
                log.exception("job %s: job.json not saved", job.id)
            else:
                time.sleep(0.5)


def run_pipeline(job: Job, settings: dict, progress: Progress | None = None, asr_lock: threading.Lock | None = None) -> Job:
    """Run every step not yet done, saving job.json at each status change; stop at the first failure.

    `progress(job, fraction)` gets fraction None on each status change and 0..1 inside long steps.
    A failed job passed in again is retried from where it failed. A folder another process is running is
    left alone (the job comes back unchanged). GPU steps wait for nothing: several processes may share the GPU,
    and transcription copes with the GPU running out of memory itself (asr.transcribe_file).
    """
    if job.status == "done":
        return job
    job.folder.mkdir(parents=True, exist_ok=True)
    lock = FolderLock(job.folder)
    if not lock.acquire():
        log.warning("job %s: folder in use by another Scribe run; left to it", job.id)
        return job
    try:
        return _run_steps(job, settings, progress, asr_lock)
    finally:
        lock.release()


def _run_steps(job: Job, settings: dict, progress: Progress | None, asr_lock) -> Job:
    job.error = None
    for status, step, applies in _STEPS:
        if not applies(job):
            continue
        try:
            job.status = status
            job.save()
            _emit(progress, job, None)
            step(job, settings, progress, asr_lock)
            job.save()
        except Exception as exc:  # a step or a save: either way the job ends failed, recorded and notified
            log.exception("job %s failed at %s", job.id, status)
            job.status = "failed"
            job.error = f"{status}: {type(exc).__name__}: {exc}"
            _save_quietly(job)
            _emit(progress, job, None)
            return job
    job.status = "done"
    try:
        job.save()
    except Exception as exc:
        log.exception("job %s: final save failed", job.id)
        job.status, job.error = "failed", f"done: {type(exc).__name__}: {exc}"
        _save_quietly(job)
    _emit(progress, job, None)
    log.info("job %s %s (%s)", job.id, job.status, job.vault)
    return job


class _BASIC_LIMITS(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _EXTENDED_LIMITS(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BASIC_LIMITS), ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x0800
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_EXTENDED_LIMIT_INFORMATION = 9
_CreateJobObjectW = _kernel32.CreateJobObjectW
_CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
_CreateJobObjectW.restype = wintypes.HANDLE
_SetInformationJobObject = _kernel32.SetInformationJobObject
_SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
_SetInformationJobObject.restype = wintypes.BOOL
_AssignProcessToJobObject = _kernel32.AssignProcessToJobObject
_AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_AssignProcessToJobObject.restype = wintypes.BOOL
_GetCurrentProcess = _kernel32.GetCurrentProcess
_GetCurrentProcess.argtypes = []
_GetCurrentProcess.restype = wintypes.HANDLE
_JOB_HANDLE = None


def kill_children_with_me() -> bool:
    """Put this process in a Job Object that kills every process in it when Scribe ends (its handle, held
    only here, closes with the process): ffmpeg, the labelling child and the Claude run never outlive the
    app, so a restart can never run a second copy beside an orphan. True when in place."""
    global _JOB_HANDLE
    if _JOB_HANDLE:
        return True
    handle = _CreateJobObjectW(None, None)
    if not handle:
        log.warning("job object not created (error %d)", ctypes.get_last_error())
        return False
    info = _EXTENDED_LIMITS()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK
    if not _SetInformationJobObject(handle, _JOB_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                    ctypes.sizeof(info)) \
            or not _AssignProcessToJobObject(handle, _GetCurrentProcess()):
        log.warning("job object not applied (error %d); children may outlive Scribe", ctypes.get_last_error())
        _CloseHandle(handle)
        return False
    _JOB_HANDLE = handle
    return True


# ---------------------------------------------------------------- recovery and queue

def recover_folder(folder: Path, settings: dict) -> Job | None:
    """The job a Recordings folder holds at start-up, or None.

    A recording.json still saying "recording" (or torn by a power cut, with no job.json) is an interrupted
    recording: it is marked stopped and, with no job.json yet, becomes a Local meeting job. A stopped
    recording with no job.json (Scribe closed while the vault picker was up) is queued the same way.
    Nothing goes to a vault without the user choosing it.
    """
    folder = Path(folder)
    rec_path = folder / RECORDING_FILE
    has_job = (folder / JOB_FILE).exists()
    rec = None
    torn = False
    if rec_path.exists():
        try:
            rec = read_json(rec_path)
        except (OSError, ValueError):
            log.warning("unreadable %s", rec_path)
            rec, torn = {}, True
        if not isinstance(rec, dict):
            rec, torn = {}, True
    # A recording.json a power cut left empty, beside no job.json, is a recording that never stopped.
    interrupted = rec is not None and (rec.get("status") == "recording" or (torn and not has_job))
    if interrupted:
        rec["status"] = "stopped"
        rec["recovered"] = now_iso()
        rec.setdefault("events", []).append({"at": now_iso(), "event": "interrupted; recovered at start-up"})
        write_json_atomic(rec_path, rec)
        log.warning("interrupted recording recovered: %s", folder.name)
    if has_job:
        job = Job.load(folder)
        if interrupted:
            job.outputs["interrupted"] = True
        return job
    if rec is None:
        return None
    # Nobody chose a vault for it: it stays Local (the draft is the deliverable) until the user sends it on.
    job = Job.new("meeting", folder, LOCAL)
    job.outputs["interrupted"] = interrupted
    return job


class JobQueue:
    """One worker thread, one job at a time (one GPU). notify(job) is called on every status change."""

    def __init__(self, settings: dict, notify: Callable[[Job], None] | None = None,
                 asr_lock: threading.Lock | None = None):
        self.settings = settings
        self.notify = notify
        self.asr_lock = asr_lock
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the worker and queue every non-terminal job found in Recordings\\*. Call before any recording starts."""
        if self._thread and self._thread.is_alive():
            return
        root = config.recordings_dir(self.settings)
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            if folder_busy(folder):
                log.info("job folder %s in use by another Scribe run; skipped", folder.name)
                continue
            try:
                job = recover_folder(folder, self.settings)
            except Exception:
                log.exception("cannot read job folder %s", folder)
                continue
            if job is not None and not job.terminal:
                log.info("resuming job %s at %s", job.id, job.status)
                self.submit(job)
        self._thread = threading.Thread(target=self._run, name="scribe-jobs", daemon=True)
        self._thread.start()

    def submit(self, job: Job) -> None:
        """Queue a job (a failed one is retried). A job already waiting or running is not queued twice."""
        if job.status == "done":
            log.info("job %s already done", job.id)
            return
        key = str(job.folder)
        with self._lock:
            if key in self._pending:
                return
            if job.status == "failed":
                job.status, job.error = "queued", None
            self._pending.add(key)
            self._jobs[key] = job
        job.save()
        self._notify(job)
        self._queue.put(job)

    def retry(self, folder: Path) -> Job:
        """Re-queue a failed job from its folder. A Claude run it started is resumed, not started again."""
        job = Job.load(folder)
        if job.status == "failed" and job.outputs.get("claude_run"):
            job.outputs["enhance_resume"] = True
        self.submit(job)
        return job

    def send_to_vault(self, folder: Path, vault: str) -> Job:
        """Send a finished job (usually Local) on to a vault: only the enhancing step runs."""
        if vault not in _setting(self.settings, "vaults"):
            raise ValueError(f"unknown vault: {vault}")
        job = Job.load(folder)
        if job.vault == vault and job.status == "done":
            return job
        job.vault = vault
        for key in ("enhance", "transcript", "notes", "claude_run", "enhance_resume", "session_id", "log"):
            job.outputs.pop(key, None)
        if job.status == "done":
            job.status = "queued"
        self.submit(job)
        return job

    def failed(self) -> list[Job]:
        """Failed jobs seen this session, for the tray's Retry."""
        with self._lock:
            return [j for j in self._jobs.values() if j.status == "failed" and str(j.folder) not in self._pending]

    def jobs(self) -> list[Job]:
        """Snapshot of the jobs seen this session, oldest first."""
        with self._lock:
            return [dataclasses.replace(j, outputs=dict(j.outputs)) for j in self._jobs.values()]

    def active(self) -> int:
        """Jobs waiting or running, for the tray count."""
        with self._lock:
            return len(self._pending)

    def stop(self, timeout: float | None = None) -> None:
        """Let the running job finish, then end the worker."""
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout)

    def _notify(self, job: Job) -> None:
        if self.notify is None:
            return
        try:
            self.notify(job)
        except Exception:
            log.exception("notify failed")

    def _progress(self, job: Job, fraction: float | None) -> None:
        if fraction is None:
            self._notify(job)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                run_pipeline(job, self.settings, progress=self._progress, asr_lock=self.asr_lock)
            except Exception:
                log.exception("job %s crashed", job.id)
            finally:
                with self._lock:
                    self._pending.discard(str(job.folder))
                self._notify(job)
