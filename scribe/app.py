"""Scribe's process: one instance, IPC for second launches, and the wiring between the hotkey, the mic,
Whisper, paste, the notetaker, the inbox and the job queue.

Tk owns the main thread (ui.py). Everything slow (Whisper, the recorder opening devices, pasting) runs on
worker threads, so the overlay never freezes and the keyboard hook's callbacks return at once.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import json
import logging
import logging.handlers
import os
import queue
import secrets
import sys
import threading
import time
from ctypes import wintypes
from multiprocessing import connection as mpc
from pathlib import Path
from typing import Callable

from . import config, jobs, ui

log = logging.getLogger("scribe.app")

MUTEX_NAME = "Local\\ScribeSingleton"
IPC_HOST = "127.0.0.1"
IPC_PORT = 47391
KEY_FILE = "ipc.key"
MAX_MESSAGE = 1 << 20
CLIENT_WAIT_S = 20.0          # a second launch keeps trying this long while the first is still starting
SR = 16000
INBOX_STABLE_S = 5.0          # a dropped file must keep its size and time this long before it is taken
INBOX_POLL_S = 1.0
ENROLL_MIN_S = 3.0            # dictations at least this long may feed the voiceprint
ENROLL_FIRST = 30             # every eligible clip until the voiceprint holds this many ...
ENROLL_EVERY = 10             # ... then one in this many
MIN_MEETING_S = 5.0           # shorter notetaker runs are kept on disk but not queued
TOGGLE_DEBOUNCE_S = 0.8       # a double click on the tray or the stop button is one toggle
HOOK_REFRESH_S = 600.0        # Windows can drop a slow low-level hook without notice; re-install it when idle
MAINT_TICK_S = 0.1            # the maintenance loop wakes this often ...
STALL_S = 0.25                # ... and a wake-up this late means Python (the hook with it) was starved
HOOK_RETRY_S = 2.0            # a hook that will not re-install is tried again this often
HOLD_WAIT_S = 600.0           # a finished clip waits at most this long for a newer hold to end
INBOX_RESCAN_S = 30.0         # the inbox folders are listed again this often (watchdog events can be lost)
REPLY_WAIT_S = 30.0           # a second launch waits this long for the running Scribe's answer

# ---------------------------------------------------------------- command line


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scribe.pyw")
    p.add_argument("--toggle-notetaker", action="store_true", help="start or stop the notetaker")
    p.add_argument("--add", nargs="+", action="extend", default=[], metavar="PATH",
                   help="transcribe these files where they lie")
    p.add_argument("--vault", metavar="NAME", help="vault for --add or --enhance; without it a picker asks")
    p.add_argument("--retry", nargs="+", action="extend", default=[], metavar="FOLDER",
                   help="retry these failed job folders")
    p.add_argument("--enhance", nargs="+", action="extend", default=[], metavar="FOLDER",
                   help="send these finished job folders to --vault")
    p.add_argument("--quit", action="store_true", help="close the running Scribe")
    p.add_argument("--status", action="store_true", help="print the running Scribe's state as JSON")
    return p


def parse_args(argv: list[str]) -> argparse.Namespace | None:
    """Parsed arguments, or None when they do not parse (logged; never a dialog)."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return build_parser().parse_args(argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            log.error("bad arguments %r: %s", argv, err.getvalue().strip().splitlines()[-1:] or "")
        return None


def commands_from_args(ns: argparse.Namespace, cwd: Path | None = None) -> list[dict]:
    """The commands an invocation asks for, with paths made absolute (the running instance has another cwd)."""
    if ns.quit:
        return [{"cmd": "quit"}]
    base = Path(cwd) if cwd else Path.cwd()
    cmds: list[dict] = []
    if ns.status:
        cmds.append({"cmd": "status"})
    if ns.toggle_notetaker:
        cmds.append({"cmd": "toggle"})
    if ns.add:
        paths = [os.path.abspath(p if os.path.isabs(p) else base / p) for p in ns.add]
        cmds.append({"cmd": "add", "paths": paths, "vault": ns.vault})
    if ns.retry:
        cmds.append({"cmd": "retry", "folders": [os.path.abspath(p if os.path.isabs(p) else base / p)
                                                 for p in ns.retry]})
    if ns.enhance:
        cmds.append({"cmd": "enhance", "folders": [os.path.abspath(p if os.path.isabs(p) else base / p)
                                                   for p in ns.enhance], "vault": ns.vault})
    if not cmds:
        cmds.append({"cmd": "show"})
    return cmds


# ---------------------------------------------------------------- single instance

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_CreateMutexW = _kernel32.CreateMutexW
_CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
_CreateMutexW.restype = wintypes.HANDLE
_CloseHandle = _kernel32.CloseHandle
_CloseHandle.argtypes = [wintypes.HANDLE]
_CloseHandle.restype = wintypes.BOOL
ERROR_ALREADY_EXISTS = 183


class SingleInstance:
    """Named mutex: the first process to create it is Scribe; later launches hand their command over IPC."""

    def __init__(self, name: str = MUTEX_NAME):
        self.name = name
        self._handle = None

    def acquire(self) -> bool:
        """True if this process now owns the name; False if another Scribe already holds it."""
        if self._handle:
            return True
        ctypes.set_last_error(0)
        handle = _CreateMutexW(None, False, self.name)
        err = ctypes.get_last_error()
        if not handle:
            raise ctypes.WinError(err)
        if err == ERROR_ALREADY_EXISTS:
            _CloseHandle(handle)
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle:
            _CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> "SingleInstance":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ---------------------------------------------------------------- IPC (JSON over an authenticated local socket)


def load_or_create_key(path: Path) -> bytes:
    """The IPC secret, created on first run (64 hex chars). Only the first instance calls this."""
    path = Path(path)
    try:
        data = path.read_bytes().strip()
        if len(data) >= 32:
            return data
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32).encode("ascii")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(key)
    os.replace(tmp, path)
    return key


def read_key(path: Path) -> bytes:
    """The IPC secret for a second launch. Raises OSError while the first instance has not written it yet."""
    data = Path(path).read_bytes().strip()
    if len(data) < 32:
        raise OSError("ipc key incomplete")
    return data


class IpcServer:
    """Accepts {"commands": [...]} from later launches and replies with the handler's dict.

    The HMAC challenge runs per connection on its own thread, so a client that stalls mid-handshake
    cannot block the next one. Only JSON is exchanged (no pickles)."""

    def __init__(self, handler: Callable[[list[dict]], dict], authkey: bytes,
                 port: int = IPC_PORT, host: str = IPC_HOST):
        self.handler, self.authkey, self.address = handler, authkey, (host, port)
        self._listener: mpc.Listener | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        """Listen. Raises OSError when the port is taken."""
        self._listener = mpc.Listener(self.address, family="AF_INET")
        self._thread = threading.Thread(target=self._serve, name="scribe-ipc", daemon=True)
        self._thread.start()
        log.info("IPC on %s:%d", *self.address)

    def _serve(self) -> None:
        while not self._closed:
            try:
                conn = self._listener.accept()
            except OSError:
                if self._closed:
                    return
                log.exception("IPC accept failed")
                time.sleep(0.2)
                continue
            threading.Thread(target=self._handle, args=(conn,), name="scribe-ipc-conn", daemon=True).start()

    def _handle(self, conn) -> None:
        with conn:
            try:
                mpc.deliver_challenge(conn, self.authkey)
                mpc.answer_challenge(conn, self.authkey)
            except Exception as exc:  # wrong key, a stray client, or garbage on the port
                log.warning("IPC client refused: %s", type(exc).__name__)
                return
            try:
                if not conn.poll(10.0):
                    return
                message = json.loads(conn.recv_bytes(MAX_MESSAGE).decode("utf-8"))
                commands = message.get("commands") if isinstance(message, dict) else None
                if not isinstance(commands, list):
                    raise ValueError("no command list")
                reply = self.handler([c for c in commands if isinstance(c, dict)])
            except Exception as exc:
                log.exception("IPC command failed")
                reply = {"ok": False, "error": type(exc).__name__}
            try:
                conn.send_bytes(json.dumps(reply, default=str).encode("utf-8"))
            except OSError:
                pass

    def close(self) -> None:
        self._closed = True
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass


class NoReply(Exception):
    """The commands reached the running Scribe, but no answer came back. Never resent: they may have run."""


def send_commands(commands: list[dict], authkey: bytes, port: int = IPC_PORT, host: str = IPC_HOST,
                  timeout: float = REPLY_WAIT_S) -> dict:
    """Hand commands to the running Scribe; its reply. Raises OSError (not running) or AuthenticationError
    before anything was delivered, and NoReply once the commands went out but no answer came."""
    conn = mpc.Client((host, port), family="AF_INET", authkey=authkey)
    with conn:
        conn.send_bytes(json.dumps({"commands": commands}).encode("utf-8"))
        try:
            if not conn.poll(timeout):
                raise TimeoutError("no reply from Scribe")
            return json.loads(conn.recv_bytes(MAX_MESSAGE).decode("utf-8"))
        except (OSError, EOFError, ValueError) as exc:
            raise NoReply(repr(exc)) from exc


def deliver(commands: list[dict], key_path: Path, port: int = IPC_PORT, wait_s: float = CLIENT_WAIT_S,
            timeout: float = REPLY_WAIT_S) -> dict | None:
    """send_commands, retried only while nothing has been delivered (the first instance still starting up,
    the key file not yet written). None if it never answers."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return send_commands(commands, read_key(key_path), port, timeout=timeout)
        except NoReply as exc:
            log.error("running Scribe took the commands but did not answer: %s", exc)
            return None
        except (OSError, EOFError, mpc.AuthenticationError, ValueError) as exc:
            if time.monotonic() >= deadline:
                log.error("running Scribe did not answer: %r", exc)
                return None
            time.sleep(0.25)


# ---------------------------------------------------------------- pure rules (tested)


def _norm(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def is_media(path: Path) -> bool:
    """A file Scribe transcribes: a known audio/video extension, not a temp or hidden file."""
    path = Path(path)
    return path.suffix.lower() in config.MEDIA_EXTS and not path.name.startswith(("~$", "."))


def should_enroll(enrolled: int, eligible_index: int, first: int = ENROLL_FIRST, every: int = ENROLL_EVERY) -> bool:
    """Voiceprint policy: every eligible dictation until `first` clips are in, then one in `every`."""
    return enrolled < first or eligible_index % every == 0


def job_label(job: jobs.Job) -> str:
    """A job's short name: the file's stem, or "Meeting 14:05" from the folder stamp."""
    if job.kind == "file" and job.source:
        return Path(job.source).stem
    stamp = job.folder.name
    return f"Meeting {stamp[11:13]}:{stamp[13:15]}" if len(stamp) >= 15 and stamp[10] == " " else "Meeting"


def job_notice(job: jobs.Job) -> tuple[str, str]:
    """(title, message) for a finished job's notification: a symbol, a name and the vault. No sentences."""
    mark = "✓" if job.status == "done" else "✕"
    return "Scribe", f"{mark} {job_label(job)} · {job.vault}"


class StableFiles:
    """Which candidate files have kept the same (size, mtime) for `stable_s`. Pure: the caller supplies time."""

    def __init__(self, stable_s: float = INBOX_STABLE_S):
        self.stable_s = stable_s
        self._seen: dict[Path, tuple[tuple | None, float]] = {}

    def touch(self, path: Path, now: float) -> None:
        """Start (or restart) watching: something just changed the file."""
        self._seen[Path(path)] = (None, now)

    def check(self, path: Path, signature: tuple | None, now: float) -> bool:
        """True once `signature` (size, mtime) has held for stable_s and the file is not empty."""
        path = Path(path)
        if signature is None:
            self._seen.pop(path, None)
            return False
        prev = self._seen.get(path)
        if prev is None or prev[0] != signature:
            self._seen[path] = (signature, now)
            return False
        return signature[0] > 0 and now - prev[1] >= self.stable_s

    def restart(self, path: Path, now: float) -> None:
        """Wait a full stable period again, from now, keeping the last signature (a refused move)."""
        path = Path(path)
        sig = self._seen.get(path, (None, now))[0]
        self._seen[path] = (sig, now)

    def forget(self, path: Path) -> None:
        self._seen.pop(Path(path), None)

    def paths(self) -> list[Path]:
        return list(self._seen)


# ---------------------------------------------------------------- inbox watcher


class InboxWatcher:
    """Watches Inbox\\<Vault>\\ folders (watchdog). A media file whose size and time hold still for
    `stable_s` is handed to on_file(path, vault); if that returns False (still locked) it is retried."""

    def __init__(self, folders: dict[str, Path], on_file: Callable[[Path, str], bool],
                 stable_s: float = INBOX_STABLE_S, poll_s: float = INBOX_POLL_S,
                 clock: Callable[[], float] = time.monotonic):
        self.folders = {v: Path(f) for v, f in folders.items()}
        self._keys = {v: _norm(f) for v, f in self.folders.items()}
        self.on_file = on_file
        self.poll_s = poll_s
        self.clock = clock
        self._tracker = StableFiles(stable_s)
        self._vaults: dict[Path, str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._observer = None
        self._watches: dict[str, object] = {}
        self._inodes: dict[str, int | None] = {}
        self._thread: threading.Thread | None = None

    def add(self, path: Path, vault: str) -> None:
        """A file appeared or changed. Only files lying directly in that vault's inbox count: a file moved
        out of the inbox (into its job folder) also raises an event, and must never be taken twice."""
        path = Path(path)
        if not is_media(path) or self._keys.get(vault) != _norm(path.parent):
            return
        with self._lock:
            self._vaults[path] = vault
            self._tracker.touch(path, self.clock())

    def scan(self) -> None:
        """Pick up files already lying in the inboxes (dropped while Scribe was closed)."""
        for vault, folder in self.folders.items():
            try:
                entries = list(folder.iterdir())
            except OSError:
                continue
            for p in entries:
                if p.is_file():
                    self.add(p, vault)

    def check_once(self) -> int:
        """One pass over the candidates; returns how many were handed on."""
        now = self.clock()
        with self._lock:
            items = list(self._vaults.items())
        taken = 0
        for path, vault in items:
            try:
                st = path.stat()
                sig = (st.st_size, st.st_mtime_ns)
            except FileNotFoundError:
                sig = None
            except OSError:
                continue
            with self._lock:
                ready = self._tracker.check(path, sig, now)
                if sig is None:
                    self._vaults.pop(path, None)
            if not ready:
                continue
            try:
                ok = bool(self.on_file(path, vault))
            except Exception:
                log.exception("inbox file %s failed", path.name)
                ok = False
            with self._lock:
                if ok:
                    self._vaults.pop(path, None)
                    self._tracker.forget(path)
                    taken += 1
                else:
                    self._tracker.restart(path, now)
        return taken

    def _handler(self, vault: str):
        from watchdog.events import FileSystemEventHandler

        watcher = self

        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    watcher.add(Path(os.fsdecode(event.src_path)), vault)

            def on_modified(self, event):
                self.on_created(event)

            def on_moved(self, event):
                if not event.is_directory:
                    watcher.add(Path(os.fsdecode(event.dest_path)), vault)

        return Handler()

    @staticmethod
    def _inode(folder: Path) -> int | None:
        try:
            return folder.stat().st_ino
        except OSError:
            return None

    def _watch(self, vault: str, folder: Path) -> None:
        """(Re)schedule the watchdog watch on one inbox folder."""
        if self._observer is None:
            return
        old = self._watches.pop(vault, None)
        if old is not None:
            try:
                self._observer.unschedule(old)
            except Exception:
                pass
        try:
            self._watches[vault] = self._observer.schedule(self._handler(vault), str(folder), recursive=False)
        except Exception:
            log.exception("inbox %s not watched", vault)

    def start(self) -> None:
        from watchdog.observers import Observer

        observer = Observer()
        observer.daemon = True
        self._observer = observer
        for vault, folder in self.folders.items():
            folder.mkdir(parents=True, exist_ok=True)
            self._inodes[vault] = self._inode(folder)
            self._watch(vault, folder)
        observer.start()
        self.scan()
        self._thread = threading.Thread(target=self._run, name="scribe-inbox", daemon=True)
        self._thread.start()
        log.info("watching %d inbox folders", len(self.folders))

    def refresh(self) -> None:
        """The periodic safety net: recreate a missing inbox folder, re-watch one that was deleted and
        recreated (watchdog stops at a deleted folder and never re-attaches), and list every folder again
        (a burst of copies can overflow the change buffer and lose events)."""
        for vault, folder in self.folders.items():
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            inode = self._inode(folder)
            if inode != self._inodes.get(vault):
                log.info("inbox %s was recreated; watching it again", vault)
                self._inodes[vault] = inode
                self._watch(vault, folder)
        self.scan()

    def _run(self) -> None:
        last = self.clock()
        while not self._stop.wait(self.poll_s):
            try:
                self.check_once()
                now = self.clock()
                if now - last >= INBOX_RESCAN_S:
                    last = now
                    self.refresh()
            except Exception:
                log.exception("inbox check failed")

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(2.0)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(2.0)


# ---------------------------------------------------------------- the app


def _remove_if_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass


class App:
    """Dictation, notetaker, drop-ins. Every entry point is safe from any thread.

    Collaborators are passed in so the flow can be tested with fakes: overlay (ui.Overlay), tray (ui.Tray),
    job_queue (jobs.JobQueue), mic (audio.MicCapture), asr (asr.DictationASR), voiceprint (diarize.Voiceprint
    or None), paste (paste.paste_text), recorder_factory (audio.MeetingRecorder), lexicon (learn.Lexicon or None),
    cleaner (cleanup.Cleaner or None), edit_watcher (learn.EditWatcher or None)."""

    def __init__(self, settings: dict, overlay, tray, job_queue, *, mic, asr, voiceprint=None,
                 paste: Callable[[str], bool], recorder_factory: Callable[[Path], object],
                 on_quit: Callable[[], None] | None = None, asr_lock: threading.Lock | None = None,
                 clock: Callable[[], float] = time.monotonic, lexicon=None, cleaner=None, edit_watcher=None):
        self.settings = settings
        self.overlay, self.tray, self.queue = overlay, tray, job_queue
        self.mic, self.asr, self.voiceprint = mic, asr, voiceprint
        self.paste, self.recorder_factory = paste, recorder_factory
        self.lexicon, self.cleaner, self.edit_watcher = lexicon, cleaner, edit_watcher
        self.on_quit = on_quit
        self.asr_lock = asr_lock
        self.clock = clock
        self.hotkey = None
        self.shortcut = None            # winkeys.Hotkey: the notetaker shortcut
        self.watcher: InboxWatcher | None = None
        self.ipc: IpcServer | None = None
        self.model_state = "loading"
        self._lock = threading.RLock()
        self._folder_lock = threading.Lock()
        self._holding = False
        self._hold_clear = threading.Event()   # set while no dictation hold is down
        self._hold_clear.set()
        self._rec_alerts: set[str] = set()
        self._pending = 0
        self._eligible = 0
        self._recorder = None
        self._rec_folder: Path | None = None
        self._toggle_busy = False
        self._last_toggle = float("-inf")
        self._quitting = False
        self._notified: dict[str, str] = {}
        self._dict_q: queue.Queue = queue.Queue()
        self._enroll_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.loading = threading.Event()   # set once the model load holds asr_lock (or has given up)

    # -- settings shortcuts

    def _dictation(self, key: str):
        return self.settings.get("dictation", {}).get(key, config.DEFAULTS["dictation"][key])

    def vaults(self) -> list[str]:
        return config.vault_names(self.settings)

    def default_vault(self) -> str:
        default = self.settings.get("default_vault", config.DEFAULTS["default_vault"])
        return default if default in self.vaults() else self.vaults()[0]

    # -- lifecycle

    def start(self, hotkey=None) -> None:
        """Worker threads, the dictation model load, and the hotkey. Returns at once."""
        for target, name in ((self._dictation_worker, "scribe-dictation"), (self._enroll_worker, "scribe-enroll"),
                             (self._load_model, "scribe-model"), (self._maintenance, "scribe-maintenance")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        self.hotkey = hotkey
        if hotkey is not None:
            try:
                hotkey.start()
            except OSError:
                log.exception("keyboard hook not installed; dictation is off")
                self.tray_notice("✕ Ctrl+Win")

    def _load_model(self) -> None:
        """Load the dictation model once, holding asr_lock so a resumed job does not load Whisper at the same time."""
        lock = self.asr_lock
        if lock is not None:
            lock.acquire()
        self.loading.set()
        try:
            self.asr.load()
            self.model_state = "ready"
        except Exception:
            self.model_state = "failed"
            log.exception("dictation model failed to load")
            self.tray_notice("✕ Model")
        finally:
            if lock is not None:
                lock.release()

    def _maintenance(self) -> None:
        """Every MAINT_TICK_S: act on the recorder's alerts, and keep the keyboard hook alive.

        Windows silently removes a low-level hook that misses LowLevelHooksTimeout, which any long GIL hold
        in this process can cause. A wake-up more than STALL_S late is that symptom, so the hook is
        re-installed at once (when no hold is down and the modifiers are up), not at the next 10-minute
        refresh. A hook that is not running is re-installed every HOOK_RETRY_S."""
        expected = time.monotonic() + MAINT_TICK_S
        last_refresh = next_try = time.monotonic()
        suspect = False
        while not self._stop.wait(MAINT_TICK_S):
            now = time.monotonic()
            late, expected = now - expected, now + MAINT_TICK_S
            if late > STALL_S:
                suspect = True
                log.info("maintenance woke %.0f ms late; keyboard hook to be re-installed", late * 1000)
            try:
                self._watch_recorder()
            except Exception:
                log.exception("recorder watch failed")
            hotkey = self.hotkey
            if hotkey is None or now < next_try:
                continue
            dead = not getattr(hotkey, "running", True)
            if not (suspect or dead or now - last_refresh >= HOOK_REFRESH_S):
                continue
            with self._lock:
                busy = self._holding or self._quitting
            if busy:
                continue
            try:
                from . import winkeys

                if winkeys.modifiers_down():
                    continue
                hotkey.stop()
                hotkey.start()
                suspect, last_refresh = False, now
                log.info("keyboard hook re-installed")
            except Exception:
                next_try = now + HOOK_RETRY_S
                log.exception("keyboard hook refresh failed")

    def _watch_recorder(self) -> None:
        """The recorder's alerts: a dead track (disk full) or a forgotten notetaker (20 min of silence, the
        6 h cap) stop the recording through the normal stop path; low disk is only shown."""
        with self._lock:
            recorder = self._recorder
            seen = set(self._rec_alerts)
        alerts = getattr(recorder, "alerts", None)
        if recorder is None or alerts is None:
            return
        new = set(alerts()) - seen
        if not new:
            return
        with self._lock:
            self._rec_alerts |= new
        if "disk_low" in new:
            self.tray_notice("⚠ Disk")
        reason = next((a for a in ("fault", "cap", "silence") if a in new), None)
        if reason is None:
            return
        log.warning("notetaker stopping by itself: %s", reason)
        self.tray_notice({"fault": "✕ Disk", "cap": "■ Cap", "silence": "■ Silence"}[reason])
        if reason == "fault":
            self.overlay.flash_error()
        self._auto_stop()

    def _auto_stop(self) -> None:
        """Stop the recording as the stop button would, without the double-click debounce."""
        with self._lock:
            if self._quitting or self._toggle_busy or self._recorder is None:
                return
            self._toggle_busy = True
            self._last_toggle = self.clock()
        threading.Thread(target=self._stop_notetaker, name="scribe-notetaker", daemon=True).start()

    def quit(self) -> None:
        """Ask the main loop to end; shutdown() then runs on the main thread."""
        with self._lock:
            if self._quitting:
                return
            self._quitting = True
        log.info("quit requested")
        if self.on_quit is not None:
            self.on_quit()

    def shutdown(self) -> None:
        """Stop everything that is not Tk. A running recording is closed cleanly; it is queued at next start."""
        with self._lock:
            self._quitting = True
            recorder, self._recorder = self._recorder, None
        self._hold_clear.set()
        self._stop.set()
        for name in ("hotkey", "shortcut", "edit_watcher"):
            part = getattr(self, name)
            if part is not None:
                try:
                    part.stop()
                except Exception:
                    log.exception("%s stop failed", name)
        if recorder is not None:
            try:
                recorder.stop()
                log.info("recording closed at quit; it is queued at next start")
            except Exception:
                log.exception("recorder stop at quit failed")
        for part in (self.watcher, self.ipc):
            if part is not None:
                try:
                    part.stop() if isinstance(part, InboxWatcher) else part.close()
                except Exception:
                    log.exception("shutdown step failed")
        self._dict_q.put(None)
        self._enroll_q.put(None)
        try:
            self.mic.close()
        except Exception:
            pass
        if self.tray is not None:
            self.tray.stop()

    # -- dictation (hotkey callbacks arrive on the hotkey worker thread)

    def is_holding(self) -> bool:
        """True while a Ctrl+Win hold is down: paste waits it out instead of timing out against it."""
        with self._lock:
            return self._holding

    def on_press(self) -> None:
        with self._lock:
            if self._quitting or self._holding:
                return
            self._holding = True
            self._hold_clear.clear()
        try:
            self.mic.start()
        except Exception:
            log.exception("microphone did not start")
            with self._lock:
                self._holding = False
                self._hold_clear.set()
            self.overlay.flash_error()
            self.refresh_tray()
            return
        if self.cleaner is not None:
            self.cleaner.warm()             # reloads the clean-up model while the user speaks if Ollama dropped it
        self.overlay.show_listening(self.mic.level)
        self.refresh_tray()

    def on_release(self) -> None:
        with self._lock:
            if not self._holding:
                return
            self._holding = False
            self._hold_clear.set()
        audio = self.mic.stop()
        seconds = len(audio) / SR
        if seconds < float(self._dictation("min_seconds")):
            log.info("dictation discarded: %.2f s", seconds)
            self._settle_overlay()
            self.refresh_tray()
            return
        with self._lock:
            self._pending += 1
        self.overlay.show_working()
        self._dict_q.put(audio)
        self.refresh_tray()

    def on_cancel(self) -> None:
        with self._lock:
            was, self._holding = self._holding, False
            self._hold_clear.set()
        if was:
            self.mic.stop()
            log.info("dictation cancelled by another key")
        self._settle_overlay()
        self.refresh_tray()

    def _settle_overlay(self) -> None:
        with self._lock:
            holding, pending = self._holding, self._pending
        if holding:
            return
        if pending:
            self.overlay.show_working()
        else:
            self.overlay.hide()

    def _dictation_worker(self) -> None:
        while True:
            audio = self._dict_q.get()
            if audio is None:
                return
            ok = self._dictate(audio)
            with self._lock:
                self._pending -= 1
                holding = self._holding
            if ok or holding:
                self._settle_overlay()
            else:
                self.overlay.flash_error()
            self.refresh_tray()

    def _dictate(self, audio) -> bool:
        """Transcribe and paste one clip. True when it worked or there was simply nothing said."""
        try:
            text = self.asr.transcribe(audio)
        except Exception:
            log.exception("dictation transcription failed")
            return False
        if not text:
            return True
        text = self._polish(text)
        # A newer hold that started while this clip was transcribing is waited out first: pasting into it
        # would time out against Scribe's own Ctrl+Win and drop this clip and the user's clipboard.
        if not self._hold_clear.wait(HOLD_WAIT_S):
            log.warning("a hold stayed down for %.0f s; pasting anyway", HOLD_WAIT_S)
        with self._lock:
            if self._quitting:
                return False
        try:
            pasted = bool(self.paste(text))
        except Exception:
            log.exception("paste failed")
            return False
        if pasted and self.edit_watcher is not None:
            try:
                self.edit_watcher.watch(text)       # the user's fixes to this paste teach the lexicon
            except Exception:
                log.exception("edit watch failed")
        # Never enrol while the notetaker runs: on a speakerphone call the clip can carry another voice.
        if pasted and self.voiceprint is not None and self._dictation("enroll_voice") \
                and len(audio) / SR >= ENROLL_MIN_S and not self.is_recording():
            self._enroll_q.put(audio)
        return pasted

    def _polish(self, text: str) -> str:
        """Learned respellings, then the local clean-up model. Each step falls back to the text it was given."""
        if self.lexicon is not None:
            try:
                text = self.lexicon.apply(text, protect=list(self.settings.get("vocabulary") or []))
            except Exception:
                log.exception("lexicon apply failed")
        if self.cleaner is not None:
            text = self.cleaner.clean(text) or text
        return text

    def _enroll_worker(self) -> None:
        while True:
            audio = self._enroll_q.get()
            if audio is None:
                return
            with self._lock:
                self._eligible += 1
                index = self._eligible
            try:
                if should_enroll(self.voiceprint.count, index):
                    self.voiceprint.enroll(audio)
            except Exception:
                log.exception("voiceprint enrolment failed")

    # -- notetaker

    def is_recording(self) -> bool:
        with self._lock:
            return self._recorder is not None

    def toggle_notetaker(self) -> None:
        """Start or stop the meeting recorder. The slow part runs on its own thread."""
        now = self.clock()
        with self._lock:
            if self._quitting or self._toggle_busy or now - self._last_toggle < TOGGLE_DEBOUNCE_S:
                return
            self._toggle_busy, self._last_toggle = True, now
            starting = self._recorder is None
        target = self._start_notetaker if starting else self._stop_notetaker
        threading.Thread(target=target, name="scribe-notetaker", daemon=True).start()

    def new_folder(self) -> Path:
        """A fresh Recordings\\<stamp> folder; serialised so two jobs in one minute get -2, -3."""
        with self._folder_lock:
            for _ in range(20):
                try:
                    return config.new_job_folder(settings=self.settings)
                except FileExistsError:
                    time.sleep(0.01)
            raise OSError("no free job folder name")

    def _start_notetaker(self) -> None:
        try:
            folder = self.new_folder()
            recorder = self.recorder_factory(folder)
            try:
                recorder.start()
            except Exception:
                log.exception("notetaker did not start")
                _remove_if_empty(folder)
                self.overlay.flash_error()
                return
            with self._lock:
                self._recorder, self._rec_folder = recorder, folder
                self._rec_alerts = set()
            self.overlay.show_recording(recorder.elapsed, self.toggle_notetaker)
            log.info("notetaker started: %s", folder.name)
        except Exception:
            log.exception("notetaker start failed")
            self.overlay.flash_error()
        finally:
            with self._lock:
                self._toggle_busy = False
            self.refresh_tray()

    def _stop_notetaker(self) -> None:
        try:
            with self._lock:
                recorder, folder = self._recorder, self._rec_folder
            if recorder is None:
                return
            try:
                result = recorder.stop() or {}
            except Exception:
                log.exception("notetaker stop failed")
                result = {}
            with self._lock:
                self._recorder, self._rec_folder = None, None
            self.overlay.hide_recording()
            seconds = float(result.get("seconds") or 0.0)
            log.info("notetaker stopped: %s, %.1f s", folder.name, seconds)
            if seconds < MIN_MEETING_S:
                self._shelve_short(folder, seconds)
                self.overlay.flash_error()
                return
            self.overlay.pick_vault(self.vaults(), self.default_vault(), int(self.settings.get("picker_timeout_s", 20)),
                                    lambda vault: self._submit_meeting(folder, vault))
        finally:
            with self._lock:
                self._toggle_busy = False
            self.refresh_tray()

    def _shelve_short(self, folder: Path, seconds: float) -> None:
        """Keep a too-short recording on disk but mark it failed, so start-up recovery never queues it."""
        job = jobs.Job.new("meeting", folder, jobs.LOCAL)
        job.status, job.error = "failed", f"too short: {seconds:.1f} s"
        job.save()

    def _submit_meeting(self, folder: Path, vault: str) -> None:
        job = jobs.Job.new("meeting", folder, vault)
        self.queue.submit(job)
        log.info("meeting %s queued for %s", folder.name, vault)
        self.refresh_tray()

    # -- files (Send to, --add, tray, inbox)

    def add_files(self, paths: list[str], vault: str | None = None) -> int:
        """Queue file jobs for media files where they lie. No vault (or an unknown one): the picker asks."""
        files = []
        for p in paths:
            path = Path(p)
            if path.is_file() and is_media(path):
                files.append(path.resolve())
            else:
                log.warning("not a media file, skipped: %s", path.name)
        if not files:
            self.overlay.flash_error()
            return 0
        if vault in self.vaults():
            self._submit_files(files, vault)
        else:
            if vault:
                log.warning("unknown vault %r; asking", vault)
            self.overlay.pick_vault(self.vaults(), self.default_vault(), int(self.settings.get("picker_timeout_s", 20)),
                                    lambda v: self._submit_files(files, v))
        return len(files)

    def add_files_picked(self, paths: list[str]) -> None:
        self.add_files(paths, None)

    def _submit_files(self, files: list[Path], vault: str) -> None:
        for f in files:
            job = jobs.Job.new("file", self.new_folder(), vault, source=f)
            self.queue.submit(job)
            log.info("file %s queued for %s", f.name, vault)
        self.refresh_tray()

    def take_inbox_file(self, path: Path, vault: str) -> bool:
        """Move a settled inbox file into a new job folder and queue it. False while another program holds it."""
        path = Path(path)
        folder = self.new_folder()
        # In a subfolder of its own: a dropped "Recording.m4a" or "audio.flac" can never meet Scribe's outputs.
        dest_dir = folder / "source"
        dest_dir.mkdir()
        dest = dest_dir / path.name
        try:
            os.replace(path, dest)  # same volume: a rename, refused while the writer still has the file open
        except OSError as exc:
            log.info("inbox file not ready: %s (%s)", path.name, type(exc).__name__)
            _remove_if_empty(dest_dir)
            _remove_if_empty(folder)
            return False
        job = jobs.Job.new("file", folder, vault, source=dest)
        self.queue.submit(job)
        log.info("inbox file %s queued for %s", path.name, vault)
        self.refresh_tray()
        return True

    def retry_jobs(self, folders: list[str] | None = None) -> int:
        """Retry failed jobs: the named folders, or every failed job of this session. A Claude run they
        started is resumed, not run again."""
        targets = [Path(f) for f in folders] if folders else [j.folder for j in self._failed_jobs()]
        done = 0
        for folder in targets:
            try:
                self.queue.retry(folder)
                done += 1
            except Exception:
                log.exception("retry of %s failed", folder)
        self.refresh_tray()
        return done

    def _failed_jobs(self) -> list:
        failed = getattr(self.queue, "failed", None)
        if failed is not None:
            return list(failed())
        return [j for j in self.queue.jobs() if j.status == "failed"]

    def send_jobs(self, folders: list[str], vault: str | None) -> int:
        """Send finished jobs (usually Local) on to a vault the user names. No vault, nothing sent."""
        if vault not in self.settings.get("vaults", {}):
            log.warning("enhance: unknown vault %r", vault)
            self.overlay.flash_error()
            return 0
        sent = 0
        for folder in folders or []:
            try:
                self.queue.send_to_vault(Path(folder), vault)
                sent += 1
            except Exception:
                log.exception("sending %s to %s failed", folder, vault)
        self.refresh_tray()
        return sent

    def local_jobs(self, limit: int = 5) -> list[tuple[str, Path]]:
        """Finished Local jobs of this session, newest first, for the tray's Send menu: (label, folder)."""
        out = []
        for j in reversed(self.queue.jobs()):
            if j.vault == jobs.LOCAL and j.status == "done":
                out.append((job_label(j), j.folder))
        return out[:limit]

    def open_recordings(self) -> None:
        os.startfile(config.recordings_dir(self.settings))

    def open_inbox(self) -> None:
        folder = config.output_root(self.settings) / "Inbox"
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    # -- jobs, tray, status

    def on_job(self, job: jobs.Job) -> None:
        """JobQueue notify: refresh the tray; one notification when a job ends."""
        if job.terminal:
            with self._lock:
                fresh = self._notified.get(job.id) != job.status
                self._notified[job.id] = job.status
            if fresh:
                title, message = job_notice(job)
                self.tray_notice(message, title)
        else:
            with self._lock:
                self._notified.pop(job.id, None)
        self.refresh_tray()

    def tray_notice(self, message: str, title: str = "Scribe") -> None:
        if self.tray is not None:
            self.tray.notify(title, message)

    def refresh_tray(self) -> None:
        if self.tray is None:
            return
        with self._lock:
            dictating = self._holding or self._pending > 0
            recording = self._recorder is not None
        try:
            count = int(self.queue.active())
            waiting = any(j.waiting for j in self.queue.jobs())
        except Exception:
            count, waiting = 0, False
        self.tray.set_state(*ui.pick_state(dictating, recording, count), waiting=waiting)

    def status(self) -> dict:
        with self._lock:
            recorder = self._recorder
            folder = self._rec_folder
        try:
            elapsed = round(float(recorder.elapsed()), 1) if recorder is not None else 0.0
        except Exception:
            elapsed = 0.0
        return {
            "running": True,
            "pid": os.getpid(),
            "dictation": self.model_state,
            "recording": recorder is not None,
            "recording_folder": str(folder) if folder else None,
            "elapsed": elapsed,
            "jobs": [{"id": j.id, "kind": j.kind, "vault": j.vault, "status": j.status, "waiting": j.waiting,
                      "error": j.error} for j in self.queue.jobs()],
        }

    def handle_commands(self, commands: list[dict]) -> dict:
        """IPC entry: run each command; status is answered, the rest are started and acknowledged."""
        reply: dict = {"ok": True}
        for c in commands:
            kind = c.get("cmd")
            if kind == "status":
                reply["status"] = self.status()
            elif kind == "toggle":
                self.toggle_notetaker()
            elif kind == "add":
                reply["added"] = self.add_files([str(p) for p in c.get("paths") or []], c.get("vault"))
            elif kind == "retry":
                reply["retried"] = self.retry_jobs([str(p) for p in c.get("folders") or []] or None)
            elif kind == "enhance":
                reply["sent"] = self.send_jobs([str(p) for p in c.get("folders") or []], c.get("vault"))
            elif kind == "quit":
                self.quit()
            elif kind == "show":
                pass
            else:
                return {"ok": False, "error": f"unknown command: {kind}"}
        return reply


# ---------------------------------------------------------------- process setup


def setup_logging(level: int = logging.INFO) -> Path:
    """Rotating log at %USERPROFILE%\\.scribe\\logs\\scribe.log for the whole scribe.* tree."""
    path = config.logs_dir() / "scribe.log"
    logger = logging.getLogger("scribe")
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5, encoding="utf-8",
                                                       delay=True)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return path


def install_excepthooks() -> None:
    """Unhandled exceptions in any thread go to the log, never to a dialog or a console."""
    def main_hook(exc_type, exc, tb):
        log.critical("unhandled exception", exc_info=(exc_type, exc, tb))

    def thread_hook(args: threading.ExceptHookArgs):
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread else "?"
        log.critical("unhandled exception in thread %s", name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = main_hook
    threading.excepthook = thread_hook


def _ensure_stdio() -> None:
    """pythonw has no stdout/stderr; libraries that write progress (tqdm, argparse) must not crash on None."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _emit(data: dict) -> None:
    try:
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
        sys.stdout.flush()
    except (AttributeError, OSError, ValueError):
        pass


def run_client(commands: list[dict], key_path: Path, port: int = IPC_PORT) -> int:
    """A second launch: hand the commands to the running Scribe and exit."""
    reply = deliver(commands, key_path, port)
    if reply is None:
        return 1
    if "status" in reply:
        _emit(reply["status"])
    if not reply.get("ok"):
        log.error("running Scribe refused: %s", reply.get("error"))
        return 1
    return 0


def cleanup_terms(settings: dict, lexicon) -> list[str]:
    """The clean-up model's spelling list: the configured vocabulary, then learned spellings only from pairs that
    are safe as orders (confirmed, or a name or coined spelling). The prompt says "spell these exactly" and the
    guards allow their words, so an unconfirmed "with" -> "Will" must stay out."""
    from . import cleanup, learn
    confirmed = set(lexicon.confirmed())
    learned = [w for h, w in lexicon.pairs()
               if learn.is_special(w) and ((h, w) in confirmed or cleanup.order_pair(h, w))]
    return list(settings.get("vocabulary") or []) + learned


def run_app(commands: list[dict]) -> int:
    """The first launch: build everything, run the Tk loop on this thread, tear down."""
    import tkinter as tk

    from . import asr, audio, cleanup, diarize, learn, paste, winkeys

    settings = config.load_settings()
    jobs.kill_children_with_me()    # ffmpeg, the labelling child and Claude end with Scribe, never orphaned
    vault_folders = {v: config.inbox_dir(v, settings) for v in config.vault_names(settings)}
    key = load_or_create_key(config.APP_DIR / KEY_FILE)

    ui.enable_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda et, ev, tb: log.error("Tk callback failed", exc_info=(et, ev, tb))
    widget = settings.get("widget") or {}
    overlay = ui.Overlay(root, idle_pill=bool(widget.get("idle_pill", True)),
                         anchor="centre" if widget.get("anchor") == "centre" else "right")

    asr_lock = threading.Lock()
    job_queue = jobs.JobQueue(settings, asr_lock=asr_lock)
    d = settings["dictation"]
    lexicon = learn.Lexicon()
    lc = {**config.DEFAULTS["learn"], **(settings.get("learn") or {})}
    edit_watcher = learn.EditWatcher(lexicon, poll_s=float(lc["poll_s"]), max_s=float(lc["max_s"]),
                                     settle_s=float(lc["settle_s"])) if lc.get("enabled", True) else None
    # Lexicon.apply has already respelled the confirmed pairs. The Cleaner sends the model only the remaining
    # pairs that are safe as orders (names and coined spellings corrected once), never common words, case-only
    # changes or one ordinary word for another. relevant_pairs keeps the first of a clash: pairs() is most used
    # first.
    cleaner = cleanup.Cleaner(settings, terms_fn=lambda: cleanup_terms(settings, lexicon), pairs_fn=lexicon.pairs)
    dictation = asr.DictationASR(d["model"], list(settings["languages"]), list(settings["vocabulary"]),
                                 strip_fillers=bool(d.get("strip_fillers", True)), prompt_terms=lexicon.terms)
    voiceprint = None
    if d.get("enroll_voice", True):
        try:
            voiceprint = diarize.Voiceprint(config.APP_DIR / "voiceprint.json", config.models_dir())
        except Exception:
            log.exception("voiceprint unavailable")
    mic = audio.MicCapture()
    app = App(settings, overlay, None, job_queue, mic=mic, asr=dictation, voiceprint=voiceprint,
              paste=lambda text: paste.paste_text(text, hold_active=app.is_holding),
              recorder_factory=audio.MeetingRecorder,
              on_quit=lambda: overlay.call(root.quit), asr_lock=asr_lock,
              lexicon=lexicon, cleaner=cleaner, edit_watcher=edit_watcher)
    overlay.set_toggle(app.toggle_notetaker)
    app.tray = ui.Tray(app.toggle_notetaker, lambda: overlay.ask_files(app.add_files_picked), app.open_recordings,
                       app.open_inbox, app.quit, is_recording=app.is_recording,
                       on_retry=app.retry_jobs, has_failed=lambda: bool(app._failed_jobs()),
                       sendable=app.local_jobs, vaults=lambda: list(settings.get("vaults", {})),
                       on_send=lambda folder, vault: app.send_jobs([str(folder)], vault))
    job_queue.notify = app.on_job

    hotkey = winkeys.HoldHotkey(app.on_press, app.on_release, app.on_cancel,
                                accept_injected=bool(settings.get("test_accept_injected_keys", False)))
    app.start(hotkey)
    app.loading.wait(10.0)          # the dictation model takes asr_lock before a resumed job can
    job_queue.start()               # before any recording: recovers interrupted recordings
    app.tray.start()
    if not getattr(hotkey, "running", True):
        app.tray_notice("✕ Ctrl+Win")
    cleaner.warm()                  # its own thread: loads the clean-up model so the first dictation is quick
    spec = str(settings.get("notetaker_hotkey") or "").strip()      # "" turns the shortcut off
    if spec:
        try:
            app.shortcut = winkeys.Hotkey(spec, app.toggle_notetaker)
        except ValueError as exc:
            log.error("notetaker_hotkey not usable: %s", exc)
            app.tray_notice("✕ Shortcut")
        else:
            if not app.shortcut.start():
                app.tray_notice(f"✕ {app.shortcut.label}")
    app.ipc = IpcServer(app.handle_commands, key)
    try:
        app.ipc.start()
    except OSError:
        log.exception("IPC port %d unavailable; second launches cannot reach this one", IPC_PORT)
        app.ipc = None
    app.watcher = InboxWatcher(vault_folders, app.take_inbox_file)
    try:
        app.watcher.start()
    except Exception:
        log.exception("inbox watcher not started")
    app.refresh_tray()
    for c in commands:
        if c.get("cmd") in ("toggle", "add", "retry", "enhance"):
            app.handle_commands([c])
    log.info("Scribe running (pid %d)", os.getpid())

    try:
        root.mainloop()
    finally:
        app.shutdown()
        try:
            overlay.close()
            root.destroy()
        except Exception:
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"   # models come from the local cache only; nothing is downloaded at run time
    _ensure_stdio()
    argv = sys.argv[1:] if argv is None else list(argv)
    setup_logging()
    install_excepthooks()
    ns = parse_args(argv)
    if ns is None:
        return 2
    commands = commands_from_args(ns)
    key_path = config.APP_DIR / KEY_FILE
    instance = SingleInstance()
    if not instance.acquire():
        code = run_client(commands, key_path)
        logging.shutdown()
        return code
    kinds = {c["cmd"] for c in commands}
    if kinds <= {"status", "quit"}:
        instance.release()
        if "status" in kinds:
            _emit({"running": False})
        logging.shutdown()
        return 1 if "status" in kinds else 0
    try:
        log.info("Scribe starting")
        code = run_app(commands)
    except Exception:
        log.critical("Scribe failed", exc_info=True)
        code = 1
    instance.release()
    log.info("Scribe stopped")
    logging.shutdown()
    # A resumed job's worker may sit inside Whisper or ffmpeg; job.json makes every step resumable,
    # so end the process outright rather than wait on daemon threads in native code.
    os._exit(code)
