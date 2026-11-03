"""app.py: CLI, single instance, IPC, inbox rules and the dictation / notetaker / file flows.

No window, tray icon, keyboard hook, microphone or model is used: the App gets fakes for all of them.
Everything is written under tempfile dirs; the IPC tests use a free port and a temp key, never 47391.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import uuid
import types
from multiprocessing import connection as mpc
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scribe import app, config, jobs, winkeys  # noqa: E402

logging.getLogger("scribe").addHandler(logging.NullHandler())

SR = 16000


def wait_for(cond, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_settings(root: Path) -> dict:
    s = copy.deepcopy(config.DEFAULTS)
    s["output_root"] = str(root / "Scribe")
    s["vaults"] = {"Work": str(root / "vaultA"), "Personal": str(root / "vaultP")}   # never opened
    s["default_vault"] = "Work"
    s["picker_timeout_s"] = 7
    return s


# ---------------------------------------------------------------- fakes


class FakeOverlay:
    def __init__(self):
        self.calls: list = []
        self.pick = "Personal"           # what the user "clicks" in the picker; None leaves it open
        self.picks: list = []
        self.lock = threading.Lock()

    def _rec(self, *call):
        with self.lock:
            self.calls.append(call)

    def names(self) -> list[str]:
        with self.lock:
            return [c[0] for c in self.calls]

    def show_listening(self, level_fn):
        self._rec("listening", level_fn())

    def show_working(self):
        self._rec("working")

    def hide(self):
        self._rec("hide")

    def flash_error(self):
        self._rec("error")

    def show_recording(self, elapsed_fn, on_stop):
        self._rec("recording", elapsed_fn(), on_stop)

    def hide_recording(self):
        self._rec("hide_recording")

    def pick_vault(self, vaults, default, timeout_s, on_pick):
        self._rec("pick", list(vaults), default, timeout_s)
        self.picks.append(on_pick)
        if self.pick is not None:
            on_pick(self.pick)


class FakeTray:
    def __init__(self):
        self.states: list = []
        self.notes: list = []
        self.stopped = False

    def set_state(self, state, count=0, waiting=False):
        self.states.append((state, count))

    def notify(self, title, message):
        self.notes.append((title, message))

    def stop(self):
        self.stopped = True


class FakeQueue:
    def __init__(self):
        self.submitted: list[jobs.Job] = []
        self.retried: list[Path] = []
        self.sent: list[tuple[Path, str]] = []

    def submit(self, job):
        self.submitted.append(job)

    def retry(self, folder):
        self.retried.append(Path(folder))

    def send_to_vault(self, folder, vault):
        self.sent.append((Path(folder), vault))

    def active(self):
        return len(self.submitted)

    def jobs(self):
        return list(self.submitted)


class FakeMic:
    def __init__(self, seconds: float = 1.0):
        self.seconds = seconds
        self.running = False
        self.starts = 0
        self.fail = False
        self.closed = False

    def start(self):
        if self.fail:
            raise OSError("no microphone")
        self.starts += 1
        self.running = True

    def stop(self):
        self.running = False
        return np.full(int(self.seconds * SR), 0.1, dtype=np.float32)

    def level(self):
        return 0.25 if self.running else 0.0

    def close(self):
        self.closed = True


class FakeASR:
    def __init__(self, text: str = "Send the numbers to BlueSky."):
        self.text = text
        self.fail = False
        self.loaded = False
        self.seen: list[int] = []
        self.lock = None                      # the app's asr_lock, to check it is held while loading
        self.lock_held_during_load = None

    def load(self):
        if self.lock is not None:
            self.lock_held_during_load = self.lock.locked()
        self.loaded = True

    def transcribe(self, audio):
        self.seen.append(len(audio))
        if self.fail:
            raise RuntimeError("cuda")
        return self.text


class FakeVoiceprint:
    def __init__(self, count: int = 0):
        self.count = count
        self.enrolled_clips: list[int] = []

    def enroll(self, audio):
        self.enrolled_clips.append(len(audio))
        self.count += 1
        return True


class FakeRecorder:
    seconds = 120.0
    fail = False
    made: list = []
    raised: set = set()                  # what alerts() reports (disk, silence, cap)

    def __init__(self, folder):
        self.folder = Path(folder)
        self.started = self.stopped = False
        FakeRecorder.made.append(self)

    def start(self):
        if FakeRecorder.fail:
            raise OSError("no device")
        self.started = True

    def stop(self):
        self.stopped = True
        return {"mic": self.folder / "mic.flac", "system": self.folder / "system.flac",
                "seconds": FakeRecorder.seconds, "events": []}

    def elapsed(self):
        return 12.0

    def alerts(self):
        return set(FakeRecorder.raised)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- CLI


class TestCommandLine(unittest.TestCase):
    def cmds(self, argv, cwd=None):
        ns = app.parse_args(argv)
        self.assertIsNotNone(ns, argv)
        return app.commands_from_args(ns, cwd=cwd)

    def test_plain_launch(self):
        self.assertEqual(self.cmds([]), [{"cmd": "show"}])

    def test_toggle(self):
        self.assertEqual(self.cmds(["--toggle-notetaker"]), [{"cmd": "toggle"}])

    def test_send_to_style_appends_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = str(Path(tmp) / "a call.mp4"), str(Path(tmp) / "b.m4a")
            out = self.cmds(["--vault", "Personal", "--add", a, b])
        self.assertEqual(out, [{"cmd": "add", "paths": [os.path.abspath(a), os.path.abspath(b)], "vault": "Personal"}])

    def test_add_then_vault_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.cmds(["--add", "x.wav", "--vault", "Local"], cwd=Path(tmp))
            self.assertEqual(out[0]["paths"], [os.path.abspath(Path(tmp) / "x.wav")])
        self.assertEqual(out[0]["vault"], "Local")

    def test_add_without_vault(self):
        out = self.cmds(["--add", r"C:\x\y.mp3"])
        self.assertIsNone(out[0]["vault"])

    def test_repeated_add_extends(self):
        out = self.cmds(["--add", r"C:\a.mp3", "--add", r"C:\b.mp3"])
        self.assertEqual(len(out[0]["paths"]), 2)

    def test_quit_wins(self):
        self.assertEqual(self.cmds(["--quit", "--toggle-notetaker"]), [{"cmd": "quit"}])

    def test_status_and_toggle_together(self):
        self.assertEqual(self.cmds(["--status", "--toggle-notetaker"]), [{"cmd": "status"}, {"cmd": "toggle"}])

    def test_retry_and_enhance(self):
        cmds = self.cmds(["--retry", "a", "--enhance", "b", "--vault", "Personal"], cwd=Path("C:/work"))
        self.assertEqual(cmds, [{"cmd": "retry", "folders": [os.path.abspath("C:/work/a")]},
                                {"cmd": "enhance", "folders": [os.path.abspath("C:/work/b")], "vault": "Personal"}])

    def test_bad_arguments_return_none(self):
        self.assertIsNone(app.parse_args(["--bogus"]))
        self.assertIsNone(app.parse_args(["--add"]))


# ---------------------------------------------------------------- single instance and IPC


class TestSingleInstance(unittest.TestCase):
    def test_acquire_release(self):
        name = f"Local\\ScribeTest-{uuid.uuid4().hex}"
        first, second = app.SingleInstance(name), app.SingleInstance(name)
        self.assertTrue(first.acquire())
        self.assertTrue(first.acquire())          # idempotent for the owner
        self.assertFalse(second.acquire())
        first.release()
        self.assertTrue(second.acquire())
        self.assertFalse(app.SingleInstance(name).acquire())
        second.release()
        with app.SingleInstance(name) as third:
            self.assertTrue(third.acquire())
        self.assertIsNone(third._handle)

    def test_default_name(self):
        self.assertEqual(app.MUTEX_NAME, "Local\\ScribeSingleton")


class TestKey(unittest.TestCase):
    def test_created_once_then_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Scribe" / "ipc.key"
            with self.assertRaises(OSError):
                app.read_key(path)
            key = app.load_or_create_key(path)
            self.assertEqual(len(key), 64)
            int(key, 16)
            self.assertEqual(app.load_or_create_key(path), key)
            self.assertEqual(app.read_key(path), key)
            path.write_bytes(b"short")
            with self.assertRaises(OSError):
                app.read_key(path)
            self.assertNotEqual(app.load_or_create_key(path), b"short")


class TestIpc(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.key = os.urandom(16).hex().encode()
        self.received: list = []

        def handler(commands):
            self.received.append(commands)
            return {"ok": True, "echo": commands}

        self.server = app.IpcServer(handler, self.key, port=self.port)
        self.server.start()

    def tearDown(self):
        self.server.close()

    def test_roundtrip(self):
        cmds = [{"cmd": "add", "paths": ["C:\\a b\\ç.mp4"], "vault": "Personal"}, {"cmd": "status"}]
        reply = app.send_commands(cmds, self.key, port=self.port)
        self.assertEqual(reply, {"ok": True, "echo": cmds})
        self.assertEqual(self.received, [cmds])

    def test_wrong_key_refused_and_server_survives(self):
        with self.assertRaises(mpc.AuthenticationError):
            app.send_commands([{"cmd": "quit"}], b"0" * 64, port=self.port)
        self.assertEqual(self.received, [])
        self.assertTrue(app.send_commands([{"cmd": "show"}], self.key, port=self.port)["ok"])

    def test_garbage_client_does_not_stop_server(self):
        with socket.create_connection(("127.0.0.1", self.port)) as s:
            s.sendall(b"\x00\x00\x00\x05hello")
        self.assertTrue(app.send_commands([{"cmd": "show"}], self.key, port=self.port)["ok"])

    def test_bad_message_gets_error_reply(self):
        conn = mpc.Client(("127.0.0.1", self.port), family="AF_INET", authkey=self.key)
        with conn:
            conn.send_bytes(b'{"commands": "not a list"}')
            self.assertTrue(conn.poll(5))
            reply = json.loads(conn.recv_bytes())
        self.assertFalse(reply["ok"])

    def test_deliver_reads_key_file_and_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "ipc.key"
            threading.Timer(0.3, lambda: key_path.write_bytes(self.key)).start()
            reply = app.deliver([{"cmd": "status"}], key_path, port=self.port, wait_s=5)
        self.assertEqual(reply["echo"], [{"cmd": "status"}])

    def test_deliver_gives_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "ipc.key"
            key_path.write_bytes(self.key)
            self.assertIsNone(app.deliver([{"cmd": "status"}], key_path, port=free_port(), wait_s=0.3))

    def test_delivered_commands_are_never_resent(self):
        slow_calls = []

        def slow(commands):
            slow_calls.append(commands)
            time.sleep(1.0)                    # a slow mapped drive: the answer comes after the client gave up
            return {"ok": True}

        port = free_port()
        server = app.IpcServer(slow, self.key, port=port)
        server.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                key_path = Path(tmp) / "ipc.key"
                key_path.write_bytes(self.key)
                with self.assertLogs("scribe.app", "ERROR"):
                    self.assertIsNone(app.deliver([{"cmd": "add", "paths": ["x.mp4"]}], key_path, port=port,
                                                  wait_s=5, timeout=0.2))
            time.sleep(1.2)
            self.assertEqual(len(slow_calls), 1)
        finally:
            server.close()

    def test_port_taken_raises(self):
        other = app.IpcServer(lambda c: {"ok": True}, self.key, port=self.port)
        with self.assertRaises(OSError):
            other.start()


# ---------------------------------------------------------------- pure rules


class TestRules(unittest.TestCase):
    def test_is_media(self):
        for name in ("a.mp4", "B.MKV", "c.mov", "d.webm", "e.m4a", "f.mp3", "g.wav", "h.flac", "i.ogg", "j.aac"):
            self.assertTrue(app.is_media(Path(name)), name)
        for name in ("a.txt", "a.mp4.crdownload", "a.part", "~$a.mp4", ".a.mp4", "a"):
            self.assertFalse(app.is_media(Path(name)), name)

    def test_should_enroll(self):
        self.assertTrue(all(app.should_enroll(n, i) for n, i in zip(range(30), range(1, 31))))
        picks = [i for i in range(31, 81) if app.should_enroll(30, i)]
        self.assertEqual(picks, [40, 50, 60, 70, 80])
        self.assertTrue(app.should_enroll(29, 7))
        self.assertFalse(app.should_enroll(100, 11))

    def test_job_notice(self):
        folder = Path("C:/x/Recordings/2026-10-30 1405")
        done = jobs.Job.new("meeting", folder, "Work")
        done.status = "done"
        self.assertEqual(app.job_notice(done), ("Scribe", "✓ Meeting 14:05 · Work"))
        failed = jobs.Job.new("file", folder.with_name("2026-10-30 1405-2"), "Local", source=Path("C:/v/Board call.mp4"))
        failed.status = "failed"
        self.assertEqual(app.job_notice(failed), ("Scribe", "✕ Board call · Local"))
        odd = jobs.Job.new("meeting", Path("C:/x/weird"), "Personal")
        odd.status = "done"
        self.assertEqual(app.job_notice(odd)[1], "✓ Meeting · Personal")


class TestStableFiles(unittest.TestCase):
    def test_ready_after_stable_period(self):
        sf = app.StableFiles(5.0)
        p = Path("x.mp4")
        sf.touch(p, 0.0)
        self.assertFalse(sf.check(p, (10, 1), 0.0))     # first sight sets the baseline
        self.assertFalse(sf.check(p, (10, 1), 4.9))
        self.assertTrue(sf.check(p, (10, 1), 5.0))

    def test_growth_resets(self):
        sf = app.StableFiles(5.0)
        p = Path("x.mp4")
        sf.check(p, (10, 1), 0.0)
        self.assertFalse(sf.check(p, (20, 2), 4.0))
        self.assertFalse(sf.check(p, (20, 2), 8.9))
        self.assertTrue(sf.check(p, (20, 2), 9.0))

    def test_same_size_new_time_resets(self):
        sf = app.StableFiles(5.0)
        p = Path("x.mp4")
        sf.check(p, (10, 1), 0.0)
        self.assertFalse(sf.check(p, (10, 2), 6.0))      # preallocated file still being written
        self.assertTrue(sf.check(p, (10, 2), 11.0))

    def test_restart_keeps_signature(self):
        sf = app.StableFiles(5.0)
        p = Path("x.mp4")
        sf.check(p, (10, 1), 0.0)
        sf.restart(p, 5.0)
        self.assertFalse(sf.check(p, (10, 1), 9.9))
        self.assertTrue(sf.check(p, (10, 1), 10.0))

    def test_touch_restarts(self):
        sf = app.StableFiles(5.0)
        p = Path("x.mp4")
        sf.check(p, (10, 1), 0.0)
        sf.touch(p, 4.0)
        self.assertFalse(sf.check(p, (10, 1), 6.0))
        self.assertTrue(sf.check(p, (10, 1), 11.0))

    def test_empty_never_ready_and_missing_forgotten(self):
        sf = app.StableFiles(5.0)
        p = Path("x.mp4")
        sf.check(p, (0, 1), 0.0)
        self.assertFalse(sf.check(p, (0, 1), 60.0))
        self.assertFalse(sf.check(p, None, 61.0))
        self.assertEqual(sf.paths(), [])


# ---------------------------------------------------------------- inbox watcher


class TestInboxWatcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.folders = {"Work": self.root / "Inbox" / "Work", "Local": self.root / "Inbox" / "Local"}
        for f in self.folders.values():
            f.mkdir(parents=True)
        self.taken: list = []
        self.accept = True

    def tearDown(self):
        self.tmp.cleanup()

    def on_file(self, path, vault):
        self.taken.append((path.name, vault))
        if self.accept:
            path.unlink()
        return self.accept

    def test_check_once_with_fake_clock(self):
        clock = FakeClock()
        w = app.InboxWatcher(self.folders, self.on_file, stable_s=5.0, clock=clock)
        f = self.folders["Local"] / "talk.wav"
        f.write_bytes(b"x" * 100)
        (self.folders["Local"] / "notes.txt").write_text("no")
        w.scan()
        self.assertEqual(w.check_once(), 0)
        clock.t += 4.0
        self.assertEqual(w.check_once(), 0)
        clock.t += 1.0
        self.assertEqual(w.check_once(), 1)
        self.assertEqual(self.taken, [("talk.wav", "Local")])
        clock.t += 10
        self.assertEqual(w.check_once(), 0)

    def test_busy_file_is_retried(self):
        clock = FakeClock()
        w = app.InboxWatcher(self.folders, self.on_file, stable_s=5.0, clock=clock)
        f = self.folders["Work"] / "call.mp4"
        f.write_bytes(b"x" * 100)
        w.add(f, "Work")
        self.accept = False
        w.check_once()
        clock.t += 5
        self.assertEqual(w.check_once(), 0)
        self.assertEqual(len(self.taken), 1)
        self.accept = True
        clock.t += 4
        self.assertEqual(w.check_once(), 0)          # a fresh stable period after a refusal
        clock.t += 1
        self.assertEqual(w.check_once(), 1)

    def test_files_outside_the_inbox_are_ignored(self):
        clock = FakeClock()
        w = app.InboxWatcher(self.folders, self.on_file, stable_s=1.0, clock=clock)
        elsewhere = self.root / "Recordings" / "2026-10-30 1405"
        elsewhere.mkdir(parents=True)
        moved = elsewhere / "call.mp4"            # what an inbox file looks like after it was taken
        moved.write_bytes(b"x" * 10)
        w.add(moved, "Work")
        w.add(self.folders["Local"] / "x.wav", "Work")   # right file, wrong vault folder
        clock.t += 10
        w.check_once()
        clock.t += 10
        self.assertEqual(w.check_once(), 0)
        self.assertEqual(self.taken, [])

    def test_vanished_file_is_dropped(self):
        clock = FakeClock()
        w = app.InboxWatcher(self.folders, self.on_file, stable_s=1.0, clock=clock)
        f = self.folders["Local"] / "gone.mp3"
        f.write_bytes(b"x")
        w.add(f, "Local")
        w.check_once()
        f.unlink()
        clock.t += 5
        self.assertEqual(w.check_once(), 0)
        self.assertEqual(w._vaults, {})

    def test_refresh_recreates_a_deleted_inbox_and_finds_what_was_dropped(self):
        import shutil

        clock = FakeClock()
        w = app.InboxWatcher(self.folders, self.on_file, stable_s=1.0, clock=clock)
        shutil.rmtree(self.folders["Local"])
        w.refresh()
        self.assertTrue(self.folders["Local"].is_dir())
        (self.folders["Local"] / "late.wav").write_bytes(b"x" * 10)     # no watchdog event ever arrives
        w.refresh()
        w.check_once()
        clock.t += 2
        self.assertEqual(w.check_once(), 1)
        self.assertEqual(self.taken, [("late.wav", "Local")])

    def test_refresh_rewatches_a_recreated_folder(self):
        import shutil

        w = app.InboxWatcher(self.folders, self.on_file)
        w._observer = types.SimpleNamespace(schedule=mock.Mock(return_value="watch"), unschedule=mock.Mock())
        w._inodes = {v: w._inode(f) for v, f in self.folders.items()}
        shutil.rmtree(self.folders["Work"])
        self.folders["Work"].mkdir()
        (self.folders["Work"] / "pad.txt").write_text("x")      # a new file index for the new folder
        w.refresh()
        self.assertEqual([c.args[1] for c in w._observer.schedule.call_args_list],
                         [str(self.folders["Work"])])

    def test_real_watchdog(self):
        got = threading.Event()

        def on_file(path, vault):
            self.taken.append((path.name, vault))
            got.set()
            return True

        w = app.InboxWatcher(self.folders, on_file, stable_s=0.3, poll_s=0.05)
        w.start()
        try:
            (self.folders["Work"] / "dropped.m4a").write_bytes(b"x" * 1000)
            self.assertTrue(got.wait(5.0))
        finally:
            w.stop()
        self.assertEqual(self.taken, [("dropped.m4a", "Work")])


# ---------------------------------------------------------------- the app with fakes


class AppCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.settings = make_settings(self.root)
        self.overlay, self.tray, self.queue = FakeOverlay(), FakeTray(), FakeQueue()
        self.mic, self.asr, self.vp = FakeMic(), FakeASR(), FakeVoiceprint()
        self.pasted: list[str] = []
        self.paste_ok = True
        self.clock = FakeClock()
        FakeRecorder.seconds, FakeRecorder.fail, FakeRecorder.made, FakeRecorder.raised = 120.0, False, [], set()
        self.lock = threading.Lock()
        self.asr.lock = self.lock
        self.quit_called = threading.Event()
        self.app = app.App(self.settings, self.overlay, self.tray, self.queue, mic=self.mic, asr=self.asr,
                           voiceprint=self.vp, paste=self.paste, recorder_factory=FakeRecorder,
                           on_quit=self.quit_called.set, asr_lock=self.lock, clock=self.clock)
        self.app.start(None)

    def tearDown(self):
        self.app.shutdown()
        self.tmp.cleanup()

    def paste(self, text):
        self.pasted.append(text)
        return self.paste_ok

    def dictate(self, seconds):
        self.mic.seconds = seconds
        self.app.on_press()
        self.app.on_release()

    def idle(self):
        return wait_for(lambda: self.app._pending == 0)


class TestStartup(AppCase):
    def test_model_loaded_under_the_asr_lock(self):
        self.assertTrue(self.app.loading.wait(2))
        self.assertTrue(wait_for(lambda: self.app.model_state == "ready"))
        self.assertTrue(self.asr.loaded)
        self.assertIs(self.asr.lock_held_during_load, True)
        self.assertTrue(self.lock.acquire(timeout=2))
        self.lock.release()

    def test_hotkey_failure_does_not_stop_the_app(self):
        class BadHotkey:
            def start(self):
                raise OSError("hook refused")

            def stop(self):
                pass

        a = app.App(self.settings, FakeOverlay(), self.tray, FakeQueue(), mic=FakeMic(), asr=FakeASR(),
                    paste=self.paste, recorder_factory=FakeRecorder)
        a.start(BadHotkey())
        self.assertEqual(self.tray.notes[-1][1], "✕ Ctrl+Win")
        a.shutdown()


class FakeHotkey:
    def __init__(self):
        self.starts = 0
        self.running = True

    def start(self):
        self.starts += 1
        self.running = True

    def stop(self):
        self.running = False


class TestHookWatch(AppCase):
    """The maintenance loop re-installs the keyboard hook after a stall, not 10 minutes later."""

    def setUp(self):
        super().setUp()
        mock.patch.object(winkeys, "modifiers_down", return_value=False).start()
        self.addCleanup(mock.patch.stopall)

    def test_a_dead_hook_is_reinstalled(self):
        hk = FakeHotkey()
        self.app.hotkey = hk
        hk.running = False                     # Windows removed it
        self.assertTrue(wait_for(lambda: hk.starts >= 1, 2.0))

    def test_a_late_wake_up_reinstalls_the_hook(self):
        hk = FakeHotkey()
        offset = {"s": 0.0}
        real = time.monotonic
        fake_time = types.SimpleNamespace(monotonic=lambda: real() + offset["s"], sleep=time.sleep)
        with mock.patch.object(app, "time", fake_time):
            self.app.hotkey = hk
            time.sleep(0.3)
            self.assertEqual(hk.starts, 0)           # healthy and on time: left alone
            offset["s"] += 1.0                        # the next wake-up looks a second late (a GIL stall)
            self.assertTrue(wait_for(lambda: hk.starts == 1, 2.0))
            time.sleep(0.3)
            self.assertEqual(hk.starts, 1)

    def test_no_reinstall_during_a_hold(self):
        hk = FakeHotkey()
        self.app.on_press()
        self.app.hotkey = hk
        hk.running = False
        time.sleep(0.4)
        self.assertEqual(hk.starts, 0)
        self.app.on_cancel()
        self.assertTrue(wait_for(lambda: hk.starts == 1, 2.0))


class TestModelNotice(AppCase):
    def test_model_failure_shows_a_mark(self):
        a = app.App(self.settings, FakeOverlay(), self.tray, FakeQueue(), mic=FakeMic(), asr=FakeASR(),
                    paste=self.paste, recorder_factory=FakeRecorder)
        a.asr.load = mock.Mock(side_effect=RuntimeError("model not cached: large-v3-turbo"))
        a.start(None)
        self.assertTrue(wait_for(lambda: a.model_state == "failed"))
        self.assertIn(("Scribe", "✕ Model"), self.tray.notes)
        a.shutdown()


class TestDictation(AppCase):
    def test_hold_speak_release_pastes(self):
        self.dictate(2.0)
        self.assertTrue(self.idle())
        self.assertEqual(self.pasted, ["Send the numbers to BlueSky."])
        self.assertEqual(self.asr.seen, [2 * SR])
        names = self.overlay.names()
        self.assertEqual(names[0], "listening")
        self.assertEqual(self.overlay.calls[0][1], 0.25)          # the pill reads the mic level
        self.assertIn("working", names)
        self.assertTrue(wait_for(lambda: self.overlay.names()[-1] == "hide"))
        self.assertIn(("dictating", 0), self.tray.states)
        self.assertTrue(wait_for(lambda: self.tray.states[-1] == ("idle", 0)))

    def test_too_short_is_dropped(self):
        self.dictate(0.2)
        self.assertTrue(self.idle())
        self.assertEqual(self.asr.seen, [])
        self.assertEqual(self.overlay.names(), ["listening", "hide"])

    def test_cancel_discards(self):
        self.mic.seconds = 2.0
        self.app.on_press()
        self.app.on_cancel()
        self.app.on_release()                 # the release after a cancel does nothing
        time.sleep(0.05)
        self.assertEqual(self.asr.seen, [])
        self.assertFalse(self.mic.running)
        self.assertEqual(self.overlay.names(), ["listening", "hide"])

    def test_silence_pastes_nothing_and_is_not_an_error(self):
        self.asr.text = ""
        self.dictate(2.0)
        self.assertTrue(self.idle())
        self.assertEqual(self.pasted, [])
        self.assertTrue(wait_for(lambda: self.overlay.names()[-1] == "hide"))
        self.assertNotIn("error", self.overlay.names())

    def test_transcription_failure_flashes(self):
        self.asr.fail = True
        self.dictate(2.0)
        self.assertTrue(self.idle())
        self.assertTrue(wait_for(lambda: self.overlay.names()[-1] == "error"))
        self.assertEqual(self.pasted, [])

    def test_mic_failure_flashes(self):
        self.mic.fail = True
        self.app.on_press()
        self.assertEqual(self.overlay.names(), ["error"])
        self.assertFalse(self.app._holding)
        self.app.on_release()
        self.assertEqual(self.asr.seen, [])

    def test_second_press_while_first_transcribes_keeps_listening(self):
        gate = threading.Event()
        real = self.asr.transcribe

        def slow(audio):
            gate.wait(2)
            return real(audio)

        self.asr.transcribe = slow
        self.dictate(1.0)
        self.mic.seconds = 1.0
        self.app.on_press()                   # holding again while the first clip is still working
        gate.set()
        time.sleep(0.3)
        # the first clip waits for the second hold to end: pasting now would time out against Ctrl+Win
        self.assertEqual(self.pasted, [])
        self.assertNotEqual(self.overlay.names()[-1], "hide")   # the second hold's pill stays up
        self.app.on_release()
        self.assertTrue(self.idle())
        self.assertEqual(len(self.pasted), 2)
        self.assertEqual(self.asr.seen, [SR, SR])

    def test_a_cancelled_second_hold_releases_the_first_clip(self):
        gate = threading.Event()
        real = self.asr.transcribe
        self.asr.transcribe = lambda audio: (gate.wait(2), real(audio))[1]
        self.dictate(1.0)
        self.app.on_press()
        gate.set()
        time.sleep(0.2)
        self.assertEqual(self.pasted, [])
        self.app.on_cancel()                  # Ctrl+Win+D: the second hold is dropped, the first clip goes in
        self.assertTrue(self.idle())
        self.assertEqual(self.pasted, ["Send the numbers to BlueSky."])

    def test_failure_does_not_flash_over_a_new_hold(self):
        gate = threading.Event()

        def failing(audio):
            gate.wait(2)
            raise RuntimeError("cuda")

        self.asr.transcribe = failing
        self.dictate(1.0)
        self.app.on_press()                   # the next hold starts while the first clip fails
        gate.set()
        self.assertTrue(wait_for(lambda: self.app._pending == 0))
        time.sleep(0.05)
        self.assertNotIn("error", self.overlay.names())
        self.app.on_cancel()

    def test_voiceprint_fed_by_long_clips_only(self):
        self.dictate(2.0)
        self.dictate(4.0)
        self.assertTrue(self.idle())
        self.assertTrue(wait_for(lambda: self.vp.enrolled_clips == [4 * SR]))

    def test_no_enrolment_when_paste_fails_or_disabled(self):
        self.paste_ok = False
        self.dictate(4.0)
        self.assertTrue(self.idle())
        self.settings["dictation"]["enroll_voice"] = False
        self.paste_ok = True
        self.dictate(4.0)
        self.assertTrue(self.idle())
        time.sleep(0.1)
        self.assertEqual(self.vp.enrolled_clips, [])

    def test_no_enrolment_while_the_notetaker_records(self):
        self.app.toggle_notetaker()
        self.assertTrue(wait_for(self.app.is_recording))
        self.dictate(4.0)                     # a speakerphone call: the clip may hold the other voice
        self.assertTrue(self.idle())
        time.sleep(0.1)
        self.assertEqual(self.pasted, ["Send the numbers to BlueSky."])
        self.assertEqual(self.vp.enrolled_clips, [])

    def test_enrolment_thins_out_after_thirty(self):
        self.vp.count = 30
        for _ in range(10):
            self.dictate(3.5)
            self.assertTrue(self.idle())
        self.assertTrue(wait_for(lambda: len(self.vp.enrolled_clips) == 1))
        time.sleep(0.1)
        self.assertEqual(len(self.vp.enrolled_clips), 1)


class FakeLexicon:
    def __init__(self):
        self.protected = None

    def apply(self, text, protect=()):
        self.protected = list(protect)
        return text.replace("Blue Sky", "BlueSky").replace("numbers", "figures")


class FakeCleaner:
    def __init__(self, out=None, fail=False):
        self.out, self.fail, self.seen = out, fail, []

        self.warmed = 0

    def warm(self):
        self.warmed += 1

    def clean(self, text):
        self.seen.append(text)
        return self.out if self.out is not None else text.upper()


class FakeEditWatcher:
    def __init__(self):
        self.watched, self.stopped = [], False

    def watch(self, pasted):
        self.watched.append(pasted)

    def stop(self):
        self.stopped = True


class TestPolish(AppCase):
    """Learned respellings, then the clean-up model, before the paste; the edit watcher after it."""

    def rebuild(self, **parts):
        self.app.shutdown()
        self.app = app.App(self.settings, self.overlay, self.tray, self.queue, mic=self.mic, asr=self.asr,
                           voiceprint=self.vp, paste=self.paste, recorder_factory=FakeRecorder,
                           on_quit=self.quit_called.set, asr_lock=self.lock, clock=self.clock, **parts)
        self.app.start(None)

    def test_lexicon_then_cleaner_then_paste_then_watch(self):
        lex, cl, ew = FakeLexicon(), FakeCleaner(), FakeEditWatcher()
        self.rebuild(lexicon=lex, cleaner=cl, edit_watcher=ew)
        self.asr.text = "Send the numbers to Blue Sky."
        self.mic.seconds = 2.0
        self.app.on_press()
        self.assertEqual(cl.warmed, 1)                                       # warmed while the user speaks
        self.app.on_release()
        self.assertTrue(self.idle())
        self.assertEqual(cl.seen, ["Send the figures to BlueSky."])        # the cleaner sees the respelled text
        self.assertEqual(self.pasted, ["SEND THE FIGURES TO BLUESKY."])
        self.assertEqual(ew.watched, self.pasted)                            # exactly what was pasted
        self.assertEqual(lex.protected, self.settings["vocabulary"])         # configured terms are left alone
        self.assertEqual(cl.warmed, 1)                                       # once per dictation

    def test_no_watch_when_the_paste_fails(self):
        ew = FakeEditWatcher()
        self.rebuild(edit_watcher=ew)
        self.paste_ok = False
        self.dictate(2.0)
        self.assertTrue(self.idle())
        self.assertEqual(ew.watched, [])

    def test_empty_clean_falls_back_to_the_text(self):
        self.rebuild(cleaner=FakeCleaner(out=""))
        self.dictate(2.0)
        self.assertTrue(self.idle())
        self.assertEqual(self.pasted, ["Send the numbers to BlueSky."])

    def test_a_failing_lexicon_still_pastes(self):
        class Broken:
            def apply(self, text, protect=()):
                raise RuntimeError("bad file")
        self.rebuild(lexicon=Broken())
        self.dictate(2.0)
        self.assertTrue(self.idle())
        self.assertEqual(self.pasted, ["Send the numbers to BlueSky."])

    def test_silence_skips_polish(self):
        cl = FakeCleaner()
        self.rebuild(cleaner=cl)
        self.asr.text = ""
        self.dictate(2.0)
        self.assertTrue(self.idle())
        self.assertEqual(cl.seen, [])
        self.assertEqual(self.pasted, [])

    def test_shutdown_stops_the_watcher_and_the_shortcut(self):
        ew, sc = FakeEditWatcher(), FakeEditWatcher()
        self.rebuild(edit_watcher=ew)
        self.app.shortcut = sc
        self.app.shutdown()
        self.assertTrue(ew.stopped)
        self.assertTrue(sc.stopped)


class TestCleanupTerms(unittest.TestCase):
    """Only learned spellings safe as orders reach the clean-up model's spelling list."""

    def test_unsafe_pairs_stay_out(self):
        from scribe import learn
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            lex = learn.Lexicon(Path(tmp) / "lexicon.json")
            lex.add("with", "Witt")                   # a common word: never an order
            lex.add("cache", "cash")                 # one ordinary word for another, seen once
            lex.add("Kiera", "Kira")                 # a name for a name: safe once
            lex.add("wispr flow", "Wispr Flow")      # a coined name phrase
            terms = app.cleanup_terms({"vocabulary": ["Acme"]}, lex)
        self.assertEqual(terms[0], "Acme")
        self.assertIn("Kira", terms)
        self.assertIn("Wispr Flow", terms)
        self.assertNotIn("Witt", terms)
        self.assertNotIn("cash", terms)


class TestNotetaker(AppCase):
    def toggle(self):
        self.app.toggle_notetaker()
        self.clock.t += 5

    def test_start_stop_pick_submit(self):
        self.toggle()
        self.assertTrue(wait_for(self.app.is_recording))
        rec = FakeRecorder.made[0]
        self.assertTrue(rec.started)
        self.assertTrue(rec.folder.is_dir())
        self.assertEqual(rec.folder.parent, Path(self.settings["output_root"]) / "Recordings")
        rec_call = [c for c in self.overlay.calls if c[0] == "recording"][0]
        self.assertEqual(rec_call[1], 12.0)
        self.assertTrue(wait_for(lambda: self.tray.states and self.tray.states[-1][0] == "recording"))
        self.assertEqual(self.app.status()["recording"], True)

        rec_call[2]()                         # the pill's stop button toggles
        self.assertTrue(wait_for(lambda: self.queue.submitted))
        self.assertTrue(rec.stopped)
        self.assertFalse(self.app.is_recording())
        pick = [c for c in self.overlay.calls if c[0] == "pick"][0]
        self.assertEqual(pick[1:], (["Work", "Personal", "Local"], "Work", 7))
        job = self.queue.submitted[0]
        self.assertEqual((job.kind, job.vault, job.folder, job.source), ("meeting", "Personal", rec.folder, None))
        names = self.overlay.names()
        self.assertLess(names.index("hide_recording"), names.index("pick"))

    def test_double_click_is_one_toggle(self):
        self.app.toggle_notetaker()
        self.app.toggle_notetaker()           # same instant
        self.assertTrue(wait_for(self.app.is_recording))
        time.sleep(0.05)
        self.assertEqual(len(FakeRecorder.made), 1)
        self.assertTrue(self.app.is_recording())

    def test_toggle_inside_debounce_window_is_ignored(self):
        self.app.toggle_notetaker()
        self.assertTrue(wait_for(lambda: self.app.is_recording() and not self.app._toggle_busy))
        self.clock.t += app.TOGGLE_DEBOUNCE_S / 2
        self.app.toggle_notetaker()           # the start has finished, but this is a double click
        time.sleep(0.05)
        self.assertTrue(self.app.is_recording())
        self.assertFalse(FakeRecorder.made[0].stopped)
        self.clock.t += app.TOGGLE_DEBOUNCE_S
        self.app.toggle_notetaker()
        self.assertTrue(wait_for(lambda: FakeRecorder.made[0].stopped))

    def test_short_recording_is_kept_but_not_queued(self):
        FakeRecorder.seconds = 2.0
        self.toggle()
        self.assertTrue(wait_for(self.app.is_recording))
        self.toggle()
        self.assertTrue(wait_for(lambda: not self.app.is_recording() and not self.app._toggle_busy))
        folder = FakeRecorder.made[0].folder
        self.assertTrue(wait_for(lambda: (folder / "job.json").exists()))
        job = jobs.Job.load(folder)
        self.assertEqual(job.status, "failed")
        self.assertIn("too short", job.error)
        self.assertEqual(self.queue.submitted, [])
        self.assertNotIn("pick", self.overlay.names())
        self.assertTrue(jobs.recover_folder(folder, self.settings).terminal)   # start-up never re-queues it

    def test_start_failure_leaves_no_folder(self):
        FakeRecorder.fail = True
        self.toggle()
        self.assertTrue(wait_for(lambda: "error" in self.overlay.names()))
        self.assertTrue(wait_for(lambda: not self.app._toggle_busy))
        self.assertFalse(self.app.is_recording())
        rec_dir = Path(self.settings["output_root"]) / "Recordings"
        self.assertEqual(list(rec_dir.iterdir()), [])

    def test_a_forgotten_notetaker_stops_itself(self):
        self.toggle()
        self.assertTrue(wait_for(self.app.is_recording))
        FakeRecorder.raised = {"silence"}
        self.assertTrue(wait_for(lambda: FakeRecorder.made[0].stopped))
        self.assertTrue(wait_for(lambda: self.queue.submitted))          # through the normal stop path
        self.assertIn(("Scribe", "■ Silence"), self.tray.notes)

    def test_a_full_disk_stops_and_flashes(self):
        self.toggle()
        self.assertTrue(wait_for(self.app.is_recording))
        FakeRecorder.raised = {"disk_low"}
        self.assertTrue(wait_for(lambda: ("Scribe", "⚠ Disk") in self.tray.notes))
        self.assertTrue(self.app.is_recording())                          # low disk is only shown
        FakeRecorder.raised = {"disk_low", "fault"}
        self.assertTrue(wait_for(lambda: FakeRecorder.made[0].stopped))
        self.assertIn(("Scribe", "✕ Disk"), self.tray.notes)
        self.assertIn("error", self.overlay.names())
        self.assertEqual(self.tray.notes.count(("Scribe", "⚠ Disk")), 1)

    def test_quit_closes_the_recording(self):
        self.toggle()
        self.assertTrue(wait_for(self.app.is_recording))
        self.app.quit()
        self.assertTrue(self.quit_called.is_set())
        self.app.toggle_notetaker()           # ignored once quitting
        self.app.shutdown()
        self.assertTrue(FakeRecorder.made[0].stopped)
        self.assertEqual(len(FakeRecorder.made), 1)
        self.assertTrue(self.tray.stopped)
        self.assertTrue(self.mic.closed)


class TestFiles(AppCase):
    def media(self, name="call.mp4", where=None) -> Path:
        folder = where or (self.root / "elsewhere")
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / name
        p.write_bytes(b"x" * 100)
        return p

    def test_add_with_vault_leaves_file_in_place(self):
        a, b = self.media("a.mp4"), self.media("b.m4a")
        n = self.app.add_files([str(a), str(b), str(self.root / "missing.mp4")], "Local")
        self.assertEqual(n, 2)
        self.assertTrue(a.exists() and b.exists())
        self.assertEqual([(j.kind, j.vault, j.source) for j in self.queue.submitted],
                         [("file", "Local", a.resolve()), ("file", "Local", b.resolve())])
        folders = {j.folder for j in self.queue.submitted}
        self.assertEqual(len(folders), 2)          # one job folder each, even within one minute

    def test_add_without_vault_asks(self):
        a = self.media()
        self.overlay.pick = "Work"
        self.app.add_files([str(a)])
        self.assertEqual(self.overlay.names(), ["pick"])
        self.assertEqual(self.queue.submitted[0].vault, "Work")

    def test_no_vaults_offers_local_only(self):
        self.app.settings["vaults"] = {}
        self.app.settings["default_vault"] = "Local"
        self.assertEqual((self.app.vaults(), self.app.default_vault()), (["Local"], "Local"))
        self.app.settings["default_vault"] = "Work"         # a stale default falls back to Local
        self.assertEqual(self.app.default_vault(), "Local")
        self.overlay.pick = "Local"
        self.app.add_files([str(self.media())])
        pick = [c for c in self.overlay.calls if c[0] == "pick"][0]
        self.assertEqual(pick[1:3], (["Local"], "Local"))
        self.assertEqual(self.queue.submitted[0].vault, "Local")
        self.assertEqual(self.app.send_jobs([str(self.root / "a")], "Work"), 0)

    def test_unknown_vault_asks(self):
        a = self.media()
        self.app.add_files([str(a)], "Nope")
        self.assertIn("pick", self.overlay.names())

    def test_non_media_flashes(self):
        t = self.root / "notes.txt"
        t.write_text("x")
        self.assertEqual(self.app.add_files([str(t), str(self.root)], "Local"), 0)
        self.assertEqual(self.overlay.names(), ["error"])
        self.assertEqual(self.queue.submitted, [])

    def test_inbox_file_moves_into_its_job_folder(self):
        inbox = config.inbox_dir("Personal", self.settings)
        src = self.media("board.mp4", inbox)
        self.assertTrue(self.app.take_inbox_file(src, "Personal"))
        self.assertFalse(src.exists())
        job = self.queue.submitted[0]
        self.assertEqual((job.kind, job.vault), ("file", "Personal"))
        self.assertEqual(job.source, job.folder / "source" / "board.mp4")   # never beside Scribe's outputs
        self.assertTrue(job.source.exists())

    def test_inbox_recording_named_like_an_output_keeps_its_name_apart(self):
        inbox = config.inbox_dir("Local", self.settings)
        src = self.media("Recording.m4a", inbox)
        self.assertTrue(self.app.take_inbox_file(src, "Local"))
        job = self.queue.submitted[0]
        self.assertNotEqual(job.source.parent, job.folder)

    def test_locked_inbox_file_is_left_and_folder_removed(self):
        inbox = config.inbox_dir("Local", self.settings)
        src = self.media("busy.wav", inbox)
        with open(src, "rb"):                 # Python opens without FILE_SHARE_DELETE, like a copier would
            ok = self.app.take_inbox_file(src, "Local")
        self.assertFalse(ok)
        self.assertTrue(src.exists())
        self.assertEqual(self.queue.submitted, [])
        self.assertEqual(list((Path(self.settings["output_root"]) / "Recordings").iterdir()), [])


class TestCommandsAndJobs(AppCase):
    def test_handle_commands(self):
        self.assertEqual(self.app.handle_commands([{"cmd": "show"}]), {"ok": True})
        reply = self.app.handle_commands([{"cmd": "status"}])
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["status"]["recording"], False)
        self.assertEqual(reply["status"]["jobs"], [])
        self.assertFalse(self.app.handle_commands([{"cmd": "dance"}])["ok"])
        self.app.handle_commands([{"cmd": "quit"}])
        self.assertTrue(self.quit_called.is_set())

    def test_add_command(self):
        p = self.root / "x.wav"
        p.write_bytes(b"x")
        reply = self.app.handle_commands([{"cmd": "add", "paths": [str(p)], "vault": "Local"}])
        self.assertEqual(reply["added"], 1)
        self.assertEqual(self.queue.submitted[0].source, p.resolve())

    def test_retry_command(self):
        reply = self.app.handle_commands([{"cmd": "retry", "folders": [str(self.root / "a")]}])
        self.assertEqual(reply["retried"], 1)
        self.assertEqual(self.queue.retried, [self.root / "a"])
        failed = jobs.Job.new("meeting", self.root / "2026-10-30 0900", "Personal")
        failed.status = "failed"
        self.queue.submitted.append(failed)
        self.assertEqual(self.app.handle_commands([{"cmd": "retry"}])["retried"], 1)   # every failed job
        self.assertEqual(self.queue.retried[-1], failed.folder)

    def test_enhance_command_needs_a_known_vault(self):
        reply = self.app.handle_commands([{"cmd": "enhance", "folders": [str(self.root / "a")], "vault": "Personal"}])
        self.assertEqual(reply["sent"], 1)
        self.assertEqual(self.queue.sent, [(self.root / "a", "Personal")])
        self.assertEqual(self.app.handle_commands([{"cmd": "enhance", "folders": ["x"], "vault": "Local"}])["sent"], 0)
        self.assertEqual(self.app.handle_commands([{"cmd": "enhance", "folders": ["x"], "vault": None}])["sent"], 0)

    def test_toggle_command(self):
        self.app.handle_commands([{"cmd": "toggle"}])
        self.assertTrue(wait_for(self.app.is_recording))

    def test_job_notifications_once_per_end(self):
        job = jobs.Job.new("meeting", self.root / "2026-10-30 0900", "Work")
        job.status = "transcribing"
        self.app.on_job(job)
        self.assertEqual(self.tray.notes, [])
        job.status = "done"
        self.app.on_job(job)
        self.app.on_job(job)
        self.assertEqual(self.tray.notes, [("Scribe", "✓ Meeting 09:00 · Work")])
        job.status = "queued"                 # retried
        self.app.on_job(job)
        job.status = "failed"
        self.app.on_job(job)
        self.assertEqual(self.tray.notes[-1], ("Scribe", "✕ Meeting 09:00 · Work"))
        self.assertEqual(len(self.tray.notes), 2)

    def test_tray_count_follows_queue(self):
        p = self.root / "x.wav"
        p.write_bytes(b"x")
        self.app.add_files([str(p)], "Local")
        self.assertEqual(self.tray.states[-1], ("working", 1))


class TestProcessSetup(unittest.TestCase):
    def test_excepthooks_log(self):
        old_sys, old_thread = sys.excepthook, threading.excepthook
        try:
            app.install_excepthooks()
            with self.assertLogs("scribe.app", level="CRITICAL") as cm:
                t = threading.Thread(target=lambda: 1 / 0, name="boom")
                t.start()
                t.join()
                try:
                    raise ValueError("main")
                except ValueError:
                    sys.excepthook(*sys.exc_info())
            self.assertEqual(len(cm.records), 2)
            self.assertIn("boom", cm.records[0].getMessage())
        finally:
            sys.excepthook, threading.excepthook = old_sys, old_thread


if __name__ == "__main__":
    unittest.main()
