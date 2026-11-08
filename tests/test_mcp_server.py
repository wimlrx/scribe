"""mcp_server.py: the tools in-process and over stdio with the MCP SDK's own client, against a fake Scribe (a real
app.IpcServer on a free port with a temp key) and temp job folders and vaults.

No GPU, Whisper model, window or real vault is used, and the running Scribe is never reached, except by the live
tests: SCRIBE_LIVE_MCP=1 asks the running Scribe for status and recent_recordings over stdio (read-only), and
SCRIBE_LIVE_MCP_GPU=1 transcribes a growing recording with large-v3-turbo on the GPU, beside whatever runs there.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anyio  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from scribe import app, config, jobs, mcp_server  # noqa: E402
from scribe.mcp_server import Link, Live, NothingYet, Scribe  # noqa: E402

logging.getLogger("scribe").addHandler(logging.NullHandler())

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SR = 16000
SECRET = "Die kwartaalsyfers is vertroulik"      # stands for spoken text: it must never reach a log
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(cond, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())


def tone(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)


def hold_named_mutex(name: str):
    """Own a named mutex on its own thread until release() is set (the GPU mutex Scribe used until 31 Oct 2026)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    taken, release = threading.Event(), threading.Event()

    def run():
        handle = kernel32.CreateMutexW(None, 1, name)
        taken.set()
        release.wait(60)
        kernel32.CloseHandle(handle)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    taken.wait(5)
    return release, t


# ---------------------------------------------------------------- a temp Scribe: settings, folders, vaults

class Env:
    """Settings pointing at temp folders, temp vaults, and job folders made the way the pipeline leaves them."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.out = tmp / "out"
        self.vaults = {"Work": tmp / "Work", "Personal": tmp / "Personal"}
        for v in self.vaults.values():
            v.mkdir(parents=True)
        self.settings = json.loads(json.dumps(config.DEFAULTS))
        self.settings.update({"output_root": str(self.out), "vaults": {k: str(v) for k, v in self.vaults.items()},
                              "picker_timeout_s": 20})
        self.root = config.recordings_dir(self.settings)

    def job(self, name: str, *, vault: str = jobs.LOCAL, status: str = "done", kind: str = "meeting",
            draft: str | None = "# Meeting transcript\n\n- [00:01] **Alex Morgan** " + SECRET + "\n",
            lines: list | None = None, final: str | None = None, notes: str | None = None, seconds: float = 61.0,
            error: str | None = None, source: Path | None = None) -> Path:
        folder = self.root / name
        folder.mkdir()
        job = jobs.Job.new(kind, folder, vault, source=source)
        job.status, job.error = status, error
        if kind == "meeting":
            jobs.write_json_atomic(folder / jobs.RECORDING_FILE, {"status": "stopped", "seconds": seconds, "events": []})
        if draft is not None:
            (folder / "draft.md").write_text(draft, encoding="utf-8")
            job.outputs["draft"] = "draft.md"
        if lines is not None:
            jobs.write_json_atomic(folder / "speakers.json", {"owner": "Alex Morgan", "owner_matched": True,
                                                              "names": {}, "diarization": [], "lines": lines})
            job.outputs["speakers"] = "speakers.json"
        for key, text in (("transcript", final), ("notes", notes)):
            if text is not None:
                rel = f"Meetings/Kickoff {key} 31 Oct 2026.md"
                path = self.vaults[vault] / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                job.outputs[key] = rel
        job.save()
        return folder


class FakeScribe:
    """Stands in for the running App behind a real app.IpcServer: status, toggle (on its own thread, a moment
    later, as the app does) and enhance. Records every command it is handed."""

    def __init__(self, env: Env):
        self.env = env
        self.recording: Path | None = None
        self.elapsed = 0.0
        self.jobs: list[dict] = []
        self.commands: list[dict] = []
        self.ignore_toggle = False
        self.enhance_sent: int | None = None       # what "enhance" reports as sent; None: every folder it was given
        self.stop_seconds = 42.0
        self.n = 0
        self.lock = threading.Lock()

    def handle(self, commands: list[dict]) -> dict:
        reply: dict = {"ok": True}
        for c in commands:
            with self.lock:
                self.commands.append(c)
            kind = c.get("cmd")
            if kind == "status":
                reply["status"] = self.status()
            elif kind == "toggle":
                threading.Timer(0.05, self._toggle).start()
            elif kind == "enhance":
                ok = c.get("vault") in self.env.vaults
                reply["sent"] = (len(c.get("folders") or []) if ok else 0) if self.enhance_sent is None \
                    else self.enhance_sent
            else:
                return {"ok": False, "error": f"unknown command: {kind}"}
        return reply

    def status(self) -> dict:
        return {"running": True, "pid": 4242, "dictation": "ready", "recording": self.recording is not None,
                "recording_folder": str(self.recording) if self.recording else None,
                "elapsed": self.elapsed if self.recording else 0.0, "jobs": list(self.jobs)}

    def _toggle(self) -> None:
        if self.ignore_toggle:
            return
        if self.recording is None:
            self.n += 1
            folder = self.env.root / f"2026-10-31 09{self.n:02d}"
            folder.mkdir()
            jobs.write_json_atomic(folder / jobs.RECORDING_FILE, {"status": "recording", "seconds": 0.0})
            self.recording, self.elapsed = folder, 1.0
        else:
            jobs.write_json_atomic(self.recording / jobs.RECORDING_FILE,
                                   {"status": "stopped", "seconds": self.stop_seconds})
            self.recording = None

    def toggles(self) -> int:
        with self.lock:
            return sum(1 for c in self.commands if c.get("cmd") == "toggle")


class Case(unittest.TestCase):
    """A temp Env, a FakeScribe on a free port and a Scribe wired to both."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.env = Env(self.tmp)
        self.fake = FakeScribe(self.env)
        self.key_path = self.tmp / "ipc.key"
        self.key_path.write_bytes(b"k" * 64)
        self.port = free_port()
        self.ipc = app.IpcServer(self.fake.handle, self.key_path.read_bytes(), port=self.port)
        self.ipc.start()
        self.live_calls: list[dict] = []
        self.live_result: dict | Exception = {"duration": 30.0, "offset": 0.0, "segments": [], "lines": None,
                                              "speakers": "off", "languages": {}}
        self.scribe = Scribe(settings_fn=lambda: self.env.settings,
                             link=Link(self.key_path, self.port, wait_s=0.3, timeout=5.0),
                             live=types.SimpleNamespace(run=self._fake_live))

    def tearDown(self):
        self.ipc.close()
        self._tmp.cleanup()

    def _fake_live(self, folder, job, settings, *, recording, speakers=True, last_minutes=None):
        self.live_calls.append({"folder": Path(folder), "job": job, "recording": recording, "speakers": speakers,
                                "last_minutes": last_minutes})
        if isinstance(self.live_result, Exception):
            raise self.live_result
        return self.live_result

    def down(self) -> Scribe:
        """A Scribe whose app is not running (nothing listens on its port)."""
        return Scribe(settings_fn=lambda: self.env.settings,
                      link=Link(self.key_path, free_port(), wait_s=0.1, timeout=1.0))

    def call(self, name: str, args: dict | None = None, scribe: Scribe | None = None) -> tuple[bool, str]:
        """One tool call through the SDK's client, in process: (is_error, text)."""
        from mcp.shared.memory import create_connected_server_and_client_session as connect

        server = mcp_server.build_server(scribe or self.scribe)
        out = {}

        async def go():
            async with connect(server) as client:
                r = await client.call_tool(name, args or {})
                out["r"] = r

        anyio.run(go)
        r = out["r"]
        return bool(r.isError), "\n".join(c.text for c in r.content if getattr(c, "type", "") == "text")


# ---------------------------------------------------------------- text helpers

class TestText(unittest.TestCase):
    def test_render_lines(self):
        lines = [{"speaker": "Alex Morgan", "start": 1.2, "text": " Goeie  môre "},
                 {"speaker": "", "start": 3700.0, "text": "unlabelled"},
                 {"speaker": "Speaker 2", "start": 5.0, "text": "   "}]
        self.assertEqual(mcp_server.render_lines(lines),
                         "[00:01] Alex Morgan: Goeie môre\n[1:01:40] unlabelled")
        self.assertEqual(mcp_server.render_lines(lines[:1], offset=60.0), "[01:01] Alex Morgan: Goeie môre")

    def test_page_walks_the_whole_text(self):
        text = "\n".join(f"[00:{i % 60:02d}] line {i} " + "x" * 50 for i in range(200))
        self.assertEqual(mcp_server.page("short", 0, 5000), "short")
        parts, offset = [], 0
        for _ in range(100):
            window = mcp_server.page(text, offset, 1000)
            body, _, last = window.rpartition("\n")
            if "call again with offset=" not in last:
                parts.append(window)
                break
            self.assertLessEqual(len(body), 1000)
            self.assertTrue(body.endswith("x"))                 # cut at a line end
            parts.append(body + "\n")
            offset = int(last.rsplit("=", 1)[1].rstrip("]"))
        self.assertEqual("".join(parts), text)
        self.assertIn("[end", mcp_server.page(text, len(text) + 5, 1000))

    def test_page_clamps_the_window(self):
        text = "a" * 5000
        self.assertEqual(len(mcp_server.page(text, 0, 1).split("\n")[0]), mcp_server.MIN_CHARS)

    def test_tail_keeps_the_newest_lines(self):
        text = "\n".join(f"line {i}" for i in range(1000))
        out = mcp_server.tail(text, 100)
        self.assertTrue(out.endswith("line 999"))
        self.assertTrue(out.startswith("["))
        self.assertNotIn("line 1\n", out)
        self.assertEqual(mcp_server.tail("short", 100), "short")


# ---------------------------------------------------------------- copying a track the recorder is writing

@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not on PATH")
class TestSnapshot(unittest.TestCase):
    def test_growing_flac_is_copied_and_the_writer_carries_on(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            writer = sf.SoundFile(str(d / "mic.flac"), "w", SR, 1, format="FLAC", subtype="PCM_16")
            writer.write(tone(3.0))
            writer.flush()
            copy_dir = d / "copy"
            copy_dir.mkdir()
            self.assertGreater(mcp_server.snapshot(d / "mic.flac", copy_dir / "mic.flac"), 1000)
            mixed = jobs._mod("audio").mix_tracks(copy_dir)
            self.assertGreater(sf.info(str(mixed)).duration, 2.5)
            writer.write(tone(2.0))                 # the recorder is not disturbed
            writer.close()
            self.assertEqual(sf.info(str(d / "mic.flac")).frames, 5 * SR)


# ---------------------------------------------------------------- live transcription (fake Whisper and labels)

class LiveCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.folder = self.tmp / "2026-10-31 0900"
        self.folder.mkdir()
        self.seen: dict = {}
        self.label_doc: dict | Exception = {"owner_matched": True, "lines": [
            {"speaker": "Alex Morgan", "start": 0.5, "end": 1.0, "text": SECRET}]}
        self.live = Live(transcribe=self.fake_transcribe, label=self.fake_label, app_dir=self.tmp / "app")
        self.settings = json.loads(json.dumps(config.DEFAULTS))

    def tearDown(self):
        self._tmp.cleanup()

    def fake_transcribe(self, audio_path, out_path, model, settings, workdir):
        self.seen.update(audio=Path(audio_path), duration=sf.info(str(audio_path)).duration, model=model,
                         workdir=Path(workdir))
        doc = {"segments": [{"start": 0.5, "end": 1.0, "text": SECRET}], "languages": {"af": 1.0, "en": 0.0}}
        Path(out_path).write_text(json.dumps(doc), encoding="utf-8")
        return doc

    def fake_label(self, audio_path, whisper_path, out_path, settings, app_dir, workdir):
        self.seen["labelled"] = Path(audio_path)
        if isinstance(self.label_doc, Exception):
            raise self.label_doc
        return self.label_doc

    def write(self, name: str, seconds: float) -> Path:
        path = self.folder / name
        sf.write(str(path), tone(seconds), SR, format="FLAC", subtype="PCM_16")
        return path


class TestLive(LiveCase):
    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not on PATH")
    def test_recording_in_progress_is_copied_not_touched(self):
        mic = sf.SoundFile(str(self.folder / "mic.flac"), "w", SR, 1, format="FLAC", subtype="PCM_16")
        system = sf.SoundFile(str(self.folder / "system.flac"), "w", SR, 1, format="FLAC", subtype="PCM_16")
        try:
            mic.write(tone(4.0))
            system.write(tone(4.0, 660.0))
            mic.flush()
            system.flush()
            before = sorted(p.name for p in self.folder.iterdir())
            r = self.live.run(self.folder, None, self.settings, recording=True)
            self.assertEqual(sorted(p.name for p in self.folder.iterdir()), before)
            self.assertNotEqual(self.seen["audio"].parent, self.folder)
            self.assertGreater(self.seen["duration"], 3.0)
            self.assertFalse(self.seen["workdir"].exists())          # the temp folder is gone
            mic.write(tone(1.0))
            system.write(tone(1.0))
        finally:
            mic.close()
            system.close()
        self.assertEqual(sf.info(str(self.folder / "mic.flac")).frames, 5 * SR)
        self.assertEqual(self.seen["model"], "large-v3-turbo")
        self.assertEqual(r["lines"][0]["speaker"], "Alex Morgan")
        self.assertEqual(r["speakers"], "owner matched")

    def test_the_old_gpu_mutex_holds_up_nothing(self):
        # until 31 Oct 2026 a live transcript queued on a GPU mutex: this process's own Local\ScribeGPU-<pid>
        self.write("mix.flac", 2.0)
        job = jobs.Job.new("meeting", self.folder, jobs.LOCAL)
        job.outputs["audio"] = "mix.flac"
        live = Live(transcribe=self.fake_transcribe, label=self.fake_label, app_dir=self.tmp / "app")
        release, t = hold_named_mutex(f"Local\\ScribeGPU-{os.getpid()}")
        try:
            began = time.monotonic()
            r = live.run(self.folder, job, self.settings, recording=False)
            self.assertLess(time.monotonic() - began, 5.0)
        finally:
            release.set()
            t.join(5)
        self.assertEqual(r["segments"][0]["text"], SECRET)

    def test_finished_job_uses_its_own_audio(self):
        audio = self.write("mix.flac", 2.0)
        job = jobs.Job.new("meeting", self.folder, jobs.LOCAL)
        job.outputs["audio"] = "mix.flac"
        self.live.run(self.folder, job, self.settings, recording=False)
        self.assertEqual(self.seen["audio"], audio)

    def test_last_minutes_transcribes_only_the_end(self):
        self.write("mix.flac", 90.0)
        job = jobs.Job.new("meeting", self.folder, jobs.LOCAL)
        job.outputs["audio"] = "mix.flac"
        r = self.live.run(self.folder, job, self.settings, recording=False, last_minutes=0.5)
        self.assertAlmostEqual(self.seen["duration"], 30.0, delta=0.1)
        self.assertAlmostEqual(r["offset"], 60.0, delta=0.1)
        self.assertAlmostEqual(r["duration"], 30.0, delta=0.1)
        r = self.live.run(self.folder, job, self.settings, recording=False, last_minutes=5)
        self.assertEqual(r["offset"], 0.0)

    def test_labels_failing_leaves_whisper_lines(self):
        self.write("mix.flac", 2.0)
        job = jobs.Job.new("meeting", self.folder, jobs.LOCAL)
        job.outputs["audio"] = "mix.flac"
        self.label_doc = FileNotFoundError("segmentation model missing: x")
        r = self.live.run(self.folder, job, self.settings, recording=False)
        self.assertIsNone(r["lines"])
        self.assertEqual(r["segments"][0]["text"], SECRET)
        self.assertIn("unavailable", r["speakers"])
        self.assertIn("segmentation model missing", r["speakers"])

    def test_speakers_off(self):
        self.write("mix.flac", 2.0)
        job = jobs.Job.new("meeting", self.folder, jobs.LOCAL)
        job.outputs["audio"] = "mix.flac"
        r = self.live.run(self.folder, job, self.settings, recording=False, speakers=False)
        self.assertNotIn("labelled", self.seen)
        self.assertEqual(r["speakers"], "off")

    def test_nothing_recorded_yet(self):
        with self.assertRaises(NothingYet):
            self.live.run(self.folder, None, self.settings, recording=True)
        (self.folder / "mic.flac").write_bytes(b"")
        with self.assertRaises(NothingYet):
            self.live.run(self.folder, None, self.settings, recording=True)

    def test_file_job_before_its_audio_step(self):
        src = self.tmp / "call.wav"
        sf.write(str(src), tone(2.0), SR)
        job = jobs.Job.new("file", self.folder, jobs.LOCAL, source=src)
        def extract_audio(path, out):
            sf.write(str(out), tone(2.0), SR, format="FLAC", subtype="PCM_16")
            return out

        real = jobs._mod
        fake_video = types.SimpleNamespace(extract_audio=extract_audio)
        with mock.patch.object(jobs, "_mod", side_effect=lambda n: fake_video if n == "video" else real(n)):
            self.live.run(self.folder, job, self.settings, recording=False)
        self.assertEqual(self.seen["audio"].name, "audio.flac")
        self.assertNotEqual(self.seen["audio"].parent, self.folder)


# ---------------------------------------------------------------- the tools, in process through the SDK's client

class TestTools(Case):
    def test_tool_list_and_annotations(self):
        from mcp.shared.memory import create_connected_server_and_client_session as connect

        out = {}

        async def go():
            async with connect(mcp_server.build_server(self.scribe)) as client:
                out["tools"] = (await client.list_tools()).tools

        anyio.run(go)
        tools = {t.name: t for t in out["tools"]}
        self.assertEqual(set(tools), {"status", "recent_recordings", "transcript", "live_transcript",
                                      "start_notetaker", "stop_notetaker", "send_to_vault"})
        for name in ("status", "recent_recordings", "transcript", "live_transcript"):
            self.assertTrue(tools[name].annotations.readOnlyHint, name)
            self.assertTrue(tools[name].description.rstrip().endswith("Read-only."), name)
        for name in ("start_notetaker", "stop_notetaker", "send_to_vault"):
            self.assertFalse(tools[name].annotations.readOnlyHint, name)
            self.assertTrue(tools[name].description.startswith("Acts:"), name)
        self.assertEqual(tools["transcript"].inputSchema["properties"]["kind"]["enum"],
                         ["final", "draft", "notes", "speakers"])

    def test_status(self):
        self.fake.jobs = [{"id": "2026-10-31 0832", "kind": "meeting", "vault": "Personal", "status": "transcribing",
                           "waiting": "gpu", "error": None}]
        err, text = self.call("status")
        self.assertFalse(err)
        doc = json.loads(text)
        self.assertEqual(doc["running"], True)
        self.assertFalse(doc["recording"])
        self.assertEqual(doc["jobs"][0]["status"], "transcribing")
        self.assertEqual(doc["jobs"][0]["vault"], "Personal")
        self.assertEqual(doc["jobs"][0]["waiting"], "gpu")          # backing off a GPU out of memory
        self.fake.recording, self.fake.elapsed = self.env.root / "2026-10-31 0900", 754.2
        doc = json.loads(self.call("status")[1])
        self.assertEqual((doc["recording"], doc["recording_id"], doc["elapsed"]), (True, "2026-10-31 0900", "12:34"))

    def test_status_when_scribe_is_not_running(self):
        err, text = self.call("status", scribe=self.down())
        self.assertFalse(err)
        self.assertEqual(json.loads(text), {"running": False})

    def test_recent_recordings(self):
        self.env.job("2026-10-30 1400", vault="Personal", final="final text", notes="notes text", seconds=3725.0,
                     lines=[{"speaker": "Alex Morgan", "start": 0.0, "text": "hi"}])
        self.env.job("2026-10-30 1500", status="failed", error="transcribing: boom", draft=None)
        self.env.job("2026-10-31 0800")
        interrupted = self.env.root / "2026-10-31 0810"
        interrupted.mkdir()
        jobs.write_json_atomic(interrupted / jobs.RECORDING_FILE, {"status": "recording", "seconds": 12.0})
        current = self.env.root / "2026-10-31 0900"
        current.mkdir()
        jobs.write_json_atomic(current / jobs.RECORDING_FILE, {"status": "recording", "seconds": 30.0})
        self.fake.recording = current
        err, text = self.call("recent_recordings", {"limit": 10})
        self.assertFalse(err)
        rows = json.loads(text)
        self.assertEqual([r["id"] for r in rows],
                         ["2026-10-31 0900", "2026-10-31 0810", "2026-10-31 0800", "2026-10-30 1500",
                          "2026-10-30 1400"])
        by = {r["id"]: r for r in rows}
        self.assertEqual(by["2026-10-31 0900"]["status"], "recording")
        self.assertEqual(by["2026-10-31 0810"]["status"], "interrupted")
        self.assertEqual(by["2026-10-30 1400"]["outputs"], ["final", "draft", "notes", "speakers"])
        self.assertEqual(by["2026-10-30 1400"]["length"], "1:02:05")
        self.assertEqual(by["2026-10-30 1400"]["time"], "2026-10-30 14:00")
        self.assertEqual(by["2026-10-30 1400"]["vault"], "Personal")
        self.assertEqual(by["2026-10-31 0800"]["outputs"], ["draft"])
        self.assertEqual(by["2026-10-30 1500"]["error"], "transcribing: boom")
        self.assertEqual(len(json.loads(self.call("recent_recordings", {"limit": 2})[1])), 2)

    def test_recent_recordings_without_scribe(self):
        folder = self.env.root / "2026-10-31 0900"
        folder.mkdir()
        jobs.write_json_atomic(folder / jobs.RECORDING_FILE, {"status": "recording", "seconds": 30.0})
        rows = json.loads(self.call("recent_recordings", scribe=self.down())[1])
        self.assertEqual(rows[0]["status"], "interrupted")      # nothing records while Scribe is down

    def test_transcript_kinds(self):
        lines = [{"speaker": "Alex Morgan", "start": 1.0, "text": "Goeie môre"},
                 {"speaker": "Speaker 2", "start": 4.0, "text": "Morning"}]
        self.env.job("2026-10-30 1400", vault="Personal", final="FINAL " + SECRET, notes="NOTES", lines=lines)
        self.assertEqual(self.call("transcript", {"id": "2026-10-30 1400"}), (False, "FINAL " + SECRET))
        self.assertEqual(self.call("transcript", {"id": "2026-10-30 1400", "kind": "notes"}), (False, "NOTES"))
        self.assertIn(SECRET, self.call("transcript", {"id": "2026-10-30 1400", "kind": "draft"})[1])
        self.assertEqual(self.call("transcript", {"id": "2026-10-30 1400", "kind": "speakers"}),
                         (False, "[00:01] Alex Morgan: Goeie môre\n[00:04] Speaker 2: Morning"))

    def test_transcript_missing_outputs(self):
        self.env.job("2026-10-31 0800")
        err, text = self.call("transcript", {"id": "2026-10-31 0800"})
        self.assertFalse(err)
        self.assertIn("No final", text)
        self.assertIn("kind draft", text)
        self.env.job("2026-10-31 0805", vault="Personal", status="transcribing", draft=None)
        self.assertIn("No draft yet", self.call("transcript", {"id": "2026-10-31 0805", "kind": "draft"})[1])
        rec = self.env.root / "2026-10-31 0900"
        rec.mkdir()
        self.assertIn("no job yet", self.call("transcript", {"id": "2026-10-31 0900"})[1])

    def test_transcript_moved_out_of_its_shelf(self):
        lines = [{"speaker": "Alex Morgan", "start": 1.0, "text": "Goeie môre"}]
        self.env.job("2026-10-31 0832", vault="Personal", final="FINAL", notes="NOTES", lines=lines)
        shelf = self.env.vaults["Personal"] / "Meetings"
        for p in shelf.iterdir():
            p.unlink()                                      # the user moved them off the shelf
        err, text = self.call("transcript", {"id": "2026-10-31 0832"})
        self.assertFalse(err)
        self.assertEqual(text, "The final transcript of 2026-10-31 0832 was shelved at "
                               "Personal\\Meetings\\Kickoff transcript 31 Oct 2026.md and is no longer there: it was "
                               "moved or deleted in the vault. Scribe still holds its draft and speakers (kind draft).")
        text = self.call("transcript", {"id": "2026-10-31 0832", "kind": "notes"})[1]
        self.assertTrue(text.startswith("The notes of 2026-10-31 0832 were shelved at Personal\\Meetings\\"), text)
        self.assertIn("are no longer there: they were moved or deleted", text)

    def test_transcript_reads_nothing_outside(self):
        secret = self.tmp / "secret.md"
        secret.write_text(SECRET, encoding="utf-8")
        folder = self.env.job("2026-10-30 1400", vault="Personal", final="FINAL")
        job = jobs.Job.load(folder)
        job.outputs["transcript"] = "../secret.md"           # a vault-relative path pointing out of the vault
        job.save()
        err, text = self.call("transcript", {"id": "2026-10-30 1400"})
        self.assertNotIn(SECRET, text)
        self.assertIn("No final", text)
        for bad in ("..", "../out", "..\\secret.md", "C:\\Windows", "a/b", ""):
            err, text = self.call("transcript", {"id": bad, "kind": "draft"})
            self.assertTrue(err, bad)
            self.assertNotIn(SECRET, text)
        err, text = self.call("transcript", {"id": "2026-01-01 0000"})
        self.assertTrue(err)
        self.assertIn("recent_recordings", text)

    def test_transcript_comes_in_windows(self):
        text = "\n".join(f"[00:{i % 60:02d}] Alex Morgan: line {i} " + "y" * 60 for i in range(300))
        self.env.job("2026-10-30 1400", vault="Personal", final=text)
        first = self.call("transcript", {"id": "2026-10-30 1400", "max_chars": 2000})[1]
        last = first.rsplit("\n", 1)[1]
        self.assertIn("call again with offset=", last)
        offset = int(last.rsplit("=", 1)[1].rstrip("]"))
        second = self.call("transcript", {"id": "2026-10-30 1400", "offset": offset, "max_chars": 2000})[1]
        self.assertTrue(text[offset:].startswith(second.rsplit("\n", 1)[0]))

    def test_logs_hold_no_spoken_text(self):
        lines = [{"speaker": "Alex Morgan", "start": 1.0, "text": SECRET}]
        self.env.job("2026-10-30 1400", vault="Personal", final=SECRET, notes=SECRET, lines=lines)
        self.live_result = {"duration": 3.0, "offset": 0.0, "segments": [], "lines": lines, "speakers": "owner matched",
                            "languages": {"af": 3.0}}
        logger = logging.getLogger("scribe")
        records: list[logging.LogRecord] = []
        handler = logging.Handler(logging.DEBUG)
        handler.emit = records.append
        logger.addHandler(handler)
        old = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            outputs = [self.call("transcript", {"id": "2026-10-30 1400", "kind": k})[1] for k in mcp_server.KINDS]
            outputs.append(self.call("live_transcript", {"id": "2026-10-30 1400"})[1])
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old)
        self.assertTrue(all(SECRET in o for o in outputs))
        self.assertTrue(records)
        for r in records:
            self.assertNotIn(SECRET, r.getMessage())
            self.assertNotIn("môre", r.getMessage())

    def test_live_transcript_current(self):
        err, text = self.call("live_transcript")
        self.assertFalse(err)
        self.assertIn("not recording", text)
        self.assertEqual(self.live_calls, [])
        current = self.env.root / "2026-10-31 0900"
        current.mkdir()
        self.fake.recording = current
        self.live_result = {"duration": 754.0, "offset": 0.0, "segments": [], "speakers": "owner matched",
                            "languages": {"af": 300.0, "en": 400.0},
                            "lines": [{"speaker": "Alex Morgan", "start": 2.0, "text": "Ja"},
                                      {"speaker": "Speaker 2", "start": 5.0, "text": "Hello"}]}
        err, text = self.call("live_transcript", {"id": "current", "last_minutes": 5})
        self.assertFalse(err)
        call = self.live_calls[-1]
        self.assertEqual(call["folder"], current)
        self.assertTrue(call["recording"])
        self.assertEqual(call["last_minutes"], 5)
        head, _, body = text.partition("\n\n")
        self.assertIn("2026-10-31 0900 · recording · 12:34 so far", head)
        self.assertIn("speakers: owner matched", head)
        self.assertEqual(body, "[00:02] Alex Morgan: Ja\n[00:05] Speaker 2: Hello")

    def test_live_transcript_of_a_finished_job(self):
        self.env.job("2026-10-30 1400", vault="Personal")
        self.live_result = {"duration": 60.0, "offset": 600.0, "lines": None, "speakers": "off", "languages": {},
                            "segments": [{"start": 1.0, "text": "unlabelled"}]}
        text = self.call("live_transcript", {"id": "2026-10-30 1400", "speakers": False})[1]
        self.assertFalse(self.live_calls[-1]["recording"])
        self.assertFalse(self.live_calls[-1]["speakers"])
        self.assertIn("done · 11:00 so far, last 01:00", text)
        self.assertTrue(text.endswith("[10:01] unlabelled"))

    def test_live_transcript_nothing_yet(self):
        current = self.env.root / "2026-10-31 0900"
        current.mkdir()
        self.fake.recording = current
        self.live_result = NothingYet("no audio recorded yet")
        self.assertEqual(self.call("live_transcript")[1], "2026-10-31 0900: no audio recorded yet.")

    def test_start_notetaker(self):
        err, text = self.call("start_notetaker")
        self.assertFalse(err)
        self.assertEqual(text, "Recording: 2026-10-31 0901. stop_notetaker ends it.")
        self.assertEqual(self.fake.toggles(), 1)
        err, text = self.call("start_notetaker")
        self.assertIn("Already recording: 2026-10-31 0901", text)
        self.assertEqual(self.fake.toggles(), 1)                # refused: no second toggle

    def test_start_notetaker_toggle_not_taken(self):
        self.fake.ignore_toggle = True
        with mock.patch.object(mcp_server, "TOGGLE_WAIT_S", 0.5):
            text = self.call("start_notetaker")[1]
        self.assertIn("did not start recording", text)

    def test_stop_notetaker(self):
        err, text = self.call("stop_notetaker")
        self.assertEqual(text, "Not recording. Nothing changed.")
        self.assertEqual(self.fake.toggles(), 0)
        self.call("start_notetaker")
        self.fake.stop_seconds = 754.0
        text = self.call("stop_notetaker")[1]
        self.assertTrue(text.startswith("Stopped: 2026-10-31 0901, 12:34."), text)
        self.assertIn("vault picker", text)
        self.assertIn("20 s", text)
        self.assertEqual(self.fake.toggles(), 2)
        self.assertIsNone(self.fake.recording)

    def test_stop_notetaker_too_short(self):
        self.call("start_notetaker")
        self.fake.stop_seconds = 3.0
        self.assertIn("Too short", self.call("stop_notetaker")[1])

    def test_notetaker_when_scribe_is_down(self):
        self.assertEqual(self.call("start_notetaker", scribe=self.down())[1], "Scribe is not running.")
        self.assertEqual(self.call("stop_notetaker", scribe=self.down())[1], "Scribe is not running.")

    def test_send_to_vault(self):
        folder = self.env.job("2026-10-31 0800")
        err, text = self.call("send_to_vault", {"id": "2026-10-31 0800", "vault": "Personal"})
        self.assertFalse(err)
        self.assertTrue(text.startswith("Queued: 2026-10-31 0800 to Personal."), text)
        sent = [c for c in self.fake.commands if c.get("cmd") == "enhance"]
        self.assertEqual(sent, [{"cmd": "enhance", "folders": [str(folder)], "vault": "Personal"}])

    def test_send_to_vault_that_scribe_did_not_queue(self):
        # app.send_jobs answers {"ok": true, "sent": 0} when JobQueue.send_to_vault raised (job.json held, say)
        self.env.job("2026-10-31 0800")
        self.fake.enhance_sent = 0
        err, text = self.call("send_to_vault", {"id": "2026-10-31 0800", "vault": "Personal"})
        self.assertFalse(err)
        self.assertTrue(text.startswith("Scribe did not queue 2026-10-31 0800"), text)
        self.assertNotIn("Queued", text)

    def test_send_to_vault_refusals(self):
        self.env.job("2026-10-30 1400", vault="Personal")
        self.env.job("2026-10-31 0800", status="labelling")
        self.env.job("2026-10-31 0805")
        self.assertIn("already went to Personal",
                      self.call("send_to_vault", {"id": "2026-10-30 1400", "vault": "Work"})[1])
        self.assertIn("not finished (status labelling)",
                      self.call("send_to_vault", {"id": "2026-10-31 0800", "vault": "Personal"})[1])
        err, text = self.call("send_to_vault", {"id": "2026-10-31 0805", "vault": "Local"})
        self.assertTrue(err)
        self.assertIn("unknown vault", text)
        self.assertEqual(self.call("send_to_vault", {"id": "2026-10-31 0805", "vault": "Personal"}, scribe=self.down())[1],
                         "Scribe is not running; nothing was sent.")
        self.assertEqual([c for c in self.fake.commands if c.get("cmd") == "enhance"], [])

    def test_send_to_vault_with_no_vaults_set(self):
        self.env.job("2026-10-31 0805")
        self.env.settings["vaults"] = {}
        err, text = self.call("send_to_vault", {"id": "2026-10-31 0805", "vault": "Work"})
        self.assertTrue(err)
        self.assertIn("no vaults are set", text)
        self.assertEqual([c for c in self.fake.commands if c.get("cmd") == "enhance"], [])


class TestLogging(unittest.TestCase):
    def test_the_sdks_root_warnings_stay_out_of_mcp_log(self):
        root, scribe_log = logging.getLogger(), logging.getLogger("scribe")
        saved = root.handlers[:], root.level, scribe_log.level
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(config, "logs_dir", return_value=Path(d)):
                path = mcp_server.setup_logging()
            handler = next(h for h in root.handlers if h not in saved[0])
            try:
                logging.warning("Failed to validate request: 31 validation errors for ClientRequest\nPingRequest.method\n"
                                "  Input should be 'ping' [type=literal_error, input_value='server/discover']")
                logging.error("a root error is kept")
                logging.getLogger("mcp.server.lowlevel.server").warning("the SDK's named warnings are kept")
                logging.getLogger("scribe.mcp").info("Scribe's own lines are kept")
            finally:
                root.removeHandler(handler)
                handler.close()
                root.setLevel(saved[1])
                scribe_log.setLevel(saved[2])
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("Failed to validate request", text)
        self.assertNotIn("server/discover", text)
        for kept in ("a root error is kept", "the SDK's named warnings are kept", "Scribe's own lines are kept"):
            self.assertIn(kept, text)


class TestBuild(unittest.TestCase):
    def test_root_logger_is_left_as_it_was(self):
        """FastMCP puts a stderr handler on a bare root logger; Claude Code would keep what it writes."""
        root = logging.getLogger()
        saved = root.handlers[:], root.level
        root.handlers[:] = []
        try:
            mcp_server.build_server(Scribe(settings_fn=dict, link=Link(port=free_port(), wait_s=0.0)))
            self.assertEqual((root.handlers, root.level), ([], saved[1]))
        finally:
            root.handlers[:] = saved[0]
            root.setLevel(saved[1])


class TestNoAnswer(Case):
    def test_a_command_without_an_answer_is_not_resent(self):
        self.ipc.close()
        port = free_port()
        hits = []

        def silent(commands):
            hits.append(commands)
            time.sleep(3)
            return {"ok": True}

        server = app.IpcServer(silent, self.key_path.read_bytes(), port=port)
        server.start()
        try:
            scribe = Scribe(settings_fn=lambda: self.env.settings,
                            link=Link(self.key_path, port, wait_s=0.3, timeout=0.5))
            text = self.call("status", scribe=scribe)[1]
        finally:
            server.close()
        self.assertIn("did not answer", text)
        self.assertEqual(len(hits), 1)


# ---------------------------------------------------------------- the transcription child

class TestChild(unittest.TestCase):
    def test_child_writes_whisper_json(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            seen = {}

            def transcribe_file(audio, model, languages, vocabulary, progress=None, on_wait=None):
                seen.update(audio=audio, model=model, languages=languages, vocabulary=vocabulary)
                progress(0.5)
                on_wait(True)                       # a GPU wait counts as a sign of life too
                on_wait(False)
                progress(1.0)
                return {"segments": [{"start": 0.0, "text": SECRET}]}

            req = d / "req.json"
            req.write_text(json.dumps({"audio": str(d / "a.flac"), "out": str(d / "w.json"), "model": "large-v3-turbo",
                                       "languages": ["en", "af"], "vocabulary": ["Acme"],
                                       "progress": str(d / "progress.txt")}), encoding="utf-8")
            fake = types.SimpleNamespace(transcribe_file=transcribe_file)
            with mock.patch.object(jobs, "_mod", return_value=fake), mock.patch("logging.basicConfig"):
                self.assertEqual(mcp_server.main(["transcribe", "--request", str(req)]), 0)
            self.assertEqual(json.loads((d / "w.json").read_text(encoding="utf-8"))["segments"][0]["text"], SECRET)
            self.assertEqual((seen["audio"], seen["model"], seen["languages"]),
                             (d / "a.flac", "large-v3-turbo", ["en", "af"]))
            self.assertEqual((d / "progress.txt").read_text(encoding="utf-8"), ".\n" * 4)

    def test_child_is_started_without_a_window(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            settings = json.loads(json.dumps(config.DEFAULTS))
            settings.update(vocabulary=["Acme", "Kira"], languages=["en", "af"])
            seen = {}

            def fake_popen(cmd, **kw):
                seen.update(cmd=cmd, **kw)
                seen["req"] = json.loads(Path(cmd[cmd.index("--request") + 1]).read_text(encoding="utf-8"))
                Path(seen["req"]["out"]).write_text(json.dumps({"segments": [{"text": SECRET}]}), encoding="utf-8")
                return types.SimpleNamespace(wait=lambda timeout=None: 0, poll=lambda: 0, kill=lambda: None)

            with mock.patch.object(mcp_server.subprocess, "Popen", fake_popen):
                doc = mcp_server.transcribe_child(FIXTURES / "short_dictation.wav", d / "w.json", "large-v3-turbo",
                                                  settings, d)
        self.assertEqual(doc["segments"][0]["text"], SECRET)
        self.assertEqual(seen["creationflags"], subprocess.CREATE_NO_WINDOW)
        self.assertEqual(seen["stdin"], subprocess.DEVNULL)
        self.assertEqual(seen["cmd"][1:4], ["-m", "scribe.mcp_server", "transcribe"])
        self.assertEqual(seen["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual((seen["req"]["model"], seen["req"]["vocabulary"], seen["req"]["languages"]),
                         ("large-v3-turbo", ["Acme", "Kira"], ["en", "af"]))
        self.assertEqual(Path(seen["req"]["progress"]).parent, d)      # beats go to the temp folder only
        self.assertNotIn("timeout", seen)            # no fixed limit: watch_child ends a child that stops

    def test_labels_as_the_pipeline_makes_them(self):
        seen = {}
        settings = json.loads(json.dumps(config.DEFAULTS))
        settings["owner_name"] = "Alex Morgan"
        settings["meetings"].update(cluster_threshold=0.7, owner_match=0.65)

        def label_job(*args, **kw):
            seen.update(args=args, kw=kw)
            return {"lines": []}

        def seg(models):
            seen["models"] = models
            if seen.get("missing"):
                raise FileNotFoundError("segmentation model missing")

        fake = types.SimpleNamespace(segmentation_model=seg, embedding_model=lambda m: m, label_job=label_job)
        app_dir, work = Path("C:/x/Scribe"), Path("C:/x/work")
        with mock.patch.object(jobs, "_mod", return_value=fake):
            mcp_server.label_child(Path("a.flac"), Path("w.json"), Path("s.json"), settings, app_dir, work)
            self.assertEqual(seen["args"], (Path("a.flac"), Path("w.json"), Path("s.json"), app_dir / "models"))
            self.assertEqual(seen["kw"], {"threshold": 0.7, "owner_name": "Alex Morgan", "owner_match": 0.65,
                                          "voiceprint_path": app_dir / "voiceprint.json",
                                          "log_path": work / "labelling.log"})
            seen.clear()
            seen["missing"] = True
            with self.assertRaises(FileNotFoundError):
                mcp_server.label_child(Path("a.flac"), Path("w.json"), Path("s.json"), settings, app_dir, work)
            self.assertNotIn("args", seen)

    def test_child_process_reports_its_failure(self):
        """A real child, with a model name no cache holds: it fails before CUDA or any download, and says why."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            settings = json.loads(json.dumps(config.DEFAULTS))
            with self.assertRaises(RuntimeError) as cm:
                mcp_server.transcribe_child(FIXTURES / "short_dictation.wav", d / "w.json", "no-such-model-xyz",
                                            settings, d)
            self.assertIn("model not cached: no-such-model-xyz", str(cm.exception))
            self.assertFalse((d / "w.json").exists())


class TestChildWatch(unittest.TestCase):
    """A real child process in place of the transcription child: ended once it stops making progress, never while
    it keeps going, however long that takes."""

    CHILD = (
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "req = json.loads(Path(sys.argv[sys.argv.index('--request') + 1]).read_text(encoding='utf-8'))\n"
        "mode = req['vocabulary'][0]\n"
        "if mode == 'hang':\n"
        "    Path(req['out']).with_name('pid.txt').write_text(str(os.getpid()))\n"
        "    time.sleep(600)\n"
        "for _ in range(int(mode)):\n"
        "    with open(req['progress'], 'a') as fh:\n"
        "        fh.write('.\\n')\n"
        "    time.sleep(0.2)\n"
        "Path(req['out']).write_text(json.dumps({'segments': []}), encoding='utf-8')\n"
    )

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        mods = self.tmp / "mods"
        mods.mkdir()
        (mods / "fake_child.py").write_text(self.CHILD, encoding="utf-8")
        for patcher in (mock.patch.object(mcp_server, "MODULE", "fake_child"),
                        mock.patch.dict(os.environ, {"PYTHONPATH": str(mods)}),
                        mock.patch.object(mcp_server, "CHILD_START_S", 1.0),
                        mock.patch.object(mcp_server, "CHILD_STALL_S", 1.0)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_child(self, mode: str) -> dict:
        settings = json.loads(json.dumps(config.DEFAULTS))
        settings["vocabulary"] = [mode]
        work = self.tmp / "work"
        work.mkdir(exist_ok=True)
        return mcp_server.transcribe_child(FIXTURES / "short_dictation.wav", work / "w.json", "large-v3-turbo",
                                           settings, work)

    def test_a_quiet_child_is_ended(self):
        import psutil

        began = time.monotonic()
        with self.assertRaises(RuntimeError) as cm:
            self.run_child("hang")
        self.assertIn("no progress for 1 s", str(cm.exception))
        self.assertLess(time.monotonic() - began, 20.0)
        pid = int((self.tmp / "work" / "pid.txt").read_text())
        self.assertFalse(psutil.pid_exists(pid))

    def test_a_child_that_keeps_going_is_not_cut_off(self):
        began = time.monotonic()
        self.assertEqual(self.run_child("10"), {"segments": []})
        self.assertGreater(time.monotonic() - began, 2.0)          # longer than either limit


# ---------------------------------------------------------------- stdio: the server as Claude Code starts it

class TestStdio(Case):
    def setUp(self):
        super().setUp()
        self.local = self.tmp / "local"
        app_dir = self.local / "Scribe"
        app_dir.mkdir(parents=True)
        (app_dir / "settings.json").write_text(json.dumps(self.env.settings), encoding="utf-8")
        shutil.copyfile(self.key_path, app_dir / app.KEY_FILE)
        self.app_dir = app_dir

    def test_protocol_over_stdio(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self.env.job("2026-10-31 0800", draft="# draft\n\n- [00:01] **Alex Morgan** " + SECRET + "\n")
        entry = mcp_server.server_entry()                         # exactly what Claude Code is told to run
        params = StdioServerParameters(command=entry["command"], args=[*entry["args"], "--ipc-port", str(self.port)],
                                       env={**entry["env"], "SCRIBE_HOME": str(self.local / "Scribe")}, cwd=str(self.tmp))
        out = {}
        errlog = self.tmp / "stderr.txt"

        async def go():
            with open(errlog, "w", encoding="utf-8") as err:
                async with stdio_client(params, errlog=err) as (read, write):
                    async with ClientSession(read, write) as session:
                        init = await session.initialize()
                        out["name"] = init.serverInfo.name
                        out["tools"] = sorted(t.name for t in (await session.list_tools()).tools)
                        for key, name, args in (("status", "status", {}), ("recent", "recent_recordings", {}),
                                                ("draft", "transcript", {"id": "2026-10-31 0800", "kind": "draft"}),
                                                ("start", "start_notetaker", {})):
                            r = await session.call_tool(name, args)
                            out[key] = r.content[0].text

        anyio.run(go)
        self.assertEqual(out["name"], "scribe")
        self.assertEqual(len(out["tools"]), 7)
        self.assertTrue(json.loads(out["status"])["running"])
        self.assertEqual(json.loads(out["recent"])[0]["id"], "2026-10-31 0800")
        self.assertIn(SECRET, out["draft"])
        self.assertTrue(out["start"].startswith("Recording: "))
        self.assertTrue(wait_for(lambda: (self.app_dir / "logs" / "mcp.log").exists()))
        logged = (self.app_dir / "logs" / "mcp.log").read_text(encoding="utf-8")
        self.assertIn("transcript 2026-10-31 0800 draft", logged)
        self.assertNotIn(SECRET, logged)
        self.assertNotIn(SECRET, errlog.read_text(encoding="utf-8"))

    def test_claude_codes_server_discover_is_answered_quietly(self):
        # Claude Code 2.1.314 asks every server for "server/discover", which SDK 1.30.0 does not know
        import queue as queue_mod

        entry = mcp_server.server_entry()
        env = {**os.environ, **entry["env"], "SCRIBE_HOME": str(self.local / "Scribe")}
        proc = subprocess.Popen([entry["command"], *entry["args"], "--ipc-port", str(self.port)], cwd=str(self.tmp),
                                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        lines: queue_mod.Queue = queue_mod.Queue()
        threading.Thread(target=lambda: [lines.put(raw) for raw in proc.stdout], daemon=True).start()

        def send(msg: dict) -> None:
            proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
            proc.stdin.flush()

        def answer(msg_id: int) -> dict:
            while True:
                doc = json.loads(lines.get(timeout=60))
                if doc.get("id") == msg_id:
                    return doc

        try:
            send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                             "clientInfo": {"name": "test", "version": "0"}}})
            self.assertEqual(answer(1)["result"]["serverInfo"]["name"], "scribe")
            send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            send({"jsonrpc": "2.0", "id": 2, "method": "server/discover", "params": {}})
            self.assertIn("error", answer(2))
            send({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
            self.assertEqual(len(answer(3)["result"]["tools"]), 7)          # still serving
        finally:
            proc.stdin.close()
            proc.wait(30)
        log_file = self.app_dir / "logs" / "mcp.log"
        logged = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
        self.assertNotIn("Failed to validate request", logged)
        self.assertNotIn("validation error", logged)

    def test_stray_output_cannot_reach_the_protocol(self):
        code = ("import os, sys, anyio\n"
                "from scribe import mcp_server as m\n"
                "i, o = m.claim_stdio()\n"
                "print('stray print'); sys.stdout.write('stray write'); os.write(1, b'stray fd')\n"
                "anyio.run(o.write, 'protocol line\\n'); anyio.run(o.flush)\n")
        for exe in (Path(sys.executable), Path(sys.executable).with_name("pythonw.exe")):
            p = subprocess.run([str(exe), "-c", code], cwd=str(ROOT), capture_output=True, stdin=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr[-400:])
            self.assertEqual(p.stdout.decode("utf-8").strip(), "protocol line", exe.name)


# ---------------------------------------------------------------- registration for Claude Code

class TestRegister(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = self.tmp / ".claude.json"
        self.config.write_text(json.dumps({"numStartups": 3, "mcpServers": {"other": {"command": "x"}}}),
                               encoding="utf-8")
        self.calls: list[list[str]] = []
        self.fail = False
        self.write_entry = True

    def tearDown(self):
        self._tmp.cleanup()

    def fake_claude(self, argv, **kw):
        """Does to the temp config what `claude mcp add/remove --scope user` does to ~/.claude.json."""
        self.calls.append(list(argv))
        self.assertEqual(kw.get("creationflags"), subprocess.CREATE_NO_WINDOW)
        if self.fail:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")
        doc = json.loads(self.config.read_text(encoding="utf-8"))
        servers = doc.setdefault("mcpServers", {})
        if argv[1:3] == ["mcp", "remove"]:
            servers.pop(argv[-1], None)
        elif argv[1:3] == ["mcp", "add"] and self.write_entry:
            i = argv.index("--")
            env = dict(a.split("=", 1) for a in argv[argv.index("-e") + 1:i:2])
            servers[argv[5]] = {"type": "stdio", "command": argv[i + 1], "args": argv[i + 2:], "env": env}
        self.config.write_text(json.dumps(doc), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    def test_entry(self):
        e = mcp_server.server_entry()
        self.assertEqual(Path(e["command"]), Path(sys.executable).with_name("pythonw.exe"))
        self.assertEqual(e["args"], ["-m", "scribe.mcp_server"])
        self.assertEqual(Path(e["env"]["PYTHONPATH"]), ROOT)

    def test_added_once_then_unchanged(self):
        claude = self.tmp / "claude.exe"
        self.assertEqual(mcp_server.register(claude, self.config, run=self.fake_claude), "added")
        e = mcp_server.server_entry()
        self.assertEqual(self.calls, [[str(claude), "mcp", "add", "--scope", "user", "scribe",
                                       "-e", f"PYTHONPATH={e['env']['PYTHONPATH']}", "--", e["command"],
                                       "-m", "scribe.mcp_server"]])
        self.assertEqual(mcp_server.register(claude, self.config, run=self.fake_claude), "unchanged")
        self.assertEqual(len(self.calls), 1)
        doc = json.loads(self.config.read_text(encoding="utf-8"))
        self.assertEqual(doc["mcpServers"]["other"], {"command": "x"})
        self.assertEqual(doc["numStartups"], 3)

    def test_an_older_entry_is_replaced(self):
        doc = json.loads(self.config.read_text(encoding="utf-8"))
        doc["mcpServers"]["scribe"] = {"type": "stdio", "command": "C:\\old\\python.exe", "args": ["-m", "x"]}
        self.config.write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(mcp_server.register(self.tmp / "claude.exe", self.config, run=self.fake_claude), "updated")
        self.assertEqual([c[1:3] for c in self.calls], [["mcp", "remove"], ["mcp", "add"]])

    def test_an_entry_with_other_env_is_replaced(self):
        e = mcp_server.server_entry()
        doc = json.loads(self.config.read_text(encoding="utf-8"))
        doc["mcpServers"]["scribe"] = {"type": "stdio", "command": e["command"], "args": e["args"],
                                       "env": {"PYTHONPATH": "C:\\old scribe"}}
        self.config.write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(mcp_server.register(self.tmp / "claude.exe", self.config, run=self.fake_claude), "updated")
        self.assertEqual([c[1:3] for c in self.calls], [["mcp", "remove"], ["mcp", "add"]])
        self.assertEqual(json.loads(self.config.read_text(encoding="utf-8"))["mcpServers"]["scribe"]["env"], e["env"])

    def test_failures_are_reported(self):
        self.fail = True
        with self.assertRaises(RuntimeError) as cm:
            mcp_server.register(self.tmp / "claude.exe", self.config, run=self.fake_claude)
        self.assertIn("boom", str(cm.exception))
        self.fail, self.write_entry = False, False
        with self.assertRaises(RuntimeError):
            mcp_server.register(self.tmp / "claude.exe", self.config, run=self.fake_claude)


# ---------------------------------------------------------------- live

@unittest.skipUnless(os.environ.get("SCRIBE_LIVE_MCP") == "1", "set SCRIBE_LIVE_MCP=1 (read-only, running Scribe)")
class TestLiveApp(unittest.TestCase):
    """status and recent_recordings against the running Scribe, over stdio as Claude Code runs the server."""

    def test_status_and_recent(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        entry = mcp_server.server_entry()
        params = StdioServerParameters(command=entry["command"], args=entry["args"], env=entry["env"],
                                       cwd=tempfile.gettempdir())
        out = {}

        async def go():
            with tempfile.TemporaryFile("w+") as err:
                async with stdio_client(params, errlog=err) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        out["status"] = (await session.call_tool("status", {})).content[0].text
                        out["recent"] = (await session.call_tool("recent_recordings", {"limit": 5})).content[0].text

        anyio.run(go)
        status = json.loads(out["status"])
        self.assertTrue(status["running"], "Scribe is not running")
        self.assertIn("jobs", status)
        rows = json.loads(out["recent"])
        self.assertIsInstance(rows, list)
        for r in rows:
            self.assertIn("status", r)
            self.assertIn("outputs", r)


@unittest.skipUnless(os.environ.get("SCRIBE_LIVE_MCP_GPU") == "1", "set SCRIBE_LIVE_MCP_GPU=1 (GPU, Whisper)")
class TestLiveWhisper(unittest.TestCase):
    """A growing recording transcribed with large-v3-turbo on the GPU, beside whatever else runs there."""

    def test_growing_recording(self):
        audio, _ = sf.read(str(FIXTURES / "two_speakers.wav"), dtype="int16")
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "2026-10-31 0900"
            folder.mkdir()
            mic = sf.SoundFile(str(folder / "mic.flac"), "w", SR, 1, format="FLAC", subtype="PCM_16")
            try:
                mic.write(audio)
                mic.flush()
                live = Live()
                began = time.monotonic()
                r = live.run(folder, None, config.load_settings(), recording=True)
                took = time.monotonic() - began
            finally:
                mic.close()
        self.assertGreater(r["duration"], 40.0)
        self.assertTrue(r["segments"])
        names = {ln["speaker"] for ln in r["lines"] or []}
        sys.stderr.write(f"\nlive whisper: {r['duration']:.0f} s in {took:.1f} s, speakers {r['speakers']}, "
                         f"{len(names)} labels, {len(r['segments'])} segments\n")


if __name__ == "__main__":
    unittest.main()
