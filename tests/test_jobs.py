"""jobs.py: pipeline order, statuses, idempotent resume, failure, Local, interrupted recordings.

Every heavy step (audio, asr, diarize, video, draft, enhance) is a fake module put in sys.modules,
so these tests need no GPU, no ffmpeg and never start Claude. Everything is written under a temp dir.
"""
from __future__ import annotations

import copy
import ctypes
import json
import logging
import sys
import tempfile
import os
import threading
import time
import types
import unittest
from ctypes import wintypes
from pathlib import Path
from unittest import mock

from scribe import config, jobs
from scribe import draft as real_draft
from scribe.jobs import Job, JobQueue, run_pipeline

logging.getLogger("scribe").addHandler(logging.NullHandler())  # expected failures stay out of the test output

WHISPER = {
    "source": "mix.flac", "model": "large-v3", "language": "af", "language_probability": 0.9,
    "languages": {"en": 2.0, "af": 2.0}, "duration": 4.0, "integrity": {"truncated": False},
    "teams_transcript": None,
    "segments": [{"start": 0.0, "end": 4.0, "text": "Goeie môre, let's start", "language": "af",
                  "words": [{"start": 0.0, "end": 1.0, "word": "Goeie"}]}],
}


class Killed(BaseException):
    """Stands in for the process dying mid-step: not an Exception, so the pipeline cannot record it."""


class Fakes:
    """Fake scribe.audio/asr/diarize/video/draft/enhance that log calls and write their outputs."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.args: dict[str, tuple] = {}
        self.fail: dict[str, BaseException] = {}
        self.claude_result = {"returncode": 0, "session_id": "sess-1",
                              "transcript": "Meetings/Kickoff transcript 30 Oct 2026.md",
                              "notes": "Meetings/Kickoff notes 30 Oct 2026.md"}
        self.transcribe_delay = 0.0
        self.barrier: threading.Barrier | None = None    # transcriptions that must all be running at once
        self.oom_waits = 0               # times the fake transcription waits out a GPU out-of-memory error
        self.fail_in_wait: BaseException | None = None   # raised while it waits
        self.vault_files = True          # the fake Claude run writes the files it reports
        self.last_result = None          # what enhance.last_result finds in claude.log
        self.target_existed: dict[str, bool] = {}
        self._busy = 0
        self.max_busy = 0
        self._lock = threading.Lock()
        f = self

        def hit(name, *args):
            with f._lock:
                f.calls.append(name)
                f.args[name] = args
            if name in f.fail:
                raise f.fail[name]

        audio = types.ModuleType("scribe.audio")

        def mix_tracks(folder):
            hit("mix_tracks", folder)
            out = Path(folder) / "mix.flac"
            out.write_bytes(b"fLaC")
            return out

        def to_m4a(src, dst):
            hit("to_m4a", src, dst)
            f.target_existed["to_m4a"] = Path(dst).exists()
            Path(dst).write_bytes(b"m4a")
            return Path(dst)

        audio.mix_tracks, audio.to_m4a = mix_tracks, to_m4a

        asr = types.ModuleType("scribe.asr")

        def transcribe_file(audio_path, model_name, languages, vocabulary, progress=None, on_wait=None):
            with f._lock:
                f._busy += 1
                f.max_busy = max(f.max_busy, f._busy)
            try:
                hit("transcribe_file", audio_path, model_name, languages, vocabulary)
                if f.barrier is not None:
                    f.barrier.wait()
                for _ in range(f.oom_waits):
                    on_wait(True)
                    if f.fail_in_wait is not None:
                        raise f.fail_in_wait
                    on_wait(False)
                if f.transcribe_delay:
                    time.sleep(f.transcribe_delay)
                if progress:
                    progress(0.5)
                    progress(1.0)
                return copy.deepcopy(WHISPER)
            finally:
                with f._lock:
                    f._busy -= 1

        asr.transcribe_file = transcribe_file

        diarize = types.ModuleType("scribe.diarize")

        def diarize_fn(audio_path, models_dir, threshold=0.5, num_speakers=None):
            hit("diarize", audio_path, models_dir, threshold)
            return [{"start": 0.0, "end": 2.0, "speaker": 0}, {"start": 2.0, "end": 4.0, "speaker": 1}]

        class Voiceprint:
            def __init__(self, path, models_dir):
                hit("Voiceprint", path, models_dir)

            @property
            def enrolled(self):
                return True

        def name_speakers(audio, diar, voiceprint, owner_name, owner_match=0.45):
            hit("name_speakers", audio, diar, voiceprint, owner_name, owner_match)
            return {0: owner_name, 1: "Speaker 2"}

        def assign_words(segments, diar, names, gap=3.0):
            hit("assign_words", segments, diar, names)
            return [{"speaker": names[0], "start": 0.0, "end": 2.0, "text": "Goeie môre"},
                    {"speaker": names[1], "start": 2.0, "end": 4.0, "text": "let's start"}]

        def label_job(audio_path, whisper_path, out_path, models_dir, *, threshold, owner_name, owner_match,
                      voiceprint_path, log_path=None):
            # stands in for the child process: the same calls, in the same order, in this process
            diar = diarize_fn(audio_path, models_dir, threshold)
            names = name_speakers([0.0] * 16, diar, Voiceprint(voiceprint_path, models_dir), owner_name, owner_match)
            segments = json.loads(Path(whisper_path).read_text(encoding="utf-8"))["segments"]
            lines = assign_words(segments, diar, names)
            return {"owner": owner_name, "owner_matched": owner_name in names.values(),
                    "names": {str(k): v for k, v in names.items()}, "diarization": diar, "lines": lines}

        diarize.diarize, diarize.Voiceprint = diarize_fn, Voiceprint
        diarize.name_speakers, diarize.assign_words = name_speakers, assign_words
        diarize.label_job = label_job

        video = types.ModuleType("scribe.video")

        def probe(path):
            hit("probe", path)
            return {"duration": 4.0, "has_video": True, "has_audio": True}

        def extract_audio(path, out_flac):
            hit("extract_audio", path, out_flac)
            f.target_existed["extract_audio"] = Path(out_flac).exists()
            Path(out_flac).write_bytes(b"fLaC")
            return Path(out_flac)

        def keyframes(path, out_dir, scene=0.3, max_gap=120.0, max_frames=200):
            hit("keyframes", path, out_dir)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            frame = Path(out_dir) / "f0001.jpg"
            frame.write_bytes(b"jpg")
            return [{"t": 1.0, "file": frame}]

        def describe_frames(frames, ollama_url, model, progress=None):
            hit("describe_frames", frames, ollama_url, model)
            return [{"t": 1.0, "text": "Slide: project pipeline"}]

        class VideoError(RuntimeError):
            pass

        video.probe, video.extract_audio = probe, extract_audio
        video.keyframes, video.describe_frames = keyframes, describe_frames
        video.VideoError = VideoError

        draft = types.ModuleType("scribe.draft")

        def render_draft(whisper, lines, visuals, meta, out):
            hit("render_draft", whisper, lines, visuals, meta, out)
            f.target_existed["render_draft"] = Path(out).exists()
            Path(out).write_text("---\n---\n# draft\n", encoding="utf-8")
            return Path(out)

        draft.render_draft = render_draft
        draft._date_from, draft._title_from = real_draft._date_from, real_draft._title_from

        enhance = types.ModuleType("scribe.enhance")

        def run_claude(job, vault_path, log_path, settings, **kw):
            hit("run_claude", job, vault_path, log_path)
            f.args["run_claude_kw"] = kw
            if kw.get("on_start"):
                kw["on_start"]({"session": kw.get("session_id"), "pid": 4242, "created": 0.0, "name": "claude.exe"})
            Path(log_path).write_text("fake log\n", encoding="utf-8")
            result = dict(f.claude_result)
            if f.vault_files and result.get("returncode") == 0:
                for key in ("transcript", "notes"):
                    if result.get(key):
                        target = Path(vault_path) / result[key]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(key, encoding="utf-8")
            return result

        enhance.run_claude = run_claude
        enhance.last_result = lambda log_path: f.last_result

        self.modules = {"scribe.audio": audio, "scribe.asr": asr, "scribe.diarize": diarize,
                        "scribe.video": video, "scribe.draft": draft, "scribe.enhance": enhance}

    def count(self, name: str) -> int:
        return self.calls.count(name)


def wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class JobsTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.settings = copy.deepcopy(config.DEFAULTS)
        self.settings.update(owner_name="Alex Morgan", languages=["en", "af"])
        self.settings["output_root"] = str(self.root / "out")
        self.settings["vaults"] = {"Work": str(self.root / "vaultA"), "Personal": str(self.root / "vaultP")}
        self.recordings = config.recordings_dir(self.settings)
        self.fakes = Fakes()
        for patcher in (
            mock.patch.dict(sys.modules, self.fakes.modules),
            mock.patch.object(config, "APP_DIR", self.root / "app"),
            mock.patch.object(jobs, "_load_audio", lambda path: [0.0] * 16),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.statuses: list[str] = []
        self.disk_statuses: list[str] = []

    def tearDown(self):
        self._tmp.cleanup()

    def progress(self, job: Job, fraction):
        if fraction is None:
            self.statuses.append(job.status)
            self.disk_statuses.append(json.loads(job.path.read_text(encoding="utf-8"))["status"])

    def meeting(self, name: str = "2026-10-30 1415", vault: str = "Personal") -> Job:
        folder = self.recordings / name
        folder.mkdir()
        (folder / "mic.flac").write_bytes(b"fLaC")
        (folder / "system.flac").write_bytes(b"fLaC")
        (folder / "recording.json").write_text(json.dumps({"status": "stopped", "events": []}), encoding="utf-8")
        return Job.new("meeting", folder, vault)

    def on_disk(self, job: Job) -> dict:
        # the queue may be replacing job.json at this moment: Windows refuses the read briefly, as it does the UI's
        return jobs._retrying(lambda: json.loads(job.path.read_text(encoding="utf-8")))


class JobRecordTest(JobsTestBase):
    def test_round_trip_and_atomic_write(self):
        source = self.root / "Inbox" / "call.m4a"
        folder = self.recordings / "2026-10-30 0900"
        job = Job.new("file", folder, "Personal", source)
        job.outputs["whisper"] = "whisper.json"
        job.save()
        data = self.on_disk(job)
        self.assertEqual(data["source"], str(source))
        self.assertEqual(data["folder"], str(folder))
        self.assertEqual(data["id"], "2026-10-30 0900")
        self.assertFalse((folder / "job.json.tmp").exists())
        loaded = Job.load(folder)
        self.assertEqual(loaded.to_dict(), job.to_dict())
        self.assertIsInstance(loaded.source, Path)
        with self.assertRaises(ValueError):
            Job.new("podcast", folder, "Personal")


class PipelineTest(JobsTestBase):
    def test_meeting_runs_every_step_in_order(self):
        job = self.meeting()
        run_pipeline(job, self.settings, progress=self.progress)
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(self.fakes.calls, ["mix_tracks", "transcribe_file", "diarize", "Voiceprint", "name_speakers",
                                            "assign_words", "render_draft", "to_m4a", "run_claude"])
        self.assertEqual(self.statuses, ["queued", "transcribing", "labelling", "drafting", "enhancing", "done"])
        self.assertEqual(self.disk_statuses, self.statuses)  # job.json is saved before anyone hears of a change
        folder = job.folder
        for name in ("mix.flac", "whisper.json", "speakers.json", "draft.md", "recording.m4a", "claude.log", "enhance.json"):
            self.assertTrue((folder / name).exists(), name)
        for leftover in ("draft.tmp.md", "recording.tmp.m4a", "job.json.tmp", "whisper.json.tmp"):
            self.assertFalse((folder / leftover).exists(), leftover)
        disk = self.on_disk(job)
        self.assertEqual(disk["status"], "done")
        self.assertEqual(disk["outputs"]["transcript"], "Meetings/Kickoff transcript 30 Oct 2026.md")
        self.assertEqual(disk["outputs"]["session_id"], "sess-1")
        # asr gets the meeting model and the configured languages and vocabulary
        _, model, languages, vocabulary = self.fakes.args["transcribe_file"]
        self.assertEqual((model, languages, vocabulary),
                         ("large-v3", ["en", "af"], config.DEFAULTS["vocabulary"]))
        # the voiceprint and models come from APP_DIR, never a vault
        vp_path, models = self.fakes.args["Voiceprint"]
        self.assertEqual(vp_path, self.root / "app" / "voiceprint.json")
        self.assertEqual(models, self.root / "app" / "models")
        speakers = json.loads((folder / "speakers.json").read_text(encoding="utf-8"))
        self.assertTrue(speakers["owner_matched"])
        self.assertEqual(speakers["names"], {"0": "Alex Morgan", "1": "Speaker 2"})
        # the draft gets the labelled lines and no visuals
        _, lines, visuals, meta, _ = self.fakes.args["render_draft"]
        self.assertEqual(lines[0]["speaker"], "Alex Morgan")
        self.assertIsNone(visuals)
        self.assertEqual(meta["date"], "2026-10-30")
        self.assertEqual(meta["recording_status"], "complete")
        self.assertTrue(meta["owner_matched"])
        # Claude runs in the chosen vault, logs in the job folder, and gets the date and owner
        payload, vault_path, log_path = self.fakes.args["run_claude"]
        self.assertEqual(vault_path, Path(self.settings["vaults"]["Personal"]))
        self.assertEqual(log_path, folder / "claude.log")
        self.assertEqual((payload["date"], payload["owner_name"], payload["video"]), ("2026-10-30", "Alex Morgan", False))
        self.assertEqual(payload["folder"], str(folder))

    def test_video_file_adds_frames_and_never_copies_the_source(self):
        inbox = self.root / "out" / "Inbox" / "Local"
        inbox.mkdir(parents=True)
        source = inbox / "demo.mp4"
        source.write_bytes(b"mp4")
        folder = self.recordings / "2026-10-30 1600"
        job = Job.new("file", folder, "Local", source)
        run_pipeline(job, self.settings, progress=self.progress)
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(self.fakes.calls, ["probe", "extract_audio", "transcribe_file", "diarize", "Voiceprint",
                                            "name_speakers", "assign_words", "keyframes", "describe_frames",
                                            "render_draft", "to_m4a"])
        self.assertEqual(self.statuses, ["queued", "transcribing", "labelling", "frames", "drafting", "done"])
        self.assertTrue(source.exists())
        self.assertFalse((folder / "demo.mp4").exists())
        self.assertEqual(self.fakes.args["extract_audio"][0], source)
        self.assertTrue((folder / "audio.flac").exists())
        self.assertFalse((folder / "audio.tmp.flac").exists())
        self.assertEqual(self.fakes.args["transcribe_file"][0], folder / "audio.flac")
        self.assertEqual(self.fakes.args["keyframes"], (source, folder / "frames"))
        self.assertEqual(self.fakes.args["describe_frames"][1:],
                         (self.settings["ollama"]["url"], self.settings["ollama"]["vision_model"]))
        _, _, visuals, meta, _ = self.fakes.args["render_draft"]
        self.assertEqual(visuals, [{"t": 1.0, "text": "Slide: project pipeline"}])
        self.assertEqual((meta["title"], meta["source"]), ("demo", str(source)))
        self.assertTrue(json.loads((folder / "visuals.json").read_text(encoding="utf-8")))

    def _video_job(self, stamp: str) -> Job:
        source = self.root / "demo.mp4"
        source.write_bytes(b"mp4")
        return Job.new("file", self.recordings / stamp, "Local", source)

    def test_video_goes_ahead_without_notes_when_vision_is_unavailable(self):
        video = self.fakes.modules["scribe.video"]

        def no_vision(frames, ollama_url, model, progress=None):
            raise video.VideoError("ollama: model 'qwen' not found")

        video.describe_frames = no_vision
        job = run_pipeline(self._video_job("2026-10-30 1610"), self.settings, progress=self.progress)
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(self.statuses, ["queued", "transcribing", "labelling", "frames", "drafting", "done"])
        self.assertEqual(json.loads((job.folder / "visuals.json").read_text(encoding="utf-8")), [])
        self.assertIn("not found", job.outputs["visuals_error"])
        _, _, visuals, _, _ = self.fakes.args["render_draft"]
        self.assertEqual(visuals, [])
        # Deleting visuals.json redoes only that step; a success clears the recorded error.
        (job.folder / "visuals.json").unlink()
        video.describe_frames = lambda frames, url, model, progress=None: [{"t": 1.0, "text": "Slide"}]
        job.status = "queued"
        job = run_pipeline(job, self.settings)
        self.assertEqual(job.status, "done", job.error)
        self.assertNotIn("visuals_error", job.outputs)
        self.assertEqual(json.loads((job.folder / "visuals.json").read_text(encoding="utf-8")), [{"t": 1.0, "text": "Slide"}])

    def test_other_frame_errors_still_fail_the_job(self):
        video = self.fakes.modules["scribe.video"]

        def broken_keyframes(path, out_dir, scene=0.3, max_gap=120.0, max_frames=200):
            raise video.VideoError("keyframe extraction failed")

        def broken_describe(frames, ollama_url, model, progress=None):
            raise ValueError("bad frame list")

        for stamp, patch in (("2026-10-30 1620", {"keyframes": broken_keyframes}),
                             ("2026-10-30 1630", {"describe_frames": broken_describe})):
            with mock.patch.multiple(video, **patch):
                job = run_pipeline(self._video_job(stamp), self.settings)
            self.assertEqual(job.status, "failed")
            self.assertTrue(job.error.startswith("frames: "), job.error)
            self.assertFalse((job.folder / "visuals.json").exists())

    def test_audio_file_skips_probe_and_frames(self):
        source = self.root / "call.m4a"
        source.write_bytes(b"m4a")
        job = Job.new("file", self.recordings / "2026-10-30 1700", "Local", source)
        run_pipeline(job, self.settings, progress=self.progress)
        self.assertEqual(job.status, "done", job.error)
        self.assertNotIn("probe", self.fakes.calls)
        self.assertNotIn("keyframes", self.fakes.calls)
        self.assertNotIn("frames", self.statuses)

    def test_local_vault_stops_at_the_draft(self):
        job = self.meeting(vault="Local")
        run_pipeline(job, self.settings, progress=self.progress)
        self.assertEqual(job.status, "done", job.error)
        self.assertNotIn("run_claude", self.fakes.calls)
        self.assertNotIn("enhancing", self.statuses)
        self.assertFalse((job.folder / "claude.log").exists())
        self.assertTrue((job.folder / "draft.md").exists())
        self.assertNotIn("transcript", job.outputs)

    def test_failed_step_records_error_stops_and_retries_from_there(self):
        job = self.meeting()
        self.fakes.fail["diarize"] = RuntimeError("sherpa model missing")
        with self.assertLogs("scribe.jobs", "ERROR") as logged:
            run_pipeline(job, self.settings, progress=self.progress)
        self.assertIn("failed at labelling", logged.output[0])
        self.assertEqual(job.status, "failed")
        self.assertTrue(job.error.startswith("labelling: RuntimeError"), job.error)
        self.assertIn("sherpa model missing", job.error)
        self.assertNotIn("render_draft", self.fakes.calls)
        self.assertNotIn("run_claude", self.fakes.calls)
        disk = self.on_disk(job)
        self.assertEqual((disk["status"], disk["error"]), ("failed", job.error))
        self.assertEqual(self.statuses[-1], "failed")

        del self.fakes.fail["diarize"]
        self.fakes.calls.clear()
        retried = run_pipeline(Job.load(job.folder), self.settings)
        self.assertEqual(retried.status, "done", retried.error)
        self.assertIsNone(retried.error)
        self.assertEqual(self.fakes.calls[:2], ["diarize", "Voiceprint"])  # mix and transcribe not redone

    def test_claude_failure_keeps_session_for_resume(self):
        job = self.meeting()
        self.fakes.claude_result = {"returncode": 1, "session_id": "sess-err", "transcript": None, "notes": None}
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "failed")
        self.assertTrue(job.error.startswith("enhancing:"), job.error)
        self.assertEqual(self.on_disk(job)["outputs"]["session_id"], "sess-err")
        self.assertFalse((job.folder / "enhance.json").exists())

    def test_claude_without_result_line_fails(self):
        job = self.meeting()
        self.fakes.claude_result = {"returncode": 0, "session_id": "s", "transcript": None, "notes": None}
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "failed")
        self.assertIn("no result line", job.error)

    def test_unknown_vault_fails_at_enhancing(self):
        job = self.meeting(vault="Nowhere")
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "enhancing: ValueError: unknown vault: Nowhere")
        self.assertNotIn("run_claude", self.fakes.calls)

    def test_stale_temp_outputs_are_cleared_before_rewriting(self):
        source = self.root / "talk.mp4"
        source.write_bytes(b"mp4")
        folder = self.recordings / "2026-10-30 1800"
        folder.mkdir()
        for stale in ("audio.tmp.flac", "draft.tmp.md", "recording.tmp.m4a"):
            (folder / stale).write_bytes(b"half")
        job = Job.new("file", folder, "Local", source)
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(self.fakes.target_existed,
                         {"extract_audio": False, "render_draft": False, "to_m4a": False})

    def test_deleted_output_redoes_only_that_step(self):
        job = self.meeting(vault="Local")
        run_pipeline(job, self.settings)
        (job.folder / "draft.md").unlink()
        job.status = "queued"
        self.fakes.calls.clear()
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "done", job.error)
        self.assertEqual(self.fakes.calls, ["render_draft"])


class FixesTest(JobsTestBase):
    """The pipeline defects found in review, each pinned by a test."""

    def file_job(self, name: str, inside: bool = False, vault: str = "Local") -> Job:
        folder = self.recordings / "2026-10-30 1700"
        folder.mkdir(exist_ok=True)
        where = folder if inside else self.root / "drop"
        where.mkdir(exist_ok=True)
        source = where / name
        source.write_bytes(b"ORIGINAL RECORDING")
        return Job.new("file", folder, vault, source)

    def test_file_job_date_and_title_come_from_the_recordings_own_stamp(self):
        job = self.file_job("Weekly sync-20261015_143000-Meeting Recording.mp4")
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "done", job.error)
        _, _, _, meta, _ = self.fakes.args["render_draft"]
        self.assertEqual((meta["date"], meta["title"]), ("2026-10-15", "Weekly sync"))

    def test_a_file_without_a_stamp_falls_back_to_its_mtime(self):
        job = self.file_job("call.m4a")
        os_time = time.mktime((2026, 10, 20, 12, 0, 0, 0, 0, -1))
        import os
        os.utime(job.source, (os_time, os_time))
        run_pipeline(job, self.settings)
        _, _, _, meta, _ = self.fakes.args["render_draft"]
        self.assertEqual((meta["date"], meta["title"]), ("2026-10-20", "call"))

    def test_a_source_named_like_an_output_is_never_overwritten(self):
        for name in ("Recording.m4a", "audio.flac", "audio.tmp.flac"):
            with self.subTest(name=name):
                job = self.file_job(name, inside=True)
                run_pipeline(job, self.settings)
                self.assertEqual(job.status, "done", job.error)
                self.assertEqual(job.source.read_bytes(), b"ORIGINAL RECORDING")
                for leftover in job.folder.iterdir():
                    if leftover.name not in (name, "job.json"):
                        leftover.unlink()

    def test_enhancing_records_its_session_before_claude_runs(self):
        job = self.meeting()
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "done", job.error)
        kw = self.fakes.args["run_claude_kw"]
        self.assertFalse(kw.get("resume"))
        run = self.on_disk(job)["outputs"]["claude_run"]
        self.assertEqual(run["session"], kw["session_id"])
        self.assertEqual(run["pid"], 4242)

    def interrupted(self) -> Job:
        job = self.meeting()
        self.fakes.fail["run_claude"] = Killed()
        with self.assertRaises(Killed):
            run_pipeline(job, self.settings)
        del self.fakes.fail["run_claude"]
        self.fakes.calls.clear()
        self.assertEqual(self.on_disk(job)["status"], "enhancing")
        return Job.load(job.folder)

    def test_a_cut_off_run_is_never_started_again_on_its_own(self):
        job = self.interrupted()
        job.outputs["claude_run"] = {"session": "sid-cut", "pid": 4242, "created": 0.0}
        job.save()
        run_pipeline(Job.load(job.folder), self.settings)
        disk = self.on_disk(job)
        self.assertEqual(disk["status"], "failed")
        self.assertIn("interrupted during enhancement", disk["error"])
        self.assertNotIn("run_claude", self.fakes.calls)

    def test_retry_resumes_the_cut_off_session(self):
        job = self.interrupted()
        job.outputs["claude_run"] = {"session": "sid-cut", "pid": 4242, "created": 0.0}
        job.save()
        run_pipeline(Job.load(job.folder), self.settings)            # fails as interrupted
        q = JobQueue(self.settings)
        self.addCleanup(q.stop, 5)
        q.start()
        q.retry(job.folder)
        self.assertTrue(wait_for(lambda: self.on_disk(job)["status"] == "done"))
        kw = self.fakes.args["run_claude_kw"]
        self.assertEqual((kw["session_id"], kw["resume"]), ("sid-cut", True))
        self.assertEqual(self.fakes.count("run_claude"), 1)

    def test_a_run_that_finished_unwatched_is_taken_from_the_log(self):
        job = self.interrupted()
        job.outputs["claude_run"] = {"session": "sid-done", "pid": 4242, "created": 0.0}
        job.save()
        vault = Path(self.settings["vaults"]["Personal"])
        for rel in ("Meetings/a transcript.md", "Meetings/a notes.md"):
            (vault / rel).parent.mkdir(parents=True, exist_ok=True)
            (vault / rel).write_text("x", encoding="utf-8")
        self.fakes.last_result = {"returncode": 0, "session_id": "sid-done", "transcript": "Meetings/a transcript.md",
                                  "notes": "Meetings/a notes.md"}
        done = run_pipeline(Job.load(job.folder), self.settings)
        self.assertEqual(done.status, "done", done.error)
        self.assertEqual(done.outputs["transcript"], "Meetings/a transcript.md")
        self.assertNotIn("run_claude", self.fakes.calls)

    def test_reported_paths_must_be_files_in_the_vault(self):
        job = self.meeting()
        self.fakes.vault_files = False
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "failed")
        self.assertIn("not in the vault", job.error)
        self.assertFalse((job.folder / "enhance.json").exists())
        job = self.meeting("2026-10-30 1416")
        self.fakes.vault_files = True
        self.fakes.claude_result = {**self.fakes.claude_result, "notes": "../../escape.md"}
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "failed")
        self.assertIn("notes", job.error)

    def test_a_save_failure_still_ends_the_job_failed(self):
        job = self.meeting(vault="Local")
        real_save = Job.save
        state = {"raised": False}

        def flaky(self_job):
            if self_job.status == "labelling" and not state["raised"]:
                state["raised"] = True
                raise PermissionError(13, "held by antivirus")
            return real_save(self_job)

        with mock.patch.object(Job, "save", flaky):
            run_pipeline(job, self.settings, progress=self.progress)
        self.assertEqual(job.status, "failed")
        self.assertTrue(job.error.startswith("labelling: PermissionError"), job.error)
        self.assertEqual(self.on_disk(job)["status"], "failed")
        self.assertEqual(self.statuses[-1], "failed")

    def test_replace_rides_out_a_sharing_violation(self):
        calls = []
        real = jobs.os.replace

        def busy(src, dst):
            calls.append(1)
            if len(calls) < 3:
                err = OSError("in use")
                err.winerror = 32
                raise err
            return real(src, dst)

        target = self.root / "x.json"
        with mock.patch.object(jobs.os, "replace", busy), mock.patch.object(jobs.time, "sleep"):
            jobs.write_json_atomic(target, {"a": 1})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"a": 1})
        self.assertEqual(len(calls), 3)

    def test_a_folder_held_by_another_runner_is_left_alone(self):
        job = self.meeting(vault="Local")
        holder = jobs.FolderLock(job.folder)
        self.assertTrue(holder.acquire())
        try:
            self.assertTrue(jobs.folder_busy(job.folder))
            run_pipeline(job, self.settings)
            self.assertEqual(job.status, "queued")
            self.assertEqual(self.fakes.calls, [])
            q = JobQueue(self.settings)
            self.addCleanup(q.stop, 5)
            job.save()
            q.start()
            time.sleep(0.3)
            self.assertEqual(self.fakes.calls, [])
        finally:
            holder.release()
        self.assertFalse(jobs.folder_busy(job.folder))

    def test_gpu_steps_run_side_by_side_across_runners(self):
        # the GPU runs more than one job at a time: no runner's GPU step waits for another's
        self.fakes.barrier = threading.Barrier(3, timeout=5)
        started = [self.meeting(f"2026-10-30 20{i}0", "Local") for i in range(3)]
        runners = [threading.Thread(target=run_pipeline, args=(job, self.settings)) for job in started]
        for t in runners:
            t.start()
        for t in runners:
            t.join(10)
        self.assertEqual([job.status for job in started], ["done"] * 3)
        self.assertEqual(self.fakes.max_busy, 3)

    def test_the_old_gpu_mutex_holds_up_nothing(self):
        # Local\ScribeGPU-<pid> queued this process's GPU steps until 31 Oct 2026; holding it changes nothing now
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        held, release = threading.Event(), threading.Event()

        def hold():
            handle = kernel32.CreateMutexW(None, True, f"Local\\ScribeGPU-{os.getpid()}")
            held.set()
            release.wait(10)
            kernel32.CloseHandle(handle)

        holder = threading.Thread(target=hold)
        holder.start()
        try:
            self.assertTrue(held.wait(5))
            job = self.meeting("2026-10-30 2100", "Local")
            runner = threading.Thread(target=run_pipeline, args=(job, self.settings))
            runner.start()
            runner.join(5)
            self.assertEqual(job.status, "done")
        finally:
            release.set()
            holder.join(5)

    def test_a_job_backing_off_for_gpu_memory_says_so(self):
        self.fakes.oom_waits = 2
        seen = []
        progress = lambda j, f: seen.append((j.status, j.waiting)) if f is None else None
        job = run_pipeline(self.meeting("2026-10-30 2100", "Local"), self.settings, progress=progress)
        self.assertEqual(job.status, "done")
        self.assertEqual([w for s, w in seen if s == "transcribing"], [None, "gpu", None, "gpu", None])
        self.assertIsNone(job.waiting)
        self.assertIsNone(self.on_disk(job)["waiting"])

    def test_a_failure_while_backing_off_clears_waiting(self):
        self.fakes.oom_waits = 1
        self.fakes.fail_in_wait = RuntimeError("stopped while waiting")
        job = run_pipeline(self.meeting("2026-10-30 2100", "Local"), self.settings)
        self.assertEqual(job.status, "failed")
        self.assertIsNone(job.waiting)
        self.assertIsNone(self.on_disk(job)["waiting"])

    def test_a_torn_recording_json_is_an_interrupted_recording(self):
        folder = self.recordings / "2026-10-29 1100"
        folder.mkdir()
        (folder / "mic.flac").write_bytes(b"fLaC")
        (folder / "recording.json").write_bytes(b"\0" * 64)          # renamed, never written: a power cut
        job = jobs.recover_folder(folder, self.settings)
        self.assertEqual(job.vault, "Local")
        self.assertTrue(job.outputs["interrupted"])
        self.assertEqual(jobs.read_json(folder / "recording.json")["status"], "stopped")

    def test_a_local_job_can_be_sent_to_a_vault_later(self):
        job = self.meeting(vault="Local")
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "done")
        self.fakes.calls.clear()
        q = JobQueue(self.settings)
        self.addCleanup(q.stop, 5)
        q.start()
        q.send_to_vault(job.folder, "Personal")
        self.assertTrue(wait_for(lambda: self.on_disk(job)["status"] == "done" and q.active() == 0))
        self.assertEqual(self.fakes.calls, ["run_claude"])
        self.assertEqual(self.on_disk(job)["vault"], "Personal")
        with self.assertRaises(ValueError):
            q.send_to_vault(job.folder, "Nowhere")


class KillOnCloseTest(unittest.TestCase):
    def test_children_die_with_the_process(self):
        import psutil
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "sleeper.pid"
            root = str(Path(jobs.__file__).resolve().parents[1])
            script = (
                "import os, subprocess, sys, time\n"
                f"sys.path.insert(0, {root!r})\n"
                "from scribe import jobs\n"
                "assert jobs.kill_children_with_me()\n"
                "code = 'import os, time; open(sys.argv[1], \"w\").write(str(os.getpid())); time.sleep(60)'\n"
                "subprocess.Popen([sys.executable, '-c', 'import sys\\n' + code, sys.argv[1]],"
                " creationflags=subprocess.CREATE_NO_WINDOW)\n"
                "deadline = time.time() + 20\n"
                "while not os.path.exists(sys.argv[1]) and time.time() < deadline: time.sleep(0.05)\n"
                "time.sleep(0.2)\n"
                "os._exit(0)\n"
            )
            subprocess.run([sys.executable, "-c", script, str(pid_file)], timeout=60,
                           creationflags=subprocess.CREATE_NO_WINDOW, check=True)
            pid = int(pid_file.read_text())
            deadline = time.monotonic() + 5
            while psutil.pid_exists(pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertFalse(psutil.pid_exists(pid), "a child outlived its parent")


class QueueTest(JobsTestBase):
    def queue(self, notified: list | None = None) -> JobQueue:
        q = JobQueue(self.settings, notify=(notified.append if notified is not None else None))
        self.addCleanup(q.stop, 5)
        return q

    def test_kill_after_transcribing_then_restart_skips_it(self):
        job = self.meeting()
        self.fakes.fail["diarize"] = Killed()
        with self.assertRaises(Killed):
            run_pipeline(job, self.settings)
        disk = self.on_disk(job)
        self.assertEqual(disk["status"], "labelling")
        self.assertTrue((job.folder / "whisper.json").exists())
        self.assertFalse((job.folder / "speakers.json").exists())

        del self.fakes.fail["diarize"]
        notified: list[Job] = []
        q = self.queue(notified)
        q.start()
        self.assertTrue(wait_for(lambda: self.on_disk(job)["status"] == "done" and q.active() == 0))
        self.assertEqual(self.fakes.count("mix_tracks"), 1)
        self.assertEqual(self.fakes.count("transcribe_file"), 1)
        self.assertEqual(self.fakes.count("diarize"), 2)
        self.assertEqual(self.fakes.count("run_claude"), 1)
        self.assertIn("done", [j.status for j in notified])
        self.assertEqual([j.status for j in q.jobs()], ["done"])

    def test_interrupted_recording_is_stopped_and_queued_local(self):
        folder = self.recordings / "2026-10-29 0900"
        folder.mkdir()
        (folder / "mic.flac").write_bytes(b"fLaC")
        (folder / "recording.json").write_text(json.dumps(
            {"status": "recording", "started": "2026-10-29T09:00:00", "last_flush": "2026-10-29T10:12:05",
             "events": []}), encoding="utf-8")
        q = self.queue()
        q.start()
        self.assertTrue(wait_for(lambda: (folder / "job.json").exists()
                                 and jobs._retrying(lambda: jobs.read_json(folder / "job.json"))["status"] == "done"))
        rec = json.loads((folder / "recording.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["status"], "stopped")
        self.assertEqual(rec["started"], "2026-10-29T09:00:00")
        self.assertTrue(rec["events"])
        job = Job.load(folder)
        self.assertEqual((job.kind, job.vault), ("meeting", "Local"))       # nobody chose a vault: none is written
        self.assertTrue(job.outputs["interrupted"])
        _, _, _, meta, _ = self.fakes.args["render_draft"]
        self.assertTrue(meta["recording_status"].startswith("partial"))
        self.assertEqual(meta["date"], "2026-10-29")
        self.assertNotIn("run_claude", self.fakes.calls)

    def test_start_ignores_finished_failed_and_empty_folders(self):
        for name, status in (("2026-10-28 0800", "done"), ("2026-10-28 0900", "failed")):
            job = Job.new("meeting", self.recordings / name, "Local")
            job.status = status
            job.save()
        (self.recordings / "2026-10-28 1000").mkdir()
        q = self.queue()
        q.start()
        time.sleep(0.3)
        self.assertEqual(self.fakes.calls, [])
        self.assertEqual(q.jobs(), [])

    def test_one_job_at_a_time_and_no_double_submit(self):
        self.fakes.transcribe_delay = 0.15
        q = self.queue()
        q.start()
        submitted = [self.meeting(f"2026-10-30 10{i}0", vault="Local") for i in range(3)]
        for job in submitted:
            q.submit(job)
        q.submit(submitted[0])  # already waiting: ignored
        self.assertTrue(wait_for(lambda: q.active() == 0 and len(q.jobs()) == 3
                                 and all(j.status == "done" for j in q.jobs())))
        self.assertEqual(self.fakes.max_busy, 1)
        self.assertEqual(self.fakes.count("transcribe_file"), 3)

    def test_retry_requeues_a_failed_job(self):
        job = self.meeting(vault="Local")
        self.fakes.fail["render_draft"] = RuntimeError("disk full")
        run_pipeline(job, self.settings)
        self.assertEqual(job.status, "failed")
        del self.fakes.fail["render_draft"]
        q = self.queue()
        q.start()
        q.retry(job.folder)
        self.assertTrue(wait_for(lambda: self.on_disk(job)["status"] == "done"))
        self.assertEqual(self.fakes.count("transcribe_file"), 1)

    def test_asr_lock_is_held_while_transcribing(self):
        lock = threading.Lock()
        held: list[bool] = []
        original = self.fakes.modules["scribe.asr"].transcribe_file

        def spy(*args, **kwargs):
            held.append(lock.locked())
            return original(*args, **kwargs)

        self.fakes.modules["scribe.asr"].transcribe_file = spy
        q = JobQueue(self.settings, asr_lock=lock)
        self.addCleanup(q.stop, 5)
        q.start()
        job = self.meeting(vault="Local")
        q.submit(job)
        self.assertTrue(wait_for(lambda: self.on_disk(job)["status"] == "done"))
        self.assertEqual(held, [True])
        self.assertFalse(lock.locked())


if __name__ == "__main__":
    unittest.main()
