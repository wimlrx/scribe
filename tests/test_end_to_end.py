"""End to end: real files through the real pipeline with vault Local, so no Claude run.

A copy of two_speakers.wav, and a generated slide video carrying the same audio, go through jobs.run_pipeline with
Whisper large-v3 on CUDA, sherpa-onnx diarisation, a voiceprint enrolled from owner_david_enroll.wav, and ffmpeg.
Everything is written under one tempfile dir: settings come from a temp settings.json (output_root), config.APP_DIR
is redirected so the voiceprint is a temp one, and the sherpa models are read, never written, from the real models
folder. The vision model runs only when Ollama has it; otherwise describe_frames is replaced by a stub. Skips when
CUDA, the cached large-v3, the sherpa models or ffmpeg are missing; it never downloads anything.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scribe import config, diarize, draft, enhance, jobs, video  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
MODELS = config.APP_DIR / "models"          # read only; captured before any test redirects APP_DIR
OWNER = "Alex Morgan"
MEETING_MODEL = config.DEFAULTS["meetings"]["model"]
VISION_MODEL = config.DEFAULTS["ollama"]["vision_model"]
FIXTURE_SECONDS = 46.41
FRONTMATTER_KEYS = ["client", "people", "date", "source", "speaker_source", "duration_min",
                    "recording_status", "transcriber", "coverage", "redactions"]
FIXTURE_WORDS = ("sourdough", "dashboard", "Tuesday")
FONT = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf"
SPOKEN = re.compile(r"^- \[(\d+:\d\d(?::\d\d)?)\] \*\*(.+?)\*\* \S")
ON_SCREEN = re.compile(r"^- \[(\d+:\d\d(?::\d\d)?)\] \*\(on screen\)\* \S")

# (seconds, background, text colour, title): slide changes at 0, 15 and 30 s, each a large jump in brightness or hue
SLIDES = [
    (15, "0x4080D0", "white", "QUARTERLY REVENUE"),
    (15, "0x10301A", "white", "COST REVIEW"),
    (17, "0xE8D060", "black", "NEXT STEPS"),
]
SLIDE_CHANGES = [0.0, 15.0, 30.0]


def missing_requirement() -> str | None:
    """Why the real pipeline cannot run here, or None. Checks only; never fetches a model."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() < 1:
            return "no CUDA device"
    except Exception as exc:
        return f"CUDA unavailable: {exc!r}"
    try:
        from faster_whisper.utils import download_model

        download_model(MEETING_MODEL, local_files_only=True)
    except Exception:
        return f"{MEETING_MODEL} not complete in the local HF cache"
    for name in (diarize.SEGMENTATION_MODEL, diarize.EMBEDDING_MODEL):
        if not (MODELS / name).is_file():
            return f"sherpa-onnx model missing: {MODELS / name}"
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return "ffmpeg or ffprobe not on PATH"
    return None


def vision_model_installed(model: str = VISION_MODEL) -> bool:
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30,
                             stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return False
    names = {line.split()[0] for line in res.stdout.splitlines()[1:] if line.strip()}
    return res.returncode == 0 and model in names


def build_slide_video(folder: Path) -> Path:
    """~46 s, 1280x720 at 10 fps: three drawtext slides over two_speakers.wav as the audio track."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(FONT, folder / "arial.ttf")           # relative font/text paths: no drive-colon escaping
    args, chains = [], []
    for i, (secs, bg, fg, title) in enumerate(SLIDES):
        (folder / f"title{i}.txt").write_text(title, encoding="ascii")
        args += ["-f", "lavfi", "-i", f"color=c={bg}:s=1280x720:r=10:d={secs}"]
        chains.append(f"[{i}:v]drawtext=fontfile=arial.ttf:textfile=title{i}.txt:fontsize=96:"
                      f"fontcolor={fg}:x=80:y=80[s{i}]")
    n = len(SLIDES)
    graph = ";".join(chains) + ";" + "".join(f"[s{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    args += ["-i", str(FIXTURES / "two_speakers.wav"), "-filter_complex", graph, "-map", "[v]", "-map", f"{n}:a",
             "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "64k", "slides.mp4"]
    res = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", *args], cwd=folder,
                         capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    if res.returncode != 0:
        raise RuntimeError(f"slide video build failed: {res.stderr[-800:]}")
    return folder / "slides.mp4"


class Sandbox:
    """One temp root holding settings.json, the output root and the app dir, plus the patches that point Scribe there."""

    def __init__(self, prefix: str):
        self.root = Path(tempfile.mkdtemp(prefix=prefix))
        self.out = self.root / "Scribe"
        self.app = self.root / "app"
        settings_path = self.root / "settings.json"
        settings_path.write_text(json.dumps({"output_root": str(self.out), "owner_name": OWNER,
                                             "languages": ["en", "af"]}), encoding="utf-8")
        self.settings = config.load_settings(settings_path)
        self.settings["vaults"] = {}                # no real vault path anywhere in a Local run
        self._patches = [mock.patch.object(config, "APP_DIR", self.app),
                         mock.patch.object(config, "models_dir", lambda: MODELS)]
        for p in self._patches:
            p.start()

    def close(self, keep: bool = False) -> None:
        """Undo the patches; remove the temp root unless keep (a live run left for inspection)."""
        for p in reversed(self._patches):
            p.stop()
        if not keep:
            shutil.rmtree(self.root, ignore_errors=True)

    def enroll_owner(self) -> bool:
        """Three 5 s clips of the enrolment fixture: the voiceprint commits once three candidates agree."""
        vp = diarize.Voiceprint(config.APP_DIR / "voiceprint.json", MODELS)
        audio = diarize.read_audio(FIXTURES / "owner_david_enroll.wav")
        sr = diarize.SAMPLE_RATE
        for k in range(3):
            vp.enroll(audio[k * 5 * sr:(k + 1) * 5 * sr])
        return vp.enrolled

    def drop(self, path: Path) -> Path:
        """A source file lying outside the Recordings tree, as a Send to or --add file would."""
        dest = self.root / "drop" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest)
        return dest


def run_local_job(sb: Sandbox, source: Path) -> tuple[jobs.Job, list[str], list]:
    """Submit a file job with vault Local and run it synchronously. Returns the job, the statuses it passed
    through, and any calls that reached enhance.run_claude (there must be none)."""
    folder = config.new_job_folder(settings=sb.settings)
    job = jobs.Job.new("file", folder, jobs.LOCAL, source=source)
    job.save()
    statuses: list[str] = []
    claude_calls: list = []

    def progress(j: jobs.Job, fraction: float | None) -> None:
        if fraction is None:
            statuses.append(j.status)

    def no_claude(*args, **kwargs):
        claude_calls.append(args)
        raise AssertionError("enhance.run_claude reached for a Local job")

    with mock.patch.object(enhance, "run_claude", no_claude):
        job = jobs.run_pipeline(job, sb.settings, progress=progress)
    return job, statuses, claude_calls


def split_draft(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """(frontmatter as ordered (key, value) pairs, every line after the closing ---)."""
    lines = text.split("\n")
    assert lines[0] == "---", lines[0]
    end = lines.index("---", 1)
    front = [tuple(line.split(": ", 1)) if ": " in line else (line.rstrip(":"), "") for line in lines[1:end]]
    return front, lines[end + 1:]


def ts_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]


class _EndToEnd:
    """Shared checks for a finished Local job built from two_speakers.wav audio."""

    PREFIX = "scribe_e2e_"
    job: jobs.Job
    sb: Sandbox

    @classmethod
    def setUpClass(cls):
        reason = missing_requirement()
        if reason:
            raise unittest.SkipTest(reason)
        cls.sb = Sandbox(cls.PREFIX)
        cls.addClassCleanup(cls.sb.close)
        cls.enrolled = cls.sb.enroll_owner()
        cls.source = cls.sb.drop(cls.make_source())
        cls.source_size = cls.source.stat().st_size
        t0 = time.perf_counter()
        cls.job, cls.statuses, cls.claude_calls = run_local_job(cls.sb, cls.source)
        cls.seconds = time.perf_counter() - t0
        cls.folder = cls.job.folder
        cls.draft_text = (cls.folder / "draft.md").read_text(encoding="utf-8") if (cls.folder / "draft.md").exists() else ""

    @classmethod
    def make_source(cls) -> Path:
        raise NotImplementedError

    # -- the job

    def test_voiceprint_enrolled_in_the_sandbox(self):
        self.assertTrue(self.enrolled)
        self.assertTrue((self.sb.app / "voiceprint.json").is_file())

    def test_job_done(self):
        self.assertEqual(self.job.status, "done", self.job.error)
        self.assertIsNone(self.job.error)
        on_disk = jobs.Job.load(self.folder)
        self.assertEqual(on_disk.status, "done")
        self.assertEqual(on_disk.vault, "Local")
        self.assertEqual(on_disk.kind, "file")
        self.assertTrue(self.folder.is_relative_to(self.sb.out / "Recordings"))

    def test_outputs_exist(self):
        for name in ("job.json", "audio.flac", "whisper.json", "speakers.json", "draft.md", "recording.m4a"):
            self.assertTrue((self.folder / name).is_file(), name)
        self.assertEqual([p.name for p in self.folder.glob("*.tmp*")], [])
        self.assertEqual([p.name for p in self.folder.glob("*.part*")], [])

    def test_no_enhance_step(self):
        self.assertNotIn("enhancing", self.statuses)
        self.assertEqual(self.claude_calls, [])
        self.assertFalse((self.folder / "claude.log").exists())
        self.assertFalse((self.folder / "enhance.json").exists())
        for key in ("enhance", "session_id", "transcript", "notes", "log"):
            self.assertNotIn(key, self.job.outputs)

    def test_source_left_where_it_lies(self):
        self.assertTrue(self.source.is_file())
        self.assertEqual(self.source.stat().st_size, self.source_size)
        self.assertFalse((self.folder / self.source.name).exists())

    def test_whisper(self):
        whisper = json.loads((self.folder / "whisper.json").read_text(encoding="utf-8"))
        self.assertEqual(whisper["language"], "en")
        self.assertIn(MEETING_MODEL, whisper["model"])
        self.assertAlmostEqual(whisper["duration"], FIXTURE_SECONDS, delta=0.6)
        self.assertGreater(len(whisper["segments"]), 3)
        self.assertGreater(whisper["languages"]["en"], 20.0)

    def test_speakers(self):
        speakers = json.loads((self.folder / "speakers.json").read_text(encoding="utf-8"))
        self.assertTrue(speakers["owner_matched"])
        self.assertEqual(set(speakers["names"].values()), {OWNER, "Speaker 2"})
        self.assertEqual({line["speaker"] for line in speakers["lines"]}, {OWNER, "Speaker 2"})

    def test_recording_m4a(self):
        info = video.probe(self.folder / "recording.m4a")
        self.assertTrue(info["has_audio"])
        self.assertFalse(info["has_video"])
        self.assertAlmostEqual(info["duration"], FIXTURE_SECONDS, delta=0.8)

    # -- draft.md

    def test_frontmatter(self):
        front, _ = split_draft(self.draft_text)
        self.assertEqual([k for k, _ in front], FRONTMATTER_KEYS)
        fm = dict(front)
        self.assertEqual(fm["client"], "")
        self.assertEqual(fm["people"], f"[{OWNER}, Speaker 2]")
        self.assertRegex(fm["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(fm["date"], self.job.outputs["date"])
        self.assertEqual(fm["source"], str(self.source))
        self.assertEqual(fm["speaker_source"], f"local diarisation (sherpa-onnx); {OWNER} matched by voiceprint")
        self.assertAlmostEqual(float(fm["duration_min"]), round(FIXTURE_SECONDS / 60, 1), delta=0.1)
        self.assertEqual(fm["recording_status"], "complete")
        self.assertEqual(fm["transcriber"], f"faster-whisper {MEETING_MODEL} (local, GPU), hand-corrected")
        self.assertTrue(fm["coverage"].startswith("whole recording; "), fm["coverage"])
        self.assertIn("Afrikaans and English, language detected per chunk (Afrikaans 0 s, English ", fm["coverage"])
        self.assertIn("speakers from local diarisation", fm["coverage"])
        self.assertEqual(fm["redactions"], "none")

    def test_title_intro_and_opening(self):
        _, body = split_draft(self.draft_text)
        date = dict(split_draft(self.draft_text)[0])["date"]
        self.assertEqual(body[0], "")
        self.assertEqual(body[1], f"# {self.source.stem} transcript — {date}")
        self.assertIn(draft.INTRO_NOTE, body)
        self.assertIn("## Opening", body)

    def test_spoken_lines(self):
        _, body = split_draft(self.draft_text)
        spoken = [m for m in (SPOKEN.match(line) for line in body) if m]
        self.assertGreaterEqual(len(spoken), 4)
        names = [m.group(2) for m in spoken]
        self.assertEqual(set(names), {OWNER, "Speaker 2"})
        turns = sum(1 for a, b in zip(names, names[1:]) if a != b)
        self.assertGreaterEqual(turns, 3, names)
        times = [ts_seconds(m.group(1)) for m in spoken]
        self.assertEqual(times, sorted(times))
        self.assertLessEqual(times[-1], FIXTURE_SECONDS)

    def test_fixture_words(self):
        for word in FIXTURE_WORDS:
            self.assertIn(word, self.draft_text, word)

    def test_resume_skips_every_step(self):
        """A done job re-queued (as after a crash between steps) redoes nothing: every output is kept as it is."""
        names = ("audio.flac", "whisper.json", "speakers.json", "draft.md", "recording.m4a")
        before = {n: (self.folder / n).stat().st_mtime_ns for n in names}
        job = jobs.Job.load(self.folder)
        job.status = "queued"
        t0 = time.perf_counter()
        with mock.patch.object(enhance, "run_claude", side_effect=AssertionError("Local job reached Claude")):
            job = jobs.run_pipeline(job, self.sb.settings)
        self.assertEqual(job.status, "done", job.error)
        self.assertLess(time.perf_counter() - t0, 5.0)
        self.assertEqual({n: (self.folder / n).stat().st_mtime_ns for n in names}, before)


class FileJobEndToEnd(_EndToEnd, unittest.TestCase):
    """two_speakers.wav dropped as a file job."""

    PREFIX = "scribe_e2e_audio_"

    @classmethod
    def make_source(cls) -> Path:
        return FIXTURES / "two_speakers.wav"

    def test_statuses(self):
        self.assertEqual(self.statuses, ["queued", "transcribing", "labelling", "drafting", "done"])
        self.assertFalse(self.job.outputs["video"])
        self.assertFalse((self.folder / "frames").exists())
        self.assertFalse((self.folder / "visuals.json").exists())
        self.assertNotIn("*(on screen)*", self.draft_text)


class VideoJobEndToEnd(_EndToEnd, unittest.TestCase):
    """A generated three-slide video over the same audio. describe_frames is live only when Ollama has the model."""

    PREFIX = "scribe_e2e_video_"

    @classmethod
    def setUpClass(cls):
        cls.live_vision = vision_model_installed()
        cls.described: list = []
        if not cls.live_vision and not missing_requirement():
            def fake_describe(frames, ollama_url, model, progress=None):
                cls.described = list(frames)
                return [{"t": f["t"], "text": f"slide {i + 1}"} for i, f in enumerate(frames)]

            patcher = mock.patch.object(video, "describe_frames", fake_describe)
            patcher.start()
            cls.addClassCleanup(patcher.stop)
        super().setUpClass()

    @classmethod
    def make_source(cls) -> Path:
        build = Path(tempfile.mkdtemp(prefix="scribe_e2e_build_"))
        cls.addClassCleanup(shutil.rmtree, build, True)
        return build_slide_video(build)

    def test_statuses(self):
        self.assertEqual(self.statuses, ["queued", "transcribing", "labelling", "frames", "drafting", "done"])
        self.assertTrue(self.job.outputs["video"])

    def test_keyframes_at_slide_changes(self):
        frames = sorted((self.folder / "frames").glob("frame_*.jpg"))
        self.assertGreaterEqual(len(frames), len(SLIDE_CHANGES))
        visuals = json.loads((self.folder / "visuals.json").read_text(encoding="utf-8"))
        times = [v["t"] for v in visuals]
        if not self.live_vision:
            self.assertEqual([f["t"] for f in self.described], times)
        for change in SLIDE_CHANGES:
            self.assertTrue(any(abs(t - change) <= 0.6 for t in times), (change, times))

    def test_on_screen_lines_interleaved(self):
        _, body = split_draft(self.draft_text)
        visuals = json.loads((self.folder / "visuals.json").read_text(encoding="utf-8"))
        shown = [line for line in body if ON_SCREEN.match(line)]
        self.assertEqual(len(shown), len([v for v in visuals if v["text"].strip()]))
        self.assertGreaterEqual(len(shown), 1)
        stamped = [ts_seconds(m.group(1)) for m in (SPOKEN.match(line) or ON_SCREEN.match(line) for line in body) if m]
        self.assertEqual(stamped, sorted(stamped))
        self.assertIn(f"on-screen notes from {len(shown)} keyframes", dict(split_draft(self.draft_text)[0])["coverage"])

    def test_live_descriptions_name_the_slides(self):
        if not self.live_vision:
            self.skipTest(f"{VISION_MODEL} not in `ollama list`; describe_frames was a stub")
        text = " ".join(v["text"] for v in json.loads((self.folder / "visuals.json").read_text(encoding="utf-8"))).upper()
        self.assertTrue(any(title in text for _, _, _, title in SLIDES), text[:400])


if __name__ == "__main__":
    unittest.main()
