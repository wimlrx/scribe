"""video.py: probe, extract_audio and keyframes against a synthetic 90 s slide video; describe_frames
mocked (no network) and, when the vision model is installed, live against Ollama."""
from __future__ import annotations

import base64
import http.server
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import requests
import soundfile as sf
from PIL import Image, ImageDraw

from scribe import config, video

FIXTURES = Path(__file__).parent / "fixtures"
FONT = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf"

# (seconds, background, title file, layout) — slide changes at 0, 20, 45 s. The second slide is shown
# twice in a row: at 32 s it is re-shown brighter, which fires the scene filter but is the same picture.
SEGMENTS = [
    (20, "0x4080D0", "a.txt", "x=120:y=120,drawbox=x=120:y=420:w=1100:h=60:color=white:t=fill,"
                              "drawbox=x=120:y=560:w=700:h=60:color=white:t=fill"),
    (12, "0x10301A", "b.txt", "x=(w-text_w)/2:y=80,drawbox=x=1200:y=600:w=140:h=380:color=white:t=fill,"
                              "drawbox=x=1450:y=360:w=140:h=620:color=white:t=fill"),
    (13, "0x70C080", "b.txt", "x=(w-text_w)/2:y=80,drawbox=x=1200:y=600:w=140:h=380:color=white:t=fill,"
                              "drawbox=x=1450:y=360:w=140:h=620:color=white:t=fill"),
    (45, "0x5A1A1A", "c.txt", "x=120:y=h-260"),
]
TITLES = {"a.txt": "QUARTERLY REVENUE", "b.txt": "COST REVIEW", "c.txt": "NEXT STEPS"}
MAX_GAP = 30.0
EXPECTED = [0.0, 20.0, 45.0, 75.0]   # three slide changes + one heartbeat on the long last slide


def ffmpeg(args: list[str], cwd: Path) -> None:
    res = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", *args], cwd=cwd,
                         capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if res.returncode != 0:
        raise RuntimeError(f"fixture build failed: {res.stderr[-800:]}")


def build_slide_video(folder: Path) -> Path:
    """90 s, 1920x1080 at 10 fps, drawtext slides, two_speakers.wav looped as the audio track."""
    shutil.copy(FONT, folder / "arial.ttf")          # relative font/text paths: no drive-colon escaping
    for name, text in TITLES.items():
        (folder / name).write_text(text, encoding="ascii")
    args, chains = [], []
    for i, (secs, colour, textfile, layout) in enumerate(SEGMENTS):
        args += ["-f", "lavfi", "-i", f"color=c={colour}:s=1920x1080:r=10:d={secs}"]
        chains.append(f"[{i}:v]drawtext=fontfile=arial.ttf:textfile={textfile}:fontsize=120:"
                      f"fontcolor=white:{layout}[s{i}]")
    n = len(SEGMENTS)
    graph = ";".join(chains) + ";" + "".join(f"[s{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    args += ["-stream_loop", "1", "-i", str(FIXTURES / "two_speakers.wav"),
             "-filter_complex", graph, "-map", "[v]", "-map", f"{n}:a", "-t", "90",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
             "slides.mp4"]
    ffmpeg(args, folder)
    return folder / "slides.mp4"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def vision_model_installed(model: str) -> tuple[bool, str]:
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"ollama list failed: {exc!r}"
    if res.returncode != 0:
        return False, f"ollama list failed: {res.stderr.strip()[:200]}"
    names = {line.split()[0] for line in res.stdout.splitlines()[1:] if line.strip()}
    return (model in names), f"{model} not in `ollama list` (have: {', '.join(sorted(names)) or 'none'})"


class MediaFileTests(unittest.TestCase):
    """Real ffmpeg runs on generated files; everything under one temp dir."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="scribe_video_"))
        cls.video = build_slide_video(cls.tmp)
        cls.log = _Records()
        logger = logging.getLogger("scribe.video")
        cls._level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(cls.log)
        try:
            cls.frames = video.keyframes(cls.video, cls.tmp / "frames", max_gap=MAX_GAP)
        finally:
            logger.removeHandler(cls.log)
            logger.setLevel(cls._level)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # probe -------------------------------------------------------------------------------------
    def test_probe_video(self) -> None:
        info = video.probe(self.video)
        self.assertEqual(set(info), {"duration", "has_video", "has_audio"})
        self.assertAlmostEqual(info["duration"], 90.0, delta=0.5)
        self.assertTrue(info["has_video"])
        self.assertTrue(info["has_audio"])

    def test_probe_audio_only(self) -> None:
        info = video.probe(FIXTURES / "two_speakers.wav")
        self.assertFalse(info["has_video"])
        self.assertTrue(info["has_audio"])
        self.assertAlmostEqual(info["duration"], 46.41, delta=0.1)

    def test_probe_cover_art_is_not_video(self) -> None:
        d = self.tmp / "cover"
        d.mkdir(exist_ok=True)
        Image.new("RGB", (64, 64), (0, 0, 200)).save(d / "cover.png")
        ffmpeg(["-i", str(FIXTURES / "short_dictation.wav"), "-i", "cover.png", "-map", "0:a", "-map", "1:v",
                "-c:a", "aac", "-c:v", "png", "-disposition:v", "attached_pic", "song.m4a"], d)
        info = video.probe(d / "song.m4a")
        self.assertFalse(info["has_video"])
        self.assertTrue(info["has_audio"])
        self.assertEqual(video.keyframes(d / "song.m4a", d / "frames"), [])

    def test_probe_scans_when_header_has_no_duration(self) -> None:
        d = self.tmp / "live"
        d.mkdir(exist_ok=True)
        ffmpeg(["-f", "lavfi", "-i", "color=c=red:s=320x240:r=10:d=6", "-i", str(FIXTURES / "short_dictation.wav"),
                "-map", "0:v", "-map", "1:a", "-t", "6", "-c:v", "libvpx", "-deadline", "realtime",
                "-c:a", "libopus", "-live", "1", "browser.webm"], d)
        info = video.probe(d / "browser.webm")
        self.assertAlmostEqual(info["duration"], 6.0, delta=0.5)
        self.assertTrue(info["has_video"])

    def test_probe_rejects_non_media(self) -> None:
        bad = self.tmp / "notes.mp4"
        bad.write_text("not a video", encoding="ascii")
        with self.assertRaises(video.VideoError):
            video.probe(bad)

    # extract_audio -----------------------------------------------------------------------------
    def test_extract_audio_16k_mono_full_length(self) -> None:
        out = video.extract_audio(self.video, self.tmp / "audio" / "audio.flac")
        info = sf.info(str(out))
        self.assertEqual((info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16"))
        self.assertAlmostEqual(info.duration, video.probe(self.video)["duration"], delta=0.5)
        self.assertEqual([p.name for p in out.parent.iterdir()], ["audio.flac"])   # no .part left behind

    def test_extract_audio_pads_a_late_track_to_the_file_clock(self) -> None:
        d = self.tmp / "late"
        d.mkdir(exist_ok=True)
        ffmpeg(["-f", "lavfi", "-i", "color=c=green:s=320x240:r=10:d=10", "-itsoffset", "3",
                "-i", str(FIXTURES / "short_dictation.wav"), "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "late.mp4"], d)
        speech = sf.info(str(FIXTURES / "short_dictation.wav")).duration
        audio, sr = sf.read(str(video.extract_audio(d / "late.mp4", d / "late.flac")))
        self.assertAlmostEqual(len(audio) / sr, 3.0 + speech, delta=0.5)
        self.assertLess(float(np.abs(audio[: int(2.5 * sr)]).max()), 1e-3)   # the 3 s offset is silence

    def test_extract_audio_without_audio_raises(self) -> None:
        d = self.tmp / "mute"
        d.mkdir(exist_ok=True)
        ffmpeg(["-f", "lavfi", "-i", "color=c=blue:s=320x240:r=10:d=3", "-c:v", "libx264", "-preset",
                "ultrafast", "mute.mp4"], d)
        with self.assertRaises(video.VideoError):
            video.extract_audio(d / "mute.mp4", d / "mute.flac")
        self.assertFalse((d / "mute.flac").exists())
        self.assertFalse((d / "mute.part.flac").exists())

    # keyframes ---------------------------------------------------------------------------------
    def test_keyframes_one_per_slide_change_plus_heartbeat(self) -> None:
        times = [f["t"] for f in self.frames]
        self.assertEqual(len(times), len(EXPECTED), times)
        for got, want in zip(times, EXPECTED):
            self.assertAlmostEqual(got, want, delta=1.0)

    def test_keyframes_timestamps_match_files(self) -> None:
        # dominant channel of each kept frame: blue slide, dark green slide, maroon slide (twice)
        dominant = []
        for f in self.frames:
            with Image.open(f["file"]) as im:
                dominant.append(int(np.asarray(im.convert("RGB")).reshape(-1, 3).mean(0).argmax()))
        self.assertEqual(dominant, [2, 1, 0, 0])

    def test_keyframes_reshown_slide_dropped_and_deleted(self) -> None:
        drops = [float(m.group(1)) for r in self.log.records
                 if (m := re.search(r"drop near-duplicate t=([\d.]+)", r.getMessage()))]
        self.assertEqual(len(drops), 1, "the brighter re-show at 32 s should reach dedupe exactly once")
        self.assertAlmostEqual(drops[0], 32.0, delta=1.0)
        self.assertFalse(any(abs(f["t"] - 32.0) < 3 for f in self.frames))
        on_disk = sorted(p.name for p in (self.tmp / "frames").glob("*.jpg"))
        self.assertEqual(on_disk, sorted(Path(f["file"]).name for f in self.frames))

    def test_keyframes_width_capped_at_1280(self) -> None:
        for f in self.frames:
            with Image.open(f["file"]) as im:
                self.assertEqual(im.size, (1280, 720))

    def test_keyframes_max_frames_thins_evenly_and_clears_stale(self) -> None:
        out = self.tmp / "thin"
        out.mkdir(exist_ok=True)
        Image.new("RGB", (8, 8)).save(out / "frame_09999.jpg")       # left over from an earlier run
        frames = video.keyframes(self.video, out, max_gap=5.0, max_frames=6)
        times = [f["t"] for f in frames]
        self.assertEqual(len(frames), 6)
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times, sorted(times))
        self.assertGreaterEqual(times[-1], 80.0)
        self.assertEqual(len(list(out.glob("*.jpg"))), 6)

    def test_keyframes_audio_only_is_empty(self) -> None:
        out = self.tmp / "noframes"
        self.assertEqual(video.keyframes(FIXTURES / "two_speakers.wav", out), [])
        self.assertFalse(out.exists())

    # describe_frames, live -----------------------------------------------------------------------
    def test_describe_frames_live(self) -> None:
        model = config.DEFAULTS["ollama"]["vision_model"]
        ok, why = vision_model_installed(model)
        if not ok:
            self.skipTest(why)
        picks = [self.frames[0], self.frames[2]]       # QUARTERLY REVENUE, NEXT STEPS
        out = video.describe_frames(picks, config.DEFAULTS["ollama"]["url"], model)
        self.assertEqual([o["t"] for o in out], [p["t"] for p in picks])
        self.assertIn("QUARTERLY REVENUE", out[0]["text"].upper())
        self.assertIn("NEXT STEPS", out[1]["text"].upper())


class DedupeTests(unittest.TestCase):
    """dHash and the dedupe/thinning rules on Pillow-drawn frames; no ffmpeg."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="scribe_dhash_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def slide(self, name: str, bg: tuple, blocks: list[tuple]) -> Path:
        im = Image.new("RGB", (1280, 720), bg)
        d = ImageDraw.Draw(im)
        for box in blocks:
            d.rectangle(box, fill=(255, 255, 255))
        p = self.tmp / name
        im.save(p, quality=90)
        return p

    def test_dhash_ignores_brightness_but_sees_layout(self) -> None:
        layout = [(100, 80, 900, 160), (100, 300, 700, 340), (100, 400, 500, 440)]
        dark = video.dhash(self.slide("dark.jpg", (20, 40, 30), layout))
        bright = video.dhash(self.slide("bright.jpg", (110, 190, 130), layout))
        other = video.dhash(self.slide("other.jpg", (20, 40, 30), [(800, 300, 900, 700), (1000, 150, 1100, 700)]))
        self.assertEqual(dark.shape, (8, 8))
        self.assertLessEqual(video.hamming(dark, bright), video.DHASH_MAX_DISTANCE)
        self.assertGreater(video.hamming(dark, other), video.DHASH_MAX_DISTANCE)

    def test_dedupe_drops_repeat_but_keeps_heartbeat(self) -> None:
        same = [(100, 80, 900, 160)]
        frames = [{"t": t, "file": self.slide(f"f{t}.jpg", (30, 30, 30), same)} for t in (0.0, 10.0, 30.0, 45.0)]
        kept = video._dedupe(frames, max_gap=30.0)
        self.assertEqual([f["t"] for f in kept], [0.0, 30.0])      # 10 and 45 repeat within max_gap
        self.assertFalse(frames[1]["file"].exists())
        self.assertFalse(frames[3]["file"].exists())

    def test_thin_keeps_first_and_spreads(self) -> None:
        frames = [{"t": float(i), "file": self.slide(f"t{i}.jpg", (0, 0, 0), [])} for i in range(10)]
        kept = video._thin(frames, 4)
        self.assertEqual([f["t"] for f in kept], [0.0, 3.0, 6.0, 9.0])
        self.assertEqual(len(list(self.tmp.glob("t*.jpg"))), 4)


class _Resp:
    def __init__(self, content: str = "", status: int = 200) -> None:
        self.status_code = status
        self._content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self) -> dict:
        return {"model": "m", "message": {"role": "assistant", "content": self._content}, "done": True}


class DescribeMockTests(unittest.TestCase):
    """describe_frames against a fake requests.post: body shape, NONE drop, failures, unload."""

    URL = "http://127.0.0.1:11434/"
    MODEL = "qwen3.6:27b-q4_K_M"

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="scribe_describe_"))
        self.frames = []
        for i, t in enumerate((0.0, 20.0, 45.0)):
            p = self.tmp / f"frame_{i:05d}.jpg"
            Image.new("RGB", (32, 18), (i * 60, 0, 0)).save(p)
            self.frames.append({"t": t, "file": p})

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_with(self, replies: list, frames=None, progress=None):
        """replies: one item per chat call (str, _Resp or exception); the unload call always succeeds."""
        queue = list(replies)

        def fake_post(url, json=None, timeout=None, proxies=None):
            if json.get("keep_alive") == 0:
                return _Resp()
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item if isinstance(item, _Resp) else _Resp(item)

        with mock.patch.object(video.requests, "post", side_effect=fake_post) as post:
            try:
                out = video.describe_frames(self.frames if frames is None else frames, self.URL, self.MODEL,
                                            progress=progress)
            except Exception as exc:          # returned so tests can inspect calls too
                out = exc
        return out, post.call_args_list

    def test_request_body_shape(self) -> None:
        _, calls = self.run_with(["Slide one", "Slide two", "Slide three"])
        url, body = calls[0].args[0], calls[0].kwargs["json"]
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(set(body), {"model", "messages", "stream", "think", "options"})
        self.assertEqual(body["model"], self.MODEL)
        self.assertIs(body["stream"], False)
        self.assertIs(body["think"], False)
        self.assertEqual(body["options"], {"num_ctx": 8192, "temperature": 0})
        (msg,) = body["messages"]
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["images"], [base64.b64encode(self.frames[0]["file"].read_bytes()).decode("ascii")])
        for word in ("NONE", "title", "number", "table", "chart"):
            self.assertIn(word, msg["content"])

    def test_none_frames_dropped_and_model_unloaded(self) -> None:
        seen = []
        out, calls = self.run_with(["NONE", "<think>hmm</think>COST REVIEW\nR 2.4m", "NONE."], progress=seen.append)
        self.assertEqual(out, [{"t": 20.0, "text": "COST REVIEW\nR 2.4m"}])
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[-1].kwargs["json"], {"model": self.MODEL, "messages": [], "keep_alive": 0})
        self.assertEqual(calls[-1].args[0], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(seen, [1 / 3, 2 / 3, 1.0])

    def test_failing_frame_retried_then_skipped(self) -> None:
        out, calls = self.run_with([requests.ReadTimeout("slow"), requests.ReadTimeout("slow"),
                                    "Slide two", _Resp(status=500), "Slide three"])
        self.assertEqual(out, [{"t": 20.0, "text": "Slide two"}, {"t": 45.0, "text": "Slide three"}])
        self.assertEqual(calls[-1].kwargs["json"]["keep_alive"], 0)

    def test_unreachable_server_raises_and_still_unloads(self) -> None:
        out, calls = self.run_with([requests.ConnectionError("refused")])
        self.assertIsInstance(out, video.VideoError)
        self.assertEqual(calls[-1].kwargs["json"]["keep_alive"], 0)

    def test_missing_model_raises(self) -> None:
        out, _ = self.run_with([_Resp(status=404)])
        self.assertIsInstance(out, video.VideoError)
        self.assertIn("not found", str(out))

    def test_all_frames_failing_raises(self) -> None:
        out, _ = self.run_with([_Resp(status=500)] * 6)
        self.assertIsInstance(out, video.VideoError)

    def test_no_frames_no_calls(self) -> None:
        out, calls = self.run_with([], frames=[])
        self.assertEqual(out, [])
        self.assertEqual(calls, [])

    def test_is_none(self) -> None:
        for text in ("NONE", "NONE.", " none ", "**NONE**", "NONE - camera feed only", ""):
            self.assertTrue(video._is_none(text), text)
        for text in ("None of the figures are final", "Agenda: NONE of the above", "Q3 revenue R 2.4m"):
            self.assertFalse(video._is_none(text), text)

    def test_every_call_refuses_proxies(self) -> None:
        """privacy-failure F6: the chat and the unload both say no proxy, for http, https and all."""
        _, calls = self.run_with(["Slide one", "Slide two", "Slide three"])
        self.assertEqual(len(calls), 4)
        for call in calls:
            self.assertEqual(call.kwargs.get("proxies"), {"http": None, "https": None, "all": None})


class _Server:
    """A local HTTP server that answers every POST like Ollama's /api/chat and records the request lines."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        rec = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                rec.lines.append(self.requestline)
                data = json.dumps({"message": {"role": "assistant", "content": "Slide"}, "done": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class DescribeProxyTests(unittest.TestCase):
    """privacy-failure F6: screen frames go only to the local server, never to a proxy set in HTTP_PROXY,
    ALL_PROXY or the Windows settings."""

    def test_proxy_never_sees_a_frame(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="scribe_proxy_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        frame = tmp / "frame_00000.jpg"
        Image.new("RGB", (32, 18), (90, 0, 0)).save(frame)
        ollama, proxy = _Server(), _Server()
        self.addCleanup(ollama.close)
        self.addCleanup(proxy.close)
        clean = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
        system = mock.patch.object(requests.utils, "getproxies", lambda: {"http": proxy.url, "https": proxy.url})
        for name, env, patcher in (("HTTP_PROXY", {"HTTP_PROXY": proxy.url}, None),
                                   ("ALL_PROXY", {"ALL_PROXY": proxy.url}, None),
                                   ("Windows", {}, system)):
            with self.subTest(name), mock.patch.dict(os.environ, {**clean, **env}, clear=True):
                if patcher:
                    patcher.start()
                try:
                    out = video.describe_frames([{"t": 0.0, "file": frame}], ollama.url, "m")
                finally:
                    if patcher:
                        patcher.stop()
                self.assertEqual(out, [{"t": 0.0, "text": "Slide"}])
                self.assertEqual(proxy.lines, [])
        self.assertEqual(len(ollama.lines), 6)          # a chat and an unload each time

    def test_the_setting_is_never_changed(self) -> None:
        """requests adds the environment's other proxies to the dict it is given: each call gets a copy."""
        saved = dict(video.NO_PROXY)
        self.addCleanup(lambda: (video.NO_PROXY.clear(), video.NO_PROXY.update(saved)))
        tmp = Path(tempfile.mkdtemp(prefix="scribe_proxy_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        frame = tmp / "frame_00000.jpg"
        Image.new("RGB", (32, 18), (90, 0, 0)).save(frame)
        ollama = _Server()
        self.addCleanup(ollama.close)
        clean = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
        with mock.patch.dict(os.environ, {**clean, "FTP_PROXY": "http://127.0.0.1:9", "NO_PROXY": "localhost"},
                             clear=True):
            video.describe_frames([{"t": 0.0, "file": frame}], ollama.url, "m")
        self.assertEqual(video.NO_PROXY, {"http": None, "https": None, "all": None})


if __name__ == "__main__":
    unittest.main()
