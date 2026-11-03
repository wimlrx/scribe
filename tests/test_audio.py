"""Tests for scribe.audio: wall-clock alignment math, mixing, m4a, and real-device capture.

Device tests use the machine's default microphone and output. Nothing audible is played: the recorder's
keep-alive stream plays zeros. Every file is written under a tempfile directory.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scribe import audio
from scribe.audio import (SR, AudioError, MeetingRecorder, MicCapture, TrackClock, chunk_offset, correction,
                          frames_owed, idle_padding, mix_tracks, to_m4a)

CHUNK = 800  # 50 ms at 16 kHz
TICK = 0.1


def _have_input() -> bool:
    try:
        pa, _ = audio._portaudio()
        pa.get_default_input_device_info()
        return True
    except Exception:
        return False


def _have_loopback() -> bool:
    try:
        pa, _ = audio._portaudio()
        pa.get_default_wasapi_loopback()
        return True
    except Exception:
        return False


def _steady(start: float, end: float, latency: float = 0.005, frames: int = CHUNK,
            period: float | None = None) -> list[tuple[float, int]]:
    """Chunks of `frames` every `period` wall seconds from `start`, each arriving `latency` after its end.
    frames != period * SR models a device clock running fast or slow against the wall clock."""
    period = frames / SR if period is None else period
    out, t = [], start
    while t + period <= end + 1e-9:
        t += period
        out.append((round(t + latency, 6), frames))
    return out


def _simulate(chunks: list[tuple[float, int]], end_s: float, clock: TrackClock | None = None):
    """Drive a TrackClock like the recorder's writer loop: every TICK, place the chunks that have arrived,
    then idle-pad. Returns (clock, [(tick, written)], [(pad, skip, keep)])."""
    clock = clock or TrackClock()
    chunks = sorted(chunks)
    i, snaps, plans = 0, [], []
    tick = TICK
    while tick <= end_s + 1e-9:
        while i < len(chunks) and chunks[i][0] <= tick:
            plans.append(clock.place(*chunks[i]))
            i += 1
        clock.idle(tick)
        snaps.append((round(tick, 6), clock.written))
        tick += TICK
    while i < len(chunks):  # in flight at stop: capped at the end
        plans.append(clock.place(*chunks[i], end_s=end_s))
        i += 1
    clock.finish(end_s)
    return clock, snaps, plans


class PureFunctionTests(unittest.TestCase):
    def test_chunk_offset(self):
        self.assertEqual(chunk_offset(1.0, 15200, 800), 0)
        self.assertEqual(chunk_offset(1.0, 0, 800), SR - 800)
        self.assertEqual(chunk_offset(1.0, 16000, 800), -800)

    def test_correction_uses_least_delayed_chunk(self):
        self.assertEqual(correction([5000, 40, 3000]), 0)          # one on-time chunk: jitter, not a gap
        self.assertEqual(correction([5000, 4000, 3000]), 3000)     # all late: a gap
        self.assertEqual(correction([-3000, -2000]), -3000)        # all ahead: trim
        self.assertEqual(correction([int(0.03 * SR)]), 0)          # inside tolerance
        self.assertEqual(correction([int(0.05 * SR)]), int(0.05 * SR))   # 50 ms late is corrected

    def test_idle_padding(self):
        self.assertEqual(idle_padding(0, 0.5, None), 0)                            # too early to judge
        self.assertEqual(idle_padding(0, 3.0, None), 2 * SR)                       # never delivered
        self.assertEqual(idle_padding(SR, 3.0, 2.5), 0)                            # delivered recently
        self.assertEqual(idle_padding(SR, 5.0, 1.0), 3 * SR)                       # quiet since 1.0 s
        self.assertEqual(idle_padding(10 * SR, 5.0, 1.0), 0)                       # never negative

    def test_frames_owed(self):
        self.assertEqual(frames_owed(SR, 2.0), SR)
        self.assertEqual(frames_owed(3 * SR, 2.0), -SR)


class TrackClockTests(unittest.TestCase):
    def test_steady_stream_needs_no_correction(self):
        clock, snaps, plans = _simulate(_steady(0.0, 10.0), 10.0)
        self.assertTrue(all(p == 0 and s == 0 for p, s, _ in plans), plans[:5])
        self.assertEqual(clock.written, 10 * SR)

    def test_slow_open_is_padded_at_the_start(self):
        clock, _, plans = _simulate(_steady(0.4, 5.0), 5.0)
        self.assertAlmostEqual(plans[0][0] / SR, 0.405, delta=0.01)
        self.assertEqual(clock.written, 5 * SR)

    def test_loopback_silent_for_first_3_seconds(self):
        end = 8.0
        mic, mic_snaps, _ = _simulate(_steady(0.3, end), end)
        system, sys_snaps, plans = _simulate(_steady(3.0, end), end)
        # the silent loopback file keeps growing by wall clock while nothing arrives
        for (t, w) in sys_snaps:
            if 1.5 <= t < 3.0:
                self.assertGreaterEqual(w, int((t - audio.HOLD_S - TICK) * SR), t)
        # once it delivers, both tracks agree to within 100 ms at every tick
        for (t, a), (_, b) in zip(mic_snaps, sys_snaps):
            if t >= 3.1:
                self.assertLessEqual(abs(a - b), 0.1 * SR, (t, a, b))
        # the zeros total the 3 s of silence and both files end exactly at the end
        zeros = int(round(3.0 * SR)) - 0  # first audio starts at 3.0 s
        idle_and_gap = system.written - sum(k for _, _, k in plans)
        self.assertAlmostEqual(idle_and_gap / SR, zeros / SR, delta=0.06)
        self.assertEqual(mic.written, int(end * SR))
        self.assertEqual(system.written, int(end * SR))

    def test_callback_jitter_is_not_mistaken_for_a_gap(self):
        chunks = _steady(0.0, 10.0)
        # chunk 80 (t = 4.0 s) is delivered 200 ms late, then the backlog arrives in a burst
        late = 0.2
        for j in range(80, 84):
            t, n = chunks[j]
            chunks[j] = (max(t, chunks[80][0] + late) + (j - 80) * 0.001, n)
        _, _, plans = _simulate(chunks, 10.0)
        self.assertTrue(all(p == 0 and s == 0 for p, s, _ in plans), [p for p in plans if p[0] or p[1]])

    def test_mid_stream_dropout_is_padded(self):
        chunks = [c for c in _steady(0.0, 10.0) if not (4.0 < c[0] <= 6.0)]
        clock, snaps, _ = _simulate(chunks, 10.0)
        for t, w in snaps:
            if t >= 6.2:
                self.assertLessEqual(abs(w - t * SR), (audio.LATE_S + 0.06) * SR, (t, w))
        self.assertEqual(clock.written, 10 * SR)

    def test_clock_drift_is_held_within_tolerance(self):
        # device clock 1% fast, then 1% slow (real sound cards drift ~0.01%). A correction fires once the
        # whole window agrees, so the bound is tolerance + drift over the window + one chunk of granularity.
        bound = (max(audio.LATE_S, audio.EARLY_S) + 0.01 * audio.WINDOW_S + CHUNK / SR + 0.01) * SR
        for frames in (808, 792):
            clock, snaps, plans = _simulate(_steady(0.0, 60.0, frames=frames, period=CHUNK / SR), 60.0)
            worst = max(abs(w - t * SR) for t, w in snaps if t > 1.0)
            self.assertLessEqual(worst, bound, frames)
            self.assertTrue(any(p or s for p, s, _ in plans), frames)

    def test_two_drifting_clocks_stay_within_100_ms_of_each_other(self):
        # a mic 100 ppm fast and a loopback 100 ppm slow, with callback jitter, over 20 minutes: each track is
        # corrected towards the wall clock on its own, so the pair must never sit more than 100 ms apart.
        import random

        rng = random.Random(7)
        end = 1200.0

        def placed(ppm: float) -> list[tuple[float, int]]:
            """(wall time a chunk's audio ends, file frames written once it is placed): where it sits."""
            clock, out = TrackClock(), []
            period = CHUNK / SR * (1 - ppm * 1e-6)
            t = 0.0
            while t + period <= end:
                t += period
                clock.place(t + 0.005 + rng.uniform(0.0, 0.03), CHUNK)
                out.append((t, clock.written))
            return out

        fast, slow = placed(+100), placed(-100)
        errors_fast = [(t, w - t * SR) for t, w in fast]
        errors_slow = [(t, w - t * SR) for t, w in slow]
        j, worst = 0, 0.0
        for t, e in errors_fast:                         # misalignment: the two errors at the same wall time
            while j + 1 < len(errors_slow) and errors_slow[j + 1][0] <= t:
                j += 1
            if t > 2.0:
                worst = max(worst, abs(e - errors_slow[j][1]))
        self.assertLessEqual(worst, 0.1 * SR, worst / SR)

    def test_nothing_is_placed_past_the_end(self):
        clock = TrackClock()
        clock.place(0.05, CHUNK)
        pad, skip, keep = clock.place(1.2, CHUNK, end_s=1.0)
        self.assertEqual(clock.written, SR)
        self.assertEqual(pad + keep, SR - CHUNK)
        self.assertEqual(clock.finish(1.0), 0)


def _write_track(path: Path, x: np.ndarray) -> None:
    sf.write(str(path), x.astype(np.float32), SR, format="FLAC", subtype="PCM_16")


def _tone(seconds: float, hz: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class MixTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="scribe-audio-"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_mix_of_two_tracks(self):
        mic, system = _tone(2.0, 440, 0.2), _tone(3.0, 660, 0.5)
        _write_track(self.dir / "mic.flac", mic)
        _write_track(self.dir / "system.flac", system)
        out = mix_tracks(self.dir)
        self.assertEqual(out, self.dir / "mix.flac")
        info = sf.info(str(out))
        self.assertEqual((info.samplerate, info.channels), (SR, 1))
        self.assertEqual(info.frames, 3 * SR)
        y, _ = sf.read(str(out), dtype="float32")
        self.assertAlmostEqual(float(np.max(np.abs(y))), 10 ** (-1 / 20), delta=0.002)
        expect = system.copy()
        expect[: mic.size] += mic
        expect *= 10 ** (-1 / 20) / np.max(np.abs(expect))
        self.assertLess(float(np.max(np.abs(y - expect))), 0.002)
        self.assertFalse((self.dir / "mix.flac.part").exists())

    def test_one_track_missing(self):
        system = _tone(2.5, 300, 0.1)
        _write_track(self.dir / "system.flac", system)
        y, _ = sf.read(str(mix_tracks(self.dir)), dtype="float32")
        self.assertEqual(y.size, system.size)
        self.assertAlmostEqual(float(np.max(np.abs(y))), 10 ** (-1 / 20), delta=0.002)

    def test_one_track_empty(self):
        _write_track(self.dir / "mic.flac", np.zeros(0, dtype=np.float32))
        _write_track(self.dir / "system.flac", _tone(1.0, 300, 0.1))
        self.assertEqual(sf.info(str(mix_tracks(self.dir))).frames, SR)

    def test_silence_only_is_not_amplified(self):
        _write_track(self.dir / "mic.flac", np.zeros(SR, dtype=np.float32))
        y, _ = sf.read(str(mix_tracks(self.dir)), dtype="float32")
        self.assertEqual((y.size, float(np.max(np.abs(y)))), (SR, 0.0))

    def test_no_tracks_raises(self):
        with self.assertRaises(AudioError):
            mix_tracks(self.dir)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
    def test_crash_truncated_track_is_read(self):
        # a FLAC flushed but never closed (the recorder crashed) has no final header
        live = self.dir / "live.flac"
        f = sf.SoundFile(str(live), "w", SR, 1, format="FLAC", subtype="PCM_16")
        f.write(audio._to_int16(_tone(3.0, 500, 0.3)))
        f.flush()
        shutil.copy(live, self.dir / "mic.flac")
        f.close()
        live.unlink()
        frames = sf.info(str(mix_tracks(self.dir))).frames
        self.assertGreater(frames, 2.5 * SR)
        self.assertLessEqual(frames, 3 * SR)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
class M4aTests(unittest.TestCase):
    def test_to_m4a(self):
        d = Path(tempfile.mkdtemp(prefix="scribe-audio-"))
        try:
            src = d / "mix.flac"
            _write_track(src, _tone(2.0, 440, 0.3))
            dst = to_m4a(src, d / "out" / "recording.m4a")
            self.assertTrue(dst.exists() and dst.stat().st_size > 1000)
            self.assertFalse(dst.with_name("recording.m4a.part").exists())
            if shutil.which("ffprobe"):
                probe = subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-show_streams", str(dst)],
                                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                stream = json.loads(probe.stdout)["streams"][0]
                self.assertEqual((stream["codec_name"], stream["channels"]), ("aac", 1))
                self.assertAlmostEqual(float(stream["duration"]), 2.0, delta=0.1)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_bad_source_raises(self):
        d = Path(tempfile.mkdtemp(prefix="scribe-audio-"))
        try:
            (d / "junk.flac").write_bytes(b"not audio")
            with self.assertRaises(AudioError):
                to_m4a(d / "junk.flac", d / "x.m4a")
            self.assertFalse((d / "x.m4a").exists())
        finally:
            shutil.rmtree(d, ignore_errors=True)


class RefreshTests(unittest.TestCase):
    """_refresh_portaudio never leaves a terminated instance behind, even when the re-init fails."""

    def test_failed_reinit_drops_the_dead_instance_and_bumps_the_generation(self):
        from unittest import mock

        class Dead:
            terminated = False

            def terminate(self):
                Dead.terminated = True

        with audio._pa_lock:
            saved, saved_gen = audio._pa, audio._pa_gen
            try:
                audio._pa = Dead()
                with mock.patch.object(audio.pyaudio, "PyAudio", side_effect=OSError("Pa_Initialize")), \
                        self.assertLogs("scribe.audio", "ERROR"):
                    audio._refresh_portaudio()
                self.assertTrue(Dead.terminated)
                self.assertIsNone(audio._pa)
                self.assertEqual(audio._pa_gen, saved_gen + 1)
                sentinel = object()
                with mock.patch.object(audio.pyaudio, "PyAudio", return_value=sentinel):
                    self.assertIs(audio._portaudio()[0], sentinel)      # the next use simply tries again
            finally:
                audio._pa = saved


class RecorderLogicTests(unittest.TestCase):
    """MeetingRecorder decisions with no device: faults, back-off, silence, disk, the id snapshot."""

    @contextlib.contextmanager
    def tmpdir(self):
        d = tempfile.mkdtemp(prefix="scribe-rec-")
        self._files = []
        try:
            yield d
        finally:
            for f in self._files:
                if not f.closed:
                    f.close()
            shutil.rmtree(d, ignore_errors=True)

    def make(self, d: Path) -> MeetingRecorder:
        rec = MeetingRecorder(d)
        rec.mic = audio._Track("mic", d / "mic.flac")
        rec.system = audio._Track("system", d / "system.flac")
        rec._tracks = [rec.mic, rec.system]
        rec._t0 = time.monotonic()
        self._files += [t.file for t in rec._tracks]
        return rec

    def test_write_error_on_one_track_still_drains_the_other_and_alerts_once(self):
        from unittest import mock

        with self.tmpdir() as tmp:
            rec = self.make(Path(tmp))

            class Src:
                channels, resampler = 1, None

            chunk = np.full(CHUNK, 0.1, np.float32).tobytes()
            for tr in rec._tracks:
                tr.queue.put((0.05, chunk, Src()))
            with mock.patch.object(rec.mic.file, "write", side_effect=RuntimeError("assert written == len(data)")):
                with self.assertLogs("scribe.audio", "ERROR"):
                    rec._tick()
                    rec._tick()
            self.assertIsNotNone(rec.mic.fault)
            self.assertTrue(rec.system.queue.empty())             # the healthy track was drained
            self.assertEqual(rec.alerts(), {"fault"})
            kinds = [e["kind"] for e in rec._events]
            self.assertEqual(kinds.count("write_failed"), 1)      # logged once, not every tick

    def test_stall_reattaches_one_track_with_back_off(self):
        from unittest import mock

        with self.tmpdir() as tmp:
            rec = self.make(Path(tmp))

            class Stalled:
                stream = None

                def active(self):
                    return True

            class Healthy(Stalled):
                pass

            rec.mic.source, rec.mic.continuous, rec.mic.opened_at = Stalled(), True, 0.0
            rec.system.source, rec.system.continuous = Healthy(), False
            with mock.patch.object(rec, "_open_streams", side_effect=lambda: setattr(rec.mic, "source", Stalled())), \
                    mock.patch.object(rec, "_reopen") as full:
                for now in (4.0, 6.0, 9.5, 14.0, 20.0, 39.0):
                    rec.mic.opened_at = now - 4.0                 # stalls again after every reattach
                    rec._check_devices(now)
            full.assert_not_called()                                # no PortAudio refresh for one stalled track
            steps = [e["detail"] for e in rec._events if e["kind"] == "reattach"]
            self.assertEqual(steps, ["stalled; next try in 5 s", "stalled; next try in 10 s",
                                     "stalled; next try in 20 s"])

    def test_default_change_snapshot_is_taken_before_the_reopen(self):
        from unittest import mock

        with self.tmpdir() as tmp:
            rec = self.make(Path(tmp))
            state = {"ids": ("headset", "laptop mic")}          # the output default has just changed

            class Ep:
                def ids(self):
                    return state["ids"]

            def refresh():                                         # the hands-free mic arrives mid-reopen
                state["ids"] = ("headset", "headset mic")

            rec._endpoints, rec._ids = Ep(), ("speakers", "laptop mic")
            with mock.patch.object(audio, "_refresh_portaudio", side_effect=refresh), \
                    mock.patch.object(rec, "_open_streams"), mock.patch.object(rec, "_close_streams"):
                rec._check_devices(10.0)        # output changed: reopen, ids read before the refresh
                self.assertEqual(rec._ids, ("headset", "laptop mic"))
                rec._check_devices(20.0)        # the input flipped during that reopen: seen on the next poll
            self.assertEqual([e["kind"] for e in rec._events].count("reopen"), 2)

    def test_forgotten_notetaker_raises_silence_and_cap(self):
        from unittest import mock

        with self.tmpdir() as tmp:
            rec = self.make(Path(tmp))
            for tr in rec._tracks:
                tr.clock.idle = lambda now: 0                      # no hours of zeros written in a unit test
            with mock.patch.object(rec, "_now", return_value=audio.SILENCE_STOP_S + 1), \
                    mock.patch.object(rec, "_flush"), mock.patch.object(rec, "_check_devices"):
                rec._tick()
            self.assertIn("silence", rec.alerts())
            rec.mic.last_voice = audio.MAX_RECORDING_S
            with mock.patch.object(rec, "_now", return_value=audio.MAX_RECORDING_S + 1), \
                    mock.patch.object(rec, "_flush"), mock.patch.object(rec, "_check_devices"):
                rec._tick()
            self.assertIn("cap", rec.alerts())
            kinds = [e["kind"] for e in rec._events]
            self.assertEqual(kinds.count("auto_stop"), 2)
            self.assertIn("resume", kinds)                       # the long gap between those ticks

    def test_start_refuses_a_nearly_full_disk(self):
        from unittest import mock

        with self.tmpdir() as tmp:
            with mock.patch.object(audio, "_free_bytes", return_value=200 << 20):
                with self.assertRaises(AudioError):
                    MeetingRecorder(Path(tmp)).start()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_heartbeat_is_fsynced_before_the_rename(self):
        from unittest import mock

        with self.tmpdir() as tmp:
            rec = self.make(Path(tmp))
            order = []
            real_fsync, real_replace = audio.os.fsync, audio.os.replace
            with mock.patch.object(audio.os, "fsync", side_effect=lambda fd: (order.append("fsync"), real_fsync(fd))), \
                    mock.patch.object(audio.os, "replace",
                                      side_effect=lambda a, b: (order.append("replace"), real_replace(a, b))):
                rec._heartbeat("recording", 1.0)
            self.assertEqual(order, ["fsync", "replace"])
            self.assertEqual(json.loads((Path(tmp) / "recording.json").read_text(encoding="utf-8"))["status"],
                             "recording")


@unittest.skipUnless(_have_input(), "no input device")
class MicCaptureTests(unittest.TestCase):
    def test_start_stop_returns_float32_of_plausible_length(self):
        mic = MicCapture()
        try:
            mic.start()
            time.sleep(1.0)
            level = mic.level()
            x = mic.stop()
            self.assertEqual((x.dtype, x.ndim), (np.float32, 1))
            self.assertGreater(x.size, 0.7 * SR)
            self.assertLess(x.size, 1.5 * SR)
            self.assertTrue(0.0 <= level <= 1.0)
            self.assertTrue(np.all(np.abs(x) <= 1.0))
            # second dictation: the stream is armed, so start is fast
            t = time.perf_counter()
            mic.start()
            started = time.perf_counter() - t
            time.sleep(0.5)
            y = mic.stop()
            self.assertLess(started, 0.1)
            self.assertGreater(y.size, 0.3 * SR)
            self.assertEqual(mic.stop().size, 0)  # stop when idle is harmless
        finally:
            mic.close()


@unittest.skipUnless(_have_input() and _have_loopback(), "no microphone or no WASAPI loopback")
class MeetingRecorderTests(unittest.TestCase):
    def test_eight_second_recording(self):
        d = Path(tempfile.mkdtemp(prefix="scribe-rec-"))
        try:
            rec = MeetingRecorder(d)
            t_start = time.monotonic()
            rec.start()
            beats = []
            for _ in range(8):
                time.sleep(1.0 - ((time.monotonic() - t_start) % 1.0))
                beats.append(json.loads((d / "recording.json").read_text(encoding="utf-8")))
                levels = rec.levels()
                self.assertEqual(len(levels), 2)
                self.assertTrue(all(0.0 <= v <= 1.0 for v in levels))
            self.assertGreater(rec.elapsed(), 7.0)
            t_stop = time.monotonic()
            res = rec.stop()
            elapsed = t_stop - t_start

            self.assertEqual(set(res), {"mic", "system", "seconds", "events"})
            self.assertAlmostEqual(res["seconds"], elapsed, delta=0.15)
            lengths = []
            for key in ("mic", "system"):
                p = Path(res[key])
                self.assertTrue(p.exists(), p)
                info = sf.info(str(p))
                self.assertEqual((info.samplerate, info.channels, info.format), (SR, 1, "FLAC"))
                lengths.append(info.frames / SR)
            self.assertLessEqual(abs(lengths[0] - lengths[1]), 0.15, lengths)
            for length in lengths:
                self.assertAlmostEqual(length, elapsed, delta=0.15)

            # heartbeat: written while recording, flushed more than once, stopped at the end
            self.assertTrue(all(b["status"] == "recording" for b in beats))
            self.assertGreaterEqual(len({b["last_flush"] for b in beats}), 3)
            final = json.loads((d / "recording.json").read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "stopped")
            self.assertEqual(set(final) >= {"status", "started", "last_flush", "events"}, True)
            self.assertAlmostEqual(final["seconds"], res["seconds"], delta=0.01)
            opened = {e["track"] for e in final["events"] if e["kind"] == "open"}
            self.assertEqual(opened, {"mic", "system"}, final["events"])
            self.assertFalse(any(e["kind"] == "reopen" for e in final["events"]), final["events"])
            self.assertEqual(rec.stop(), res)  # idempotent
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_default_device_change_reopens_and_stays_aligned(self):
        d = Path(tempfile.mkdtemp(prefix="scribe-rec-"))
        mic = MicCapture()
        try:
            mic.start()
            mic.stop()  # armed on the current PortAudio instance, which the reopen will replace
            rec = MeetingRecorder(d)
            t_start = time.monotonic()
            rec.start()
            time.sleep(2.0)
            self.assertIsNotNone(rec._endpoints, "default-device watch not running")
            rec._endpoints.ids = lambda: ("other-render", "other-capture")  # as if a headset was plugged in
            time.sleep(4.0)
            t_stop = time.monotonic()
            res = rec.stop()
            elapsed = t_stop - t_start

            kinds = [(e["track"], e["kind"]) for e in res["events"]]
            reopens = [e for e in res["events"] if e["kind"] == "reopen"]
            self.assertEqual(len(reopens), 1, res["events"])
            self.assertIn("default device changed", reopens[0]["detail"])
            self.assertEqual(kinds.count(("mic", "open")), 2, kinds)
            self.assertEqual(kinds.count(("system", "open")), 2, kinds)
            lengths = [sf.info(str(res[k])).frames / SR for k in ("mic", "system")]
            self.assertLessEqual(abs(lengths[0] - lengths[1]), 0.15, lengths)
            for length in lengths:
                self.assertAlmostEqual(length, elapsed, delta=0.15)

            # the dictation mic's stream was closed by the refresh; it reopens on the next start
            mic.start()
            time.sleep(0.5)
            self.assertGreater(mic.stop().size, 0.3 * SR)
        finally:
            mic.close()
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
