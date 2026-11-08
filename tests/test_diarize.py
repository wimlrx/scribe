"""diarize: sherpa-onnx diarisation on the fixtures, voiceprint enrolment and naming, word assignment on synthetic data.
Model tests skip when the sherpa models are not in %USERPROFILE%\\.scribe\\models. Writes only under tempfile dirs."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scribe import config, diarize  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
MODELS = config.APP_DIR / "models"   # read only; config.models_dir() would mkdir
SR = diarize.SAMPLE_RATE
OWNER = "Alex Morgan"


def models_present() -> bool:
    return (MODELS / diarize.SEGMENTATION_MODEL).is_file() and (MODELS / diarize.EMBEDDING_MODEL).is_file()


def write_wav(path: Path, audio: np.ndarray, rate: int = SR) -> Path:
    import soundfile as sf

    sf.write(str(path), audio, rate, subtype="PCM_16")
    return path


def speaker_at(diar: list[dict], t: float) -> int:
    return next(s["speaker"] for s in diar if s["start"] <= t <= s["end"])


def enroll_owner(vp, owner: np.ndarray) -> None:
    """Three 5 s clips of the 15 s enrolment fixture: the first three candidates agree, so they commit."""
    for k in range(3):
        assert vp.enroll(owner[k * 5 * SR:(k + 1) * 5 * SR])
    assert vp.enrolled and vp.count == 3, (vp.enrolled, vp.count)


def buzz(f0: float, seconds: float, level: float = 0.3) -> np.ndarray:
    """A voice-like buzz at f0 Hz (11 harmonics, 3% vibrato, a syllable envelope): a pitch, and nobody's voice."""
    t = np.arange(int(round(seconds * SR))) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))) / SR
    wave = sum(np.sin(k * phase) / k for k in range(1, 12))
    return (level * wave / np.max(np.abs(wave)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2)).astype(np.float32)


def hiss(seconds: float, level: float = 0.001, seed: int = 0) -> np.ndarray:
    """Noise: at 0.001 a pause, at 0.1 a stretch with no pitch (a fricative, a breath)."""
    return (np.random.default_rng(seed).standard_normal(int(round(seconds * SR))) * level).astype(np.float32)


def timeline(*pieces: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """The pieces end to end, and each one's (start, end) in seconds."""
    spans, t = [], 0.0
    for piece in pieces:
        spans.append((round(t, 3), round(t + piece.size / SR, 3)))
        t += piece.size / SR
    return np.concatenate(pieces), spans


def span(start: float, end: float, speaker: int) -> dict:
    return {"start": start, "end": end, "speaker": speaker}


def triples(diar: list[dict]) -> list[tuple]:
    return [(s["start"], s["end"], s["speaker"]) for s in diar]


@unittest.skipUnless(models_present(), "sherpa-onnx models not in %USERPROFILE%\\.scribe\\models")
class DiarizeFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.two_path = FIXTURES / "two_speakers.wav"
        cls.two = diarize.read_audio(cls.two_path)
        cls.owner = diarize.read_audio(FIXTURES / "owner_david_enroll.wav")
        cls.diar = diarize.diarize(cls.two_path, MODELS, threshold=0.5)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # -- diarize

    def test_two_speakers_at_default_threshold(self):
        self.assertEqual({s["speaker"] for s in self.diar}, {0, 1})
        self.assertEqual(self.diar[0]["speaker"], 0, "numbered by first appearance")
        self.assertEqual(self.diar, sorted(self.diar, key=lambda s: s["start"]))
        turns = sum(1 for a, b in zip(self.diar, self.diar[1:]) if a["speaker"] != b["speaker"])
        self.assertGreaterEqual(turns, 5, "the fixture alternates voices about every 6 s")
        speech = sum(s["end"] - s["start"] for s in self.diar)
        self.assertGreater(speech, 30.0)
        for s in self.diar:
            self.assertEqual(set(s), {"start", "end", "speaker"})
            self.assertLess(s["start"], s["end"])

    def test_threshold_band(self):
        """Calibration: two clusters across 0.3-0.7; over-splits at 0.1, merges at 0.9. Default 0.5 sits mid-band."""
        for th in (0.3, 0.7):
            self.assertEqual(len({s["speaker"] for s in diarize.diarize(self.two_path, MODELS, threshold=th)}), 2, th)
        self.assertGreater(len({s["speaker"] for s in diarize.diarize(self.two_path, MODELS, threshold=0.1)}), 2)
        self.assertEqual(len({s["speaker"] for s in diarize.diarize(self.two_path, MODELS, threshold=0.9)}), 1)

    def test_fixed_speaker_count(self):
        diar = diarize.diarize(self.two_path, MODELS, threshold=0.9, num_speakers=2)
        self.assertEqual({s["speaker"] for s in diar}, {0, 1})

    def test_single_voice_is_cluster_zero(self):
        diar = diarize.diarize(FIXTURES / "short_dictation.wav", MODELS)
        self.assertTrue(diar)
        self.assertEqual({s["speaker"] for s in diar}, {0})

    def test_silence_and_empty(self):
        self.assertEqual(diarize.diarize(write_wav(self.dir / "silence.wav", np.zeros(SR * 5, np.float32)), MODELS), [])
        self.assertEqual(diarize.diarize(write_wav(self.dir / "blip.wav", np.zeros(SR // 10, np.float32)), MODELS), [])

    def test_missing_models(self):
        with self.assertRaises(FileNotFoundError):
            diarize.diarize(self.two_path, self.dir)

    # -- voiceprint

    def other_voice(self) -> np.ndarray:
        """About 10 s of the fixture's second voice (the cluster that is not David)."""
        david = speaker_at(self.diar, 3.0)
        parts = [self.two[int(s["start"] * SR):int(s["end"] * SR)] for s in self.diar if s["speaker"] != david]
        return np.concatenate(parts)[: 10 * SR]

    def test_enroll_persists_running_mean(self):
        path = self.dir / "voiceprint.json"
        vp = diarize.Voiceprint(path, MODELS)
        self.assertFalse(vp.enrolled)
        self.assertEqual(vp.similarity(np.ones(256, np.float32)), -1.0)
        self.assertTrue(vp.enroll(self.owner[: 5 * SR]))
        self.assertTrue(vp.enroll(self.owner[5 * SR: 10 * SR]))
        self.assertFalse(vp.enrolled, "two clips are candidates, not yet a voiceprint")
        self.assertEqual(vp.pending, 2)
        self.assertTrue(vp.enroll(self.owner[10 * SR:]))
        self.assertTrue(vp.enrolled)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["count"], 3)
        self.assertNotIn("pending", data)
        self.assertEqual(len(data["embedding"]), 256)
        self.assertTrue(all(isinstance(x, float) for x in data["embedding"]))

        other = diarize.Voiceprint(path, MODELS)       # a second instance reads the file
        self.assertTrue(other.enrolled)
        self.assertEqual(other.count, 3)
        self.assertTrue(other.enroll(self.owner[: SR * 6]))
        self.assertEqual(vp.count, 4, "the first instance sees the other's enrolment")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["count"], 4)
        self.assertGreater(vp.similarity(vp.embed(self.owner)), 0.9)

    def test_another_voice_never_joins_the_voiceprint(self):
        vp = diarize.Voiceprint(self.dir / "voiceprint.json", MODELS)
        enroll_owner(vp, self.owner)
        before = vp.similarity(vp.embed(self.owner))
        with self.assertLogs("scribe.diarize", "WARNING"):
            self.assertFalse(vp.enroll(self.other_voice()))
        self.assertEqual(vp.count, 3)
        self.assertAlmostEqual(vp.similarity(vp.embed(self.owner)), before, places=4)

    def test_a_foreign_first_clip_is_dropped_as_the_outlier(self):
        vp = diarize.Voiceprint(self.dir / "voiceprint.json", MODELS)
        self.assertTrue(vp.enroll(self.other_voice()))          # someone else used Ctrl+Win first
        self.assertFalse(vp.enrolled)
        enroll_owner(vp, self.owner)
        self.assertGreater(vp.similarity(vp.embed(self.owner)), 0.9)
        self.assertLess(vp.similarity(vp.embed(self.other_voice())), diarize.ENROLL_AGREE)

    def test_enroll_needs_two_seconds_of_speech(self):
        vp = diarize.Voiceprint(self.dir / "voiceprint.json", MODELS)
        self.assertFalse(vp.enroll(self.owner[: int(SR * 1.5)]))
        self.assertFalse(vp.enroll(np.zeros(SR * 5, np.float32)))
        self.assertFalse(vp.enroll(np.zeros(0, np.float32)))
        self.assertFalse(vp.enrolled)
        self.assertFalse((self.dir / "voiceprint.json").exists())

    def test_unreadable_or_foreign_voiceprint_is_not_enrolled(self):
        bad = self.dir / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        self.assertFalse(diarize.Voiceprint(bad, MODELS).enrolled)
        foreign = self.dir / "foreign.json"
        foreign.write_text(json.dumps({"model": "other.onnx", "count": 3, "embedding": [0.1] * 256}), encoding="utf-8")
        self.assertFalse(diarize.Voiceprint(foreign, MODELS).enrolled)

    # -- naming

    def test_enrolled_owner_names_david_cluster(self):
        vp = diarize.Voiceprint(self.dir / "voiceprint.json", MODELS)
        enroll_owner(vp, self.owner)
        david = speaker_at(self.diar, 3.0)           # the fixture opens with the David voice
        self.assertGreater(vp.similarity(vp.embed(self.two[int(1.6 * SR):int(5.8 * SR)])), 0.8)
        names = diarize.name_speakers(self.two, self.diar, vp, OWNER)
        self.assertEqual(names[david], OWNER)
        self.assertEqual(names[1 - david], "Speaker 2")
        self.assertEqual(len(names), 2)

    def test_no_voiceprint_numbers_speakers(self):
        self.assertEqual(diarize.name_speakers(self.two, self.diar, None, OWNER), {0: "Speaker 1", 1: "Speaker 2"})
        empty = diarize.Voiceprint(self.dir / "none.json", MODELS)
        self.assertEqual(diarize.name_speakers(self.two, self.diar, empty, OWNER), {0: "Speaker 1", 1: "Speaker 2"})

    def test_owner_below_match_threshold(self):
        vp = diarize.Voiceprint(self.dir / "voiceprint.json", MODELS)
        enroll_owner(vp, self.owner)
        names = diarize.name_speakers(self.two, self.diar, vp, OWNER, owner_match=0.999)
        self.assertEqual(names, {0: "Speaker 1", 1: "Speaker 2"})


@unittest.skipUnless(models_present(), "sherpa-onnx models not in %USERPROFILE%\\.scribe\\models")
class LabelByPitch(unittest.TestCase):
    """The labelling step parts voices that sherpa merged, by pitch. The fixture's two voices sit near 90 Hz and
    180 Hz; at threshold 0.9 sherpa puts them in one cluster, as it put a man and a woman on one laptop microphone
    in one cluster at 0.5 (31 Oct)."""

    @classmethod
    def setUpClass(cls):
        cls.two_path = FIXTURES / "two_speakers.wav"
        cls.two = diarize.read_audio(cls.two_path)
        cls.owner = diarize.read_audio(FIXTURES / "owner_david_enroll.wav")
        cls.diar = diarize.diarize(cls.two_path, MODELS, threshold=0.5)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def label(self, path: Path, threshold: float, voiceprint_path: Path | None = None, segments=()) -> dict:
        return diarize.label(path, list(segments), MODELS, threshold=threshold, owner_name=OWNER, owner_match=0.6,
                             voiceprint_path=voiceprint_path)

    def enrolled(self) -> Path:
        path = self.dir / "voiceprint.json"
        enroll_owner(diarize.Voiceprint(path, MODELS), self.owner)
        return path

    def clip(self, start: float, end: float) -> np.ndarray:
        return self.two[int(start * SR):int(end * SR)]

    def reply_after_a_pause(self) -> tuple[Path, list[tuple[float, float]]]:
        """David, the other voice, David again, a 0.4 s pause and 0.86 s of the other voice (186 Hz against her
        181 Hz register, as the 31 Oct reply read 184 Hz against 188 Hz): the 31 Oct shape."""
        audio, spans = timeline(self.clip(1.58, 5.82), hiss(0.6), self.clip(8.64, 13.58), hiss(0.6),
                                self.clip(27.57, 32.31), hiss(0.4, seed=1), self.clip(37.31, 38.17), hiss(0.3, seed=2))
        return write_wav(self.dir / "reply.wav", audio), spans

    def words_in(self, spans: list[tuple[float, float]]) -> list[dict]:
        """One made-up word in the middle of each of the four voiced pieces."""
        mids = [(a + b) / 2 for a, b in (spans[0], spans[2], spans[4], spans[6])]
        return [seg([(m - 0.1, m + 0.1, f"w{k}")]) for k, m in enumerate(mids)]

    def test_voices_sherpa_merged_are_parted(self):
        merged = diarize.diarize(self.two_path, MODELS, threshold=0.9)
        self.assertEqual({s["speaker"] for s in merged}, {0}, "sherpa merges both voices at 0.9")
        out = self.label(self.two_path, 0.9)
        self.assertEqual(triples(out["diarization"]), triples(self.diar), "the same two clusters as at 0.5")
        self.assertEqual(out["names"], {"0": "Speaker 1", "1": "Speaker 2"})

    def test_the_owner_is_named_once_the_voices_are_parted(self):
        out = self.label(self.two_path, 0.9, self.enrolled())
        david = speaker_at(out["diarization"], 3.0)
        self.assertTrue(out["owner_matched"])
        self.assertEqual(out["names"], {str(david): OWNER, str(1 - david): "Speaker 2"})

    def test_a_reply_after_a_pause_goes_to_its_speaker(self):
        """sherpa at 0.4 on a recorded meeting: the woman's first turn its own cluster, her short reply inside his segment."""
        path, spans = self.reply_after_a_pause()
        merged = [span(spans[0][0], spans[0][1], 0), span(spans[2][0], spans[2][1], 1),
                  span(spans[4][0], spans[6][1], 0)]
        with unittest.mock.patch.object(diarize, "diarize", return_value=merged):
            out = self.label(path, 0.4, self.enrolled(), self.words_in(spans))
        self.assertEqual([line["speaker"] for line in out["lines"]], [OWNER, "Speaker 2", OWNER, "Speaker 2"])
        cut = next(s["start"] for s in out["diarization"] if s["start"] > spans[4][0])
        self.assertTrue(spans[5][0] < cut < spans[5][1], f"cut at {cut}, inside the pause {spans[5]}")

    def test_a_fragment_of_the_other_voice_shares_its_label(self):
        """sherpa at 0.3 on a recorded meeting: the short reply came back as a third cluster of its own."""
        path, spans = self.reply_after_a_pause()
        split = [span(spans[0][0], spans[0][1], 0), span(spans[2][0], spans[2][1], 1),
                 span(spans[4][0], spans[4][1], 0), span(spans[6][0], spans[6][1], 2)]
        with unittest.mock.patch.object(diarize, "diarize", return_value=split):
            out = self.label(path, 0.3, self.enrolled(), self.words_in(spans))
        self.assertEqual([line["speaker"] for line in out["lines"]], [OWNER, "Speaker 2", OWNER, "Speaker 2"])
        self.assertEqual(out["names"], {"0": OWNER, "1": "Speaker 2", "2": "Speaker 2"})

    def test_one_voice_stays_one_speaker(self):
        for path, threshold in ((FIXTURES / "owner_david_enroll.wav", 0.3), (FIXTURES / "owner_david_enroll.wav", 0.9),
                                (FIXTURES / "short_dictation.wav", 0.5)):
            out = self.label(path, threshold)
            self.assertEqual(set(out["names"].values()), {"Speaker 1"}, (path.name, threshold))
            self.assertEqual(out["diarization"], diarize.diarize(path, MODELS, threshold=threshold))

    def test_a_failed_refinement_keeps_sherpa_clusters(self):
        with unittest.mock.patch.object(diarize, "refine", side_effect=RuntimeError("boom")), \
                self.assertLogs("scribe.diarize", "WARNING"):
            out = self.label(self.two_path, 0.9)
        self.assertEqual({s["speaker"] for s in out["diarization"]}, {0})
        self.assertEqual(out["names"], {"0": "Speaker 1"})


class OwnerClusters(unittest.TestCase):
    def test_split_owner_clusters_are_all_named(self):
        # the 60 min probe: one voice split into clusters 0 (0.93) and 2 (0.94), the other voice 0.41
        self.assertEqual(diarize.owner_clusters({0: 0.93, 1: 0.41, 2: 0.94}, 0.45), {0, 2})

    def test_band_is_relative_to_the_best(self):
        self.assertEqual(diarize.owner_clusters({0: 0.95, 1: 0.80}, 0.45), {0})

    def test_weak_match_needs_a_margin(self):
        self.assertEqual(diarize.owner_clusters({0: 0.50, 1: 0.41}, 0.45), set())   # a stranger just over the line
        self.assertEqual(diarize.owner_clusters({0: 0.66, 1: 0.41}, 0.45), {0})
        self.assertEqual(diarize.owner_clusters({0: 0.74, 1: 0.60}, 0.45), {0})     # clear match: no margin needed

    def test_below_threshold_or_empty(self):
        self.assertEqual(diarize.owner_clusters({0: 0.44}, 0.45), set())
        self.assertEqual(diarize.owner_clusters({}, 0.45), set())

    def test_names_with_two_owner_clusters(self):
        class VP:
            enrolled = True

            def embed(self, clip):
                return np.ones(4, np.float32) * float(clip[0])

            def similarity(self, vec):
                return {1.0: 0.93, 2.0: 0.41, 3.0: 0.94}[round(float(vec[0]), 1)]

        audio = np.concatenate([np.full(3 * SR, v, np.float32) for v in (1.0, 2.0, 3.0)])
        diar = [{"start": 0.0, "end": 3.0, "speaker": 0}, {"start": 3.0, "end": 6.0, "speaker": 1},
                {"start": 6.0, "end": 9.0, "speaker": 2}]
        names = diarize.name_speakers(audio, diar, VP(), OWNER)
        self.assertEqual(names, {0: OWNER, 1: "Speaker 2", 2: OWNER})


class AgreeingGroup(unittest.TestCase):
    def test_outlier_left_out(self):
        a = np.array([1.0, 0.0, 0.0], np.float32)
        b = np.array([0.95, 0.3, 0.0], np.float32)
        c = np.array([0.9, 0.0, 0.4], np.float32)
        x = np.array([0.0, 1.0, 0.0], np.float32)
        self.assertEqual(diarize.agreeing_group([x, a, b, c]), (1, 2, 3))
        self.assertIsNone(diarize.agreeing_group([x, a, b]))


class LabelChild(unittest.TestCase):
    """label_job runs the whole labelling step in a child Python with no window and reads its JSON back."""

    def test_child_writes_speakers_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            audio = write_wav(tmp / "a.wav", np.zeros(SR, np.float32))     # silence: diarisation finds nothing
            whisper = tmp / "whisper.json"
            whisper.write_text(json.dumps({"segments": []}), encoding="utf-8")
            calls = []
            real_run = diarize.subprocess.run

            def spy(cmd, **kw):
                calls.append(kw)
                return real_run(cmd, **kw)

            models = MODELS if models_present() else tmp
            with unittest.mock.patch.object(diarize.subprocess, "run", side_effect=spy):
                if models_present():
                    out = diarize.label_job(audio, whisper, tmp / "speakers.json", models, threshold=0.5,
                                            owner_name=OWNER, owner_match=0.6, voiceprint_path=tmp / "vp.json")
                    self.assertEqual(out, {"owner": OWNER, "owner_matched": False, "names": {}, "diarization": [],
                                           "lines": []})
                    self.assertTrue((tmp / "speakers.json").is_file())
                else:
                    with self.assertRaises(RuntimeError):
                        diarize.label_job(audio, whisper, tmp / "speakers.json", models, threshold=0.5,
                                          owner_name=OWNER, owner_match=0.6, voiceprint_path=None)
            self.assertEqual(calls[0]["creationflags"], diarize.subprocess.CREATE_NO_WINDOW)
            self.assertIs(calls[0]["stdin"], diarize.subprocess.DEVNULL)


class NamingPure(unittest.TestCase):
    def test_first_appearance_order_without_voiceprint(self):
        diar = [{"start": 4.0, "end": 6.0, "speaker": 2}, {"start": 0.5, "end": 3.0, "speaker": 5},
                {"start": 7.0, "end": 9.0, "speaker": 0}, {"start": 9.5, "end": 11.0, "speaker": 5}]
        names = diarize.name_speakers(np.zeros(SR * 12, np.float32), diar, None, OWNER)
        self.assertEqual(names, {5: "Speaker 1", 2: "Speaker 2", 0: "Speaker 3"})

    def test_empty_diarisation(self):
        self.assertEqual(diarize.name_speakers(np.zeros(SR, np.float32), [], None, OWNER), {})


class PitchFrames(unittest.TestCase):
    def register(self, audio: np.ndarray, diar: list[dict] | None = None) -> dict[int, float]:
        return diarize.pitch_registers(audio, diar or [span(0.0, audio.size / SR, 0)])

    def test_reads_the_pitch_of_a_buzz(self):
        for f0 in (70.0, 126.0, 188.0, 390.0):
            self.assertAlmostEqual(self.register(buzz(f0, 1.0))[0], f0, delta=0.02 * f0)

    def test_no_pitch_in_silence_a_constant_or_noise(self):
        for audio in (np.zeros(SR, np.float32), np.full(SR, 0.5, np.float32), hiss(1.0, 0.2, seed=3)):
            self.assertEqual(self.register(audio), {})

    def test_quiet_frames_are_left_out(self):
        """A trailing syllable far below the loud level reads at twice his pitch; it must not count."""
        audio = np.concatenate([buzz(100.0, 0.6), buzz(200.0, 1.0, level=0.02)])
        self.assertAlmostEqual(self.register(audio)[0], 100.0, delta=3.0)

    def test_only_the_diarised_speech_is_read(self):
        audio = np.concatenate([buzz(100.0, 1.0), buzz(200.0, 1.0)])
        pitch, _rms = diarize.pitch_frames(audio, [span(1.0, 2.0, 0)])
        self.assertTrue(np.all(np.isnan(pitch[:95])))
        self.assertAlmostEqual(self.register(audio, [span(1.0, 2.0, 0)])[0], 200.0, delta=4.0)

    def test_one_register_per_cluster(self):
        audio, spans = timeline(buzz(126.0, 2.0), hiss(0.5), buzz(188.0, 2.0), hiss(0.5), buzz(126.0, 1.0))
        regs = self.register(audio, [span(*spans[0], 0), span(*spans[2], 1), span(*spans[4], 0)])
        self.assertEqual(set(regs), {0, 1})
        self.assertAlmostEqual(regs[0], 126.0, delta=3.0)
        self.assertAlmostEqual(regs[1], 188.0, delta=4.0)


class Refine(unittest.TestCase):
    """refine on buzzes: a man's register at 126 Hz and a woman's at 188 Hz, a recorded meeting's two voices.
    Her short reply after his last turn is 0.3 s of pitch then 0.4 s with none, as a "ja" ends in breath."""

    @classmethod
    def setUpClass(cls):
        cls.audio, s = timeline(buzz(126.0, 4.0), hiss(0.6), buzz(188.0, 3.0), hiss(0.6, seed=1), buzz(126.0, 2.5),
                                hiss(0.35, seed=2), np.concatenate([buzz(184.0, 0.3), hiss(0.4, 0.1, seed=3)]))
        cls.low1, cls.high, cls.low2, cls.pause, cls.tail = s[0], s[2], s[4], s[5], s[6]

    def refine(self, diar, audio=None, embed=None) -> list[dict]:
        return diarize.refine(self.audio if audio is None else audio, diar, embed=embed)

    def merged(self, high_speaker: int = 0) -> list[dict]:
        return [span(*self.low1, 0), span(*self.high, high_speaker), span(self.low2[0], self.tail[1], 0)]

    def assert_parted(self, out: list[dict]) -> None:
        self.assertEqual([s["speaker"] for s in out], [0, 1, 0, 1])
        self.assertEqual(triples(out[:2]), [(*self.low1, 0), (*self.high, 1)])
        cut = out[2]["end"]
        self.assertEqual((out[2]["start"], out[3]["start"], out[3]["end"]), (self.low2[0], cut, self.tail[1]))
        self.assertTrue(self.pause[0] < cut < self.pause[1], f"cut at {cut}, inside the pause {self.pause}")

    def test_one_cluster_of_a_low_and_a_high_voice_is_parted(self):
        """sherpa at 0.5: everything in one cluster."""
        self.assert_parted(self.refine(self.merged()))

    def test_a_short_reply_joins_the_voice_it_matches(self):
        """sherpa at 0.4: her first turn its own cluster, the reply inside his last segment."""
        self.assert_parted(self.refine(self.merged(high_speaker=1)))

    def test_a_reply_joins_a_voice_within_pitch_same(self):
        """A short reply's pitch is marked (a question, a "ja"): it joins a voice 1.2x from it, not one 1.3x away."""
        for reply, joins in ((192.0, True), (208.0, False)):
            audio, s = timeline(buzz(126.0, 4.0), hiss(0.6), buzz(160.0, 3.0), hiss(0.6, seed=1), buzz(126.0, 2.5),
                                hiss(0.35, seed=2), np.concatenate([buzz(reply, 0.3), hiss(0.4, 0.1, seed=3)]))
            diar = [span(*s[0], 0), span(*s[2], 1), span(s[4][0], s[6][1], 0)]
            out = self.refine(diar, audio=audio)
            self.assertEqual([x["speaker"] for x in out], [0, 1, 0, 1] if joins else [0, 1, 0], reply)

    def test_a_short_reply_with_no_voice_to_join_stays(self):
        diar = [span(*self.low1, 0), span(self.low2[0], self.tail[1], 0)]
        self.assertEqual(self.refine(diar), diar)

    def test_a_long_stretch_in_another_register_is_a_voice_of_its_own(self):
        audio, s = timeline(buzz(126.0, 4.0), hiss(0.6), buzz(126.0, 2.5), hiss(0.35, seed=2), buzz(184.0, 0.8))
        out = self.refine([span(*s[0], 0), span(s[2][0], s[4][1], 0)], audio=audio)
        self.assertEqual([s["speaker"] for s in out], [0, 0, 1])
        self.assertEqual(out[2]["end"], s[4][1])

    def test_a_reply_that_matches_two_voices_stays(self):
        audio, s = timeline(buzz(126.0, 4.0), hiss(0.6), buzz(180.0, 3.0), hiss(0.6, seed=1), buzz(196.0, 3.0),
                            hiss(0.6, seed=2), buzz(126.0, 2.5), hiss(0.35, seed=3),
                            np.concatenate([buzz(188.0, 0.3), hiss(0.4, 0.1, seed=4)]))
        diar = [span(*s[0], 0), span(*s[2], 1), span(*s[4], 2), span(s[6][0], s[8][1], 0)]
        self.assertEqual(self.refine(diar, audio=audio), diar)

    def test_one_voice_over_its_whole_range_stays_one(self):
        """100 to 158 Hz is 1.58x, wider than PITCH_SPLIT, but in steps of 1.12x: no gap between two voices."""
        audio, s = timeline(*[x for k, f in enumerate((100.0, 112.0, 126.0, 141.0, 158.0, 230.0))
                              for x in (buzz(f, 1.5), hiss(0.6, seed=k))])
        diar = [span(s[0][0], s[8][1], 0), span(*s[10], 1)]
        self.assertEqual(self.refine(diar, audio=audio), diar)

    def test_registers_closer_than_pitch_split_stay_one_voice(self):
        audio, s = timeline(buzz(126.0, 3.0), hiss(0.6), buzz(160.0, 1.5), hiss(0.6, seed=1), buzz(230.0, 3.0))
        diar = [span(s[0][0], s[2][1], 0), span(*s[4], 1)]
        self.assertEqual(self.refine(diar, audio=audio), diar)

    def test_embeddings_that_agree_keep_one_voice(self):
        same = lambda clip: np.ones(4, np.float32)   # noqa: E731
        self.assertEqual(self.refine(self.merged(), embed=same), self.merged())

        def too_short(clip):
            raise ValueError("audio too short to embed")

        self.assert_parted(self.refine(self.merged(), embed=too_short))
        apart = lambda clip: np.eye(4, dtype=np.float32)[int(clip.size > 5 * SR)]   # his 6.7 s, her 3.9 s  # noqa: E731
        self.assert_parted(self.refine(self.merged(), embed=apart))

    def test_a_stretch_with_no_pitch_goes_with_its_neighbours(self):
        audio, s = timeline(buzz(126.0, 4.0), hiss(0.6), buzz(188.0, 1.0), hiss(0.3, seed=1), hiss(0.5, 0.1, seed=2),
                            hiss(0.3, seed=3), buzz(188.0, 1.0), hiss(0.6, seed=4), buzz(126.0, 3.0))
        out = self.refine([span(*s[0], 0), span(s[2][0], s[6][1], 0), span(*s[8], 0)], audio=audio)
        self.assertEqual(triples(out), [(*s[0], 0), (s[2][0], s[6][1], 1), (*s[8], 0)])

    def test_nothing_to_part_comes_back_as_it_was(self):
        diar = [span(*self.low1, 0), span(*self.high, 1), span(*self.low2, 0)]
        self.assertEqual(self.refine(diar), diar)
        silent = [span(0.0, 1.0, 0), span(1.5, 2.0, 1)]
        self.assertEqual(self.refine(silent, audio=np.zeros(3 * SR, np.float32)), silent)
        self.assertEqual(self.refine([]), [])


class LengthVoiceprint:
    """A voiceprint whose score depends only on a clip's length, so each cluster's score is set by its segment."""
    enrolled = True

    def __init__(self, scores: dict[float, float]):
        self.scores = scores

    def embed(self, clip):
        return np.array([clip.size / SR, 1.0], np.float32)

    def similarity(self, vec):
        return self.scores[round(float(vec[0]), 1)]


class NamingByPitch(unittest.TestCase):
    def names(self, pieces: list[tuple[float, float]], voiceprint=None) -> dict[int, str]:
        """pieces: (pitch, seconds) per cluster 0, 1, ..., each one segment, in this order."""
        audio, s = timeline(*[x for f, sec in pieces for x in (buzz(f, sec), hiss(0.5))])
        diar = [span(*s[2 * k], k) for k in range(len(pieces))]
        return diarize.name_speakers(audio, diar, voiceprint, OWNER)

    def test_a_voice_far_in_pitch_is_never_the_owner(self):
        """0.86 is within OWNER_BAND of the owner's 0.95, as the woman in a recorded meeting (0.84) nearly was."""
        vp = LengthVoiceprint({3.0: 0.95, 2.0: 0.86})
        self.assertEqual(self.names([(126.0, 3.0), (188.0, 2.0)], vp), {0: OWNER, 1: "Speaker 2"})
        self.assertEqual(self.names([(126.0, 3.0), (133.0, 2.0)], vp), {0: OWNER, 1: OWNER}, "one voice split")

    def test_owner_band_with_pitch(self):
        self.assertEqual(diarize.owner_clusters({0: 0.95, 1: 0.86}, 0.6), {0, 1})
        self.assertEqual(diarize.owner_clusters({0: 0.95, 1: 0.86}, 0.6, {0: 126.0, 1: 188.0}), {0})
        self.assertEqual(diarize.owner_clusters({0: 0.93, 1: 0.41, 2: 0.94}, 0.45, {0: 120.0, 1: 190.0, 2: 125.0}),
                         {0, 2})
        self.assertEqual(diarize.owner_clusters({0: 0.95, 1: 0.86}, 0.6, {0: 126.0}), {0, 1}, "pitch unknown")

    def test_a_fragment_shares_the_label_of_its_voice(self):
        self.assertEqual(self.names([(126.0, 3.0), (188.0, 2.0), (184.0, 0.8)]),
                         {0: "Speaker 1", 1: "Speaker 2", 2: "Speaker 2"})
        vp = LengthVoiceprint({3.0: 0.95, 2.0: 0.5})
        self.assertEqual(self.names([(126.0, 3.0), (188.0, 2.0), (184.0, 0.8)], vp),
                         {0: OWNER, 1: "Speaker 2", 2: "Speaker 2"})

    def test_a_fragment_never_takes_the_owner_name(self):
        vp = LengthVoiceprint({3.0: 0.95, 2.0: 0.5})
        self.assertEqual(self.names([(126.0, 3.0), (188.0, 2.0), (124.0, 0.8)], vp),
                         {0: OWNER, 1: "Speaker 2", 2: "Speaker 3"})
        self.assertEqual(self.names([(126.0, 3.0), (188.0, 2.0), (124.0, 0.8)]),
                         {0: "Speaker 1", 1: "Speaker 2", 2: "Speaker 1"}, "no owner: it joins the low voice")

    def test_a_fragment_that_matches_two_voices_keeps_its_own_label(self):
        self.assertEqual(self.names([(126.0, 3.0), (180.0, 2.0), (196.0, 2.0), (188.0, 0.8)]),
                         {0: "Speaker 1", 1: "Speaker 2", 2: "Speaker 3", 3: "Speaker 4"})

    def test_labels_follow_first_appearance_with_a_fragment(self):
        self.assertEqual(self.names([(126.0, 3.0), (188.0, 0.8), (300.0, 2.0), (188.0, 2.0)]),
                         {0: "Speaker 1", 1: "Speaker 2", 2: "Speaker 3", 3: "Speaker 2"})


def seg(words: list[tuple[float, float, str]]) -> dict:
    return {"start": words[0][0], "end": words[-1][1], "text": " ".join(w for _, _, w in words),
            "words": [{"start": a, "end": b, "word": w} for a, b, w in words]}


class AssignWords(unittest.TestCase):
    NAMES = {0: OWNER, 1: "Speaker 2"}

    def test_greatest_overlap_wins(self):
        diar = [{"start": 0.0, "end": 2.0, "speaker": 0}, {"start": 1.8, "end": 5.0, "speaker": 1}]
        # "straddle" overlaps speaker 0 by 0.1 s and speaker 1 by 0.5 s
        lines = diarize.assign_words([seg([(0.2, 0.6, "Hello"), (0.7, 1.2, "there."), (1.9, 2.4, "straddle"),
                                           (2.5, 3.0, "word")])], diar, self.NAMES)
        self.assertEqual([(l["speaker"], l["text"]) for l in lines],
                         [(OWNER, "Hello there."), ("Speaker 2", "straddle word")])
        self.assertEqual((lines[0]["start"], lines[0]["end"]), (0.2, 1.2))
        self.assertEqual((lines[1]["start"], lines[1]["end"]), (1.9, 3.0))

    def test_no_overlap_goes_to_nearest(self):
        diar = [{"start": 0.0, "end": 2.0, "speaker": 0}, {"start": 6.0, "end": 8.0, "speaker": 1}]
        lines = diarize.assign_words([seg([(1.5, 1.9, "one"), (2.3, 2.6, "near-first"), (5.2, 5.6, "near-second"),
                                           (6.1, 6.5, "two"), (9.0, 9.3, "after")])], diar, self.NAMES)
        self.assertEqual([(l["speaker"], l["text"]) for l in lines],
                         [(OWNER, "one near-first"), ("Speaker 2", "near-second two after")])
        early = diarize.assign_words([seg([(-0.5, -0.1, "before")])], diar, self.NAMES)
        self.assertEqual(early[0]["speaker"], OWNER)

    def test_gap_splits_same_speaker(self):
        diar = [{"start": 0.0, "end": 20.0, "speaker": 0}]
        lines = diarize.assign_words([seg([(0.0, 0.5, "a"), (1.0, 1.5, "b")]), seg([(5.0, 5.4, "c"), (8.3, 8.6, "d")])],
                                     diar, self.NAMES)
        self.assertEqual([l["text"] for l in lines], ["a b", "c d"])   # 3.5 s pause splits, 2.9 s does not
        self.assertEqual(len(diarize.assign_words([seg([(0.0, 0.5, "a"), (5.0, 5.4, "c")])], diar, self.NAMES, gap=10.0)), 1)

    def test_speaker_change_splits_and_rejoins(self):
        diar = [{"start": 0, "end": 2, "speaker": 0}, {"start": 2, "end": 4, "speaker": 1}, {"start": 4, "end": 6, "speaker": 0}]
        lines = diarize.assign_words([seg([(0.5, 1.0, "x"), (2.5, 3.0, "y"), (4.5, 5.0, "z")])], diar, self.NAMES)
        self.assertEqual([(l["speaker"], l["text"]) for l in lines], [(OWNER, "x"), ("Speaker 2", "y"), (OWNER, "z")])

    def test_overlapping_speech_nested_segment(self):
        diar = [{"start": 0.0, "end": 30.0, "speaker": 0}, {"start": 10.0, "end": 11.0, "speaker": 1}]
        lines = diarize.assign_words([seg([(9.0, 9.5, "long"), (10.1, 10.9, "interjection"), (12.0, 12.4, "resumes")])],
                                     diar, self.NAMES)
        # inside the nested span both overlap equally (0.8 s); the previous word's speaker keeps the floor
        self.assertEqual([(l["speaker"], l["text"]) for l in lines], [(OWNER, "long interjection resumes")])
        same_start = [{"start": 10.0, "end": 30.0, "speaker": 0}, {"start": 10.0, "end": 11.0, "speaker": 1}]
        lines = diarize.assign_words([seg([(10.1, 10.9, "interjection"), (12.0, 12.4, "resumes")])], same_start, self.NAMES)
        self.assertEqual([l["speaker"] for l in lines], ["Speaker 2", OWNER])

    def test_backchannel_inside_a_monologue_keeps_one_line(self):
        diar = [{"start": 0.0, "end": 20.0, "speaker": 0}, {"start": 10.0, "end": 10.7, "speaker": 1}]
        words = [(9.0, 9.4, "we"), (9.5, 9.9, "should"), (10.05, 10.3, "send"), (10.35, 10.6, "the"),
                 (11.0, 11.3, "pack"), (11.4, 11.8, "today.")]
        lines = diarize.assign_words([seg(words)], diar, self.NAMES)
        self.assertEqual([(l["speaker"], l["text"]) for l in lines], [(OWNER, "we should send the pack today.")])

    def test_tie_after_the_interjector_stays_with_the_interjector(self):
        diar = [{"start": 0.0, "end": 20.0, "speaker": 0}, {"start": 9.0, "end": 11.0, "speaker": 1}]
        lines = diarize.assign_words([seg([(9.0, 9.5, "ja"), (10.0, 10.5, "nee")])],
                                     [{"start": 9.0, "end": 9.6, "speaker": 1}, *diar], self.NAMES)
        self.assertEqual(lines[0]["speaker"], "Speaker 2")

    def test_segment_without_words_and_unknown_cluster(self):
        diar = [{"start": 0.0, "end": 5.0, "speaker": 3}]
        lines = diarize.assign_words([{"start": 1.0, "end": 4.0, "text": "  no   word stamps ", "words": []}], diar, {})
        self.assertEqual(lines, [{"speaker": "Speaker 4", "start": 1.0, "end": 4.0, "text": "no word stamps"}])

    def test_empty_inputs(self):
        self.assertEqual(diarize.assign_words([], [], {}), [])
        lines = diarize.assign_words([seg([(0.0, 0.5, "solo")])], [], {})
        self.assertEqual(lines[0]["speaker"], "")

    def test_zero_length_word(self):
        diar = [{"start": 0.0, "end": 1.0, "speaker": 0}, {"start": 1.0, "end": 2.0, "speaker": 1}]
        lines = diarize.assign_words([seg([(1.5, 1.5, "point")])], diar, self.NAMES)
        self.assertEqual(lines[0]["speaker"], "Speaker 2")


class ReadAudio(unittest.TestCase):
    def test_downmix_and_resample(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = np.arange(48000 * 2) / 48000
            stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1).astype(np.float32) * 0.5
            audio = diarize.read_audio(write_wav(Path(tmp) / "s.wav", stereo, 48000))
            self.assertEqual(audio.dtype, np.float32)
            self.assertEqual(audio.ndim, 1)
            self.assertAlmostEqual(audio.size / SR, 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
