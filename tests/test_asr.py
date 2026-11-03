"""asr: pure chunking, loop, filler and language-renormalisation tests, then GPU tests on the fixtures."""
from __future__ import annotations

import gc
import importlib.util
import json
import logging
import os
import random
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scribe import asr

FIXTURES = Path(__file__).parent / "fixtures"
SKILL_SCRIPT = Path.home() / ".claude" / "skills" / "transcribe" / "transcribe.py"
EPS = 1e-6


def _cuda() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _cached(model_name: str) -> bool:
    """True when the model is complete in the local HF cache, so a test never waits on a download."""
    try:
        from faster_whisper.utils import download_model
        download_model(model_name, local_files_only=True)
        return True
    except Exception:
        return False


def _wav(name: str) -> np.ndarray:
    from faster_whisper import decode_audio
    return decode_audio(str(FIXTURES / name), sampling_rate=asr.SAMPLE_RATE)


def _w(word: str, start: float) -> dict:
    return {"start": start, "end": start + 0.2, "word": word}


# ---------------------------------------------------------------- chunking

class TestPlanChunks(unittest.TestCase):
    def check_invariants(self, regions, chunks, max_s=asr.CHUNK_MAX_S):
        regs = asr._normalise_regions(regions)
        for a, b in chunks:
            self.assertLessEqual(b - a, max_s + EPS)
            self.assertLess(a, b)
        for (a0, b0), (a1, b1) in zip(chunks, chunks[1:]):
            self.assertLessEqual(b0, a1 + EPS)
        for s, e in regs:  # every bit of speech lands in a chunk
            covered = sum(max(0.0, min(e, b) - max(s, a)) for a, b in chunks)
            self.assertAlmostEqual(covered, e - s, places=6)
        bounds = {x for r in regs for x in r}
        for a, b in chunks:  # cuts fall on region edges, except inside a region too long to keep whole
            for t in (a, b):
                inside_long = any(s < t < e and e - s > max_s for s, e in regs)
                self.assertTrue(t in bounds or inside_long, f"cut at {t} is mid-region")

    def test_empty(self):
        self.assertEqual(asr.plan_chunks([]), [])

    def test_fits_in_one(self):
        regs = [(0.5, 4.0), (5.0, 12.0), (13.0, 20.0)]
        self.assertEqual(asr.plan_chunks(regs), [(0.5, 20.0)])

    def test_cuts_at_longest_pause(self):
        regs = [(0.0, 10.0), (10.5, 16.0), (18.0, 24.0), (24.2, 30.0)]
        chunks = asr.plan_chunks(regs)
        self.assertEqual(chunks, [(0.0, 16.0), (18.0, 30.0)])
        self.check_invariants(regs, chunks)

    def test_prefers_long_enough_chunk_over_early_long_pause(self):
        regs = [(0.0, 2.0), (12.0, 18.0), (18.6, 26.0), (26.3, 40.0)]
        chunks = asr.plan_chunks(regs)
        # the 10 s pause at 2 s would leave a 2 s chunk; of the cuts keeping >= 14 s, 0.6 s at 18 s is longest
        self.assertEqual(chunks, [(0.0, 18.0), (18.6, 40.0)])
        self.check_invariants(regs, chunks)

    def test_single_70s_region_hard_split(self):
        chunks = asr.plan_chunks([(0.0, 70.0)])
        self.assertEqual(len(chunks), 3)
        self.assertAlmostEqual(chunks[0][0], 0.0)
        self.assertEqual(chunks[-1][1], 70.0)
        for (a, b), (c, d) in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(b, c)
        for a, b in chunks:
            self.assertAlmostEqual(b - a, 70.0 / 3)
        self.check_invariants([(0.0, 70.0)], chunks)

    def test_long_region_between_short_ones(self):
        regs = [(0.0, 5.0), (6.0, 76.0), (77.0, 80.0)]
        chunks = asr.plan_chunks(regs)
        self.assertEqual(chunks[0], (0.0, 5.0))
        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks[-1], (77.0, 80.0))
        self.check_invariants(regs, chunks)

    def test_exactly_max_is_kept_whole(self):
        self.assertEqual(asr.plan_chunks([(3.0, 31.0)]), [(3.0, 31.0)])

    def test_overlapping_regions_merge(self):
        self.assertEqual(asr.plan_chunks([(5.0, 9.0), (0.0, 6.0)]), [(0.0, 9.0)])

    def test_random_invariants(self):
        rng = random.Random(7)
        for _ in range(300):
            t, regs = rng.uniform(0, 3), []
            for _ in range(rng.randint(1, 60)):
                s = t + rng.choice([0.0, rng.uniform(0, 0.8), rng.uniform(0, 6)])
                e = s + rng.choice([rng.uniform(0.2, 12), rng.uniform(0.2, 3), rng.uniform(20, 90)])
                regs.append((s, e))
                t = e
            self.check_invariants(regs, asr.plan_chunks(regs))


# ---------------------------------------------------------------- loops

def _skill():
    if not SKILL_SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location("transcribe_skill_ref", SKILL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLoops(unittest.TestCase):
    def test_loop_marked(self):
        words = [_w("hello", 0.0)] + [_w(x, 1.0 + i * 0.3) for i, x in enumerate("we will send it".split() * 4)] + [_w("bye", 9.0)]
        out = asr.collapse_word_loops(words)
        self.assertEqual([w["word"] for w in out[:5]], ["hello", "we", "will", "send", "it"])
        self.assertEqual(out[5]["word"], "[repetition loop removed, 3 repeats to 00:06]")
        self.assertEqual(out[6]["word"], "bye")
        self.assertEqual(len(out), 7)

    def test_two_repeats_kept(self):
        words = [_w(x, i) for i, x in enumerate("a b c a b c".split())]
        self.assertEqual(asr.collapse_word_loops(words), words)

    def test_matches_skill(self):
        skill = _skill()
        if skill is None:
            self.skipTest("transcribe skill script not present")
        rng = random.Random(3)
        vocab = ["the", "The", "fund,", "fund", "pack.", "is", "ok", "a", "b"]
        for _ in range(400):
            words, t = [], 0.0
            while len(words) < rng.randint(0, 80):
                if rng.random() < 0.25:
                    phrase = [rng.choice(vocab) for _ in range(rng.randint(1, 16))]
                    for _ in range(rng.randint(2, 5)):
                        for x in phrase:
                            words.append(_w(x, t)); t += 0.3
                else:
                    words.append(_w(rng.choice(vocab), t)); t += 0.3
            self.assertEqual(asr.collapse_word_loops(words), skill.collapse_word_loops(words))
        self.assertEqual(asr.fmt_ts(3725.4), skill.fmt_ts(3725.4))

    def test_segment_loops_across_segments(self):
        phrase = "send the pack now".split()
        t, segs = 0.0, []
        for k in range(4):  # the phrase repeats once per segment
            ws = []
            for x in phrase:
                ws.append(_w(x, t)); t += 0.3
            segs.append({"start": ws[0]["start"], "end": ws[-1]["end"], "text": " ".join(phrase), "language": "en", "words": ws})
        tail = [_w("done", t + 1)]
        segs.append({"start": tail[0]["start"], "end": tail[0]["end"], "text": "done", "language": "en", "words": tail})
        out = asr.collapse_segment_loops(segs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "send the pack now [repetition loop removed, 3 repeats to 00:05]")
        self.assertEqual(out[1]["text"], "done")
        flat = [w for s in out for w in s["words"]]
        self.assertTrue(all(a["start"] <= b["start"] for a, b in zip(flat, flat[1:])))
        self.assertEqual(out[0]["end"], out[0]["words"][-1]["end"])

    def test_segment_loops_noop(self):
        segs = [{"start": 0, "end": 1, "text": "a b", "language": "en", "words": [_w("a", 0), _w("b", 0.5)]}]
        self.assertIs(asr.collapse_segment_loops(segs), segs)


class TestMonotonic(unittest.TestCase):
    def test_overlaps_removed(self):
        segs = [{"start": 0.0, "end": 2.0, "text": "a b", "language": "en",
                 "words": [{"start": 0.0, "end": 1.2, "word": "a"}, {"start": 1.0, "end": 0.9, "word": "b"}]},
                {"start": 1.1, "end": 3.0, "text": "c", "language": "en", "words": [{"start": 1.1, "end": 3.0, "word": "c"}]}]
        asr.make_monotonic(segs)
        flat = [w for s in segs for w in s["words"]]
        for a, b in zip(flat, flat[1:]):
            self.assertLessEqual(a["end"], b["start"])
        for w in flat:
            self.assertLessEqual(w["start"], w["end"])
        self.assertEqual(segs[1]["start"], flat[2]["start"])


# ---------------------------------------------------------------- text and gates

class TestFillers(unittest.TestCase):
    def test_cases(self):
        cases = {
            "Um, so I think we should go.": "So I think we should go.",
            "I was, um, thinking about it.": "I was thinking about it.",
            "That's it, uh.": "That's it.",
            "Send the uh report today. Hmm, maybe tomorrow.": "Send the report today. Maybe tomorrow.",
            "Uhm I'll do it. Er, no.": "I'll do it. No.",
            "Ummm, uhh, the Project Planner.": "The Project Planner.",
            "Hmm.": "",
            "Ah ja, dis reg.": "Ja, dis reg.",
            "Umbrella and hummus are words.": "Umbrella and hummus are words.",
        }
        for src, want in cases.items():
            self.assertEqual(asr.strip_fillers(src), want, src)


class TestApplyVocabulary(unittest.TestCase):
    def test_cases(self):
        vocab = ["Project Planner", "BlueSky", "Help Desk", "Claude", "ORCA", "Capital IQ", "Bloomberg", "Acme", "Kira"]
        cases = {
            "Please send the project planner results to Bluesky by Friday.":
                "Please send the Project Planner results to BlueSky by Friday.",
            "Put it on the help  desk for claude's run.": "Put it on the Help Desk for Claude's run.",
            "orca and capital iq, then bloomberg.": "ORCA and Capital IQ, then Bloomberg.",
            "Acmes and Kira-ish stay as they are.": "Acmes and Kira-ish stay as they are.",
            "The helpdesk is one word here.": "The helpdesk is one word here.",
            "": "",
        }
        for src, want in cases.items():
            self.assertEqual(asr.apply_vocabulary(src, vocab), want, src)

    def test_empty_vocabulary(self):
        self.assertEqual(asr.apply_vocabulary("project planner", []), "project planner")


class TestGates(unittest.TestCase):
    def test_speech_rms(self):
        self.assertEqual(asr.speech_rms(np.zeros(16000, np.float32)), 0.0)
        t = np.arange(16000 * 3) / 16000
        quiet = np.zeros_like(t, dtype=np.float32)
        quiet[16000:16000 + 8000] = 0.05 * np.sin(2 * np.pi * 220 * t[:8000])  # 0.5 s of tone in 3 s
        self.assertGreater(asr.speech_rms(quiet), asr.MIN_RMS)
        self.assertGreater(asr.speech_rms((quiet * 32767).astype(np.int16)), asr.MIN_RMS)

    def test_hallucination_rule(self):
        self.assertTrue(asr.is_hallucination(0.7, -1.2))
        self.assertFalse(asr.is_hallucination(0.7, -0.5))
        self.assertFalse(asr.is_hallucination(0.5, -1.5))

    def test_stock_phrase_on_room_noise_is_dropped(self):
        # measured on a silent notetaker run: room tone gave this at no_speech 0.82, avg_logprob -0.82
        self.assertTrue(asr.is_hallucination(0.823, -0.818, " Thank you for watching."))
        self.assertTrue(asr.is_hallucination(0.35, -0.3, "Thank you."))
        self.assertTrue(asr.is_hallucination(0.4, -0.2, "Baie dankie!"))
        self.assertFalse(asr.is_hallucination(0.05, -0.2, "Thank you."))      # said in the room: kept
        self.assertFalse(asr.is_hallucination(0.823, -0.818, "Thank you, Adam, send the pack."))

    def test_prompts(self):
        self.assertEqual(asr.vocab_prompt(["Acme", " BlueSky "]), "Acme, BlueSky.")
        self.assertIsNone(asr.vocab_prompt([]))
        self.assertTrue(asr.file_prompt(["Kira"]).startswith("Kira. "))
        self.assertIn(asr.CODE_SWITCH_SAMPLE, asr.file_prompt([]))

    def test_silence_gated_without_model(self):
        d = asr.DictationASR("unused-model", ["en", "af"], [])
        self.assertEqual(d.transcribe(np.zeros(16000 * 2, np.float32)), "")
        self.assertIsNone(d.model)


class _FakeModel:
    supported_languages = ["en", "af", "nl", "de"]

    def __init__(self, probs):
        self.probs = probs
        self.calls = 0

    def detect_language(self, audio=None, **_):
        self.calls += 1
        self.audio_len = len(audio)
        ranked = sorted(self.probs.items(), key=lambda kv: -kv[1])
        return ranked[0][0], ranked[0][1], ranked


class _Seg:
    def __init__(self, text):
        self.text, self.no_speech_prob, self.avg_logprob = text, 0.0, -0.2


class _ChunkModel(_FakeModel):
    """Records every decode call; returns the clip's length as its text."""

    def __init__(self):
        super().__init__({"en": 0.9, "af": 0.1})
        self.decodes = []

    def transcribe(self, clip, **kw):
        self.decodes.append((len(clip), kw))
        return iter([_Seg(f"<{len(clip) / 16000:.1f}>")]), None


class TestLongDictation(unittest.TestCase):
    """A hold over 28 s is decoded in VAD chunks, each with the prompt, so no speech is lost at 30 s steps."""

    def setUp(self):
        self.d = asr.DictationASR("unused", ["en", "af"], ["Acme"], strip_fillers=False)
        self.d.model = _ChunkModel()

    def test_short_hold_is_one_call_without_timestamps(self):
        self.d.transcribe(np.full(16000 * 10, 0.1, np.float32))
        self.assertEqual(len(self.d.model.decodes), 1)
        self.assertTrue(self.d.model.decodes[0][1]["without_timestamps"])

    def test_long_hold_is_cut_at_pauses(self):
        regions = [(0.0, 20.0), (21.0, 40.0), (41.0, 55.0)]
        with mock.patch.object(asr, "speech_regions", return_value=regions):
            text = self.d.transcribe(np.full(16000 * 56, 0.1, np.float32))
        lengths = [n / 16000 for n, _ in self.d.model.decodes]
        self.assertEqual(lengths, [20.0, 19.0, 14.0])
        for _, kw in self.d.model.decodes:
            self.assertEqual((kw["language"], kw["beam_size"], kw["condition_on_previous_text"],
                              kw["initial_prompt"]), ("en", 5, False, "Acme."))
            self.assertTrue(kw["without_timestamps"])
        self.assertEqual(text, "<20.0> <19.0> <14.0>")

    def test_learned_terms_join_the_prompt_only(self):
        self.d.prompt_terms = lambda: ["Alex", "ACME", "Kira"]          # most used first
        self.d.transcribe(np.full(16000 * 10, 0.1, np.float32))
        self.assertEqual(self.d.model.decodes[0][1]["initial_prompt"], "Kira, Alex, Acme.")
        self.assertEqual(self.d.vocabulary, ["Acme"])                 # never added to the forced spellings

    def test_configured_vocabulary_ends_the_prompt(self):
        # faster-whisper keeps the prompt's tail: the configured terms and the most used learned terms
        self.d.prompt_terms = lambda: [f"Name{i}" for i in range(60)]
        self.d.transcribe(np.full(16000 * 10, 0.1, np.float32))
        prompt = self.d.model.decodes[0][1]["initial_prompt"]
        self.assertTrue(prompt.endswith("Name1, Name0, Acme."), prompt[-40:])

    def test_failing_learned_terms_keep_the_vocabulary(self):
        def broken():
            raise OSError("lexicon busy")
        self.d.prompt_terms = broken
        self.d.transcribe(np.full(16000 * 10, 0.1, np.float32))
        self.assertEqual(self.d.model.decodes[0][1]["initial_prompt"], "Acme.")

    def test_long_hold_without_vad_regions_lets_whisper_seek(self):
        with mock.patch.object(asr, "speech_regions", return_value=[]):
            self.d.transcribe(np.full(16000 * 40, 0.1, np.float32))
        self.assertEqual(len(self.d.model.decodes), 1)
        self.assertFalse(self.d.model.decodes[0][1]["without_timestamps"])


class TestOfflineModels(unittest.TestCase):
    def test_hub_is_offline(self):
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")

    def test_missing_model_never_downloads(self):
        calls = []

        def fake_download(name, local_files_only=False, **kw):
            calls.append(local_files_only)
            raise FileNotFoundError(name)

        with mock.patch("faster_whisper.utils.download_model", side_effect=fake_download):
            with self.assertRaises(asr.ModelNotCached) as ctx, self.assertLogs("scribe.asr", "ERROR"):
                asr.model_path("distil-large-v3")
        self.assertEqual(calls, [True])
        self.assertEqual(str(ctx.exception), "model not cached: distil-large-v3")


# ---------------------------------------------------------------- the GPU running out of memory

OOM = "CUDA failed with error out of memory"            # ctranslate2's text, forced on this machine 31 Oct 2026
BROKEN = "parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal"   # that model's next call


def capture_warnings(test: unittest.TestCase) -> list[str]:
    """scribe.asr's warnings for one test, kept off stderr: which path a transcription took."""
    seen: list[str] = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = lambda record: seen.append(record.getMessage())
    logger = logging.getLogger("scribe.asr")
    logger.addHandler(handler)
    test.addCleanup(logger.removeHandler, handler)
    return seen


class _Word:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _FileSeg(_Seg):
    def __init__(self, text, start, end):
        super().__init__(text)
        self.start, self.end = start, end
        self.words = [_Word(text, start, end)]


class _OomModel(_FakeModel):
    """A fake Whisper that names each decode by its clip length. On the decode calls in `fail_on` (counted from 1)
    it raises the CUDA out-of-memory error after its first segment, and like the real model it is broken from then
    on. Every decode is recorded in `log` as (model tag, clip seconds)."""

    def __init__(self, tag, log, fail_on=()):
        super().__init__({"en": 0.9, "af": 0.1})
        self.tag, self.log, self.fail_on = tag, log, set(fail_on)
        self.decodes = 0
        self.broken = False

    def detect_language(self, audio=None, **kw):
        if self.broken:
            raise RuntimeError(BROKEN)
        return super().detect_language(audio=audio, **kw)

    def transcribe(self, clip, **kw):
        if self.broken:
            raise RuntimeError(BROKEN)
        self.decodes += 1
        n = len(clip) / 16000

        def segments():
            yield _FileSeg(f"{n:.0f}a", 0.0, 1.0)
            if self.decodes in self.fail_on:
                self.broken = True
                raise RuntimeError(OOM)
            self.log.append((self.tag, round(n)))
            yield _FileSeg(f"{n:.0f}b", 1.0, 2.0)

        return segments(), None


class TestOutOfMemory(unittest.TestCase):
    """A CUDA out-of-memory error: a file is retried on a fresh GPU model, then finished on the CPU."""

    REGIONS = [(0.0, 20.0), (21.0, 40.0), (41.0, 55.0)]      # three chunks: 20 s, 19 s, 14 s

    def setUp(self):
        self.log, self.loads, self.waits, self.plan = [], [], [], []
        self.warnings = capture_warnings(self)
        for patcher in (mock.patch("faster_whisper.decode_audio", return_value=np.zeros(16000 * 56, np.float32)),
                        mock.patch.object(asr, "speech_regions", return_value=self.REGIONS),
                        mock.patch.object(asr, "load_model", side_effect=self.fake_load),
                        mock.patch.object(asr, "GPU_RETRY_S", 0.01)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_load(self, name, device=asr.DEVICE):
        self.loads.append(device)
        step = self.plan.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def run_file(self):
        return asr.transcribe_file(Path("call.flac"), "large-v3", ["en", "af"], [], on_wait=self.waits.append)

    def texts(self, result):
        return [s["text"] for s in result["segments"]]

    def test_the_error_is_recognised(self):
        self.assertTrue(asr.out_of_memory(RuntimeError(OOM)))
        self.assertTrue(asr.out_of_memory(RuntimeError("cuBLAS failed with status CUBLAS_STATUS_ALLOC_FAILED")))
        self.assertFalse(asr.out_of_memory(RuntimeError(BROKEN)))
        self.assertFalse(asr.out_of_memory(asr.ModelNotCached("model not cached: x")))
        self.assertFalse(asr.out_of_memory(OSError("out of memory")))

    def test_a_chunk_is_redone_on_a_fresh_gpu_model(self):
        self.plan = [_OomModel("gpu1", self.log, fail_on={2}), _OomModel("gpu2", self.log)]
        r = self.run_file()
        self.assertEqual(self.loads, ["cuda", "cuda"])
        self.assertEqual(self.log, [("gpu1", 20), ("gpu2", 19), ("gpu2", 14)])
        self.assertEqual(self.texts(r), ["20a", "20b", "19a", "19b", "14a", "14b"])   # no half chunk kept
        self.assertEqual(self.waits, [True, False])
        self.assertEqual(r["model"], "faster-whisper large-v3 (cuda, float16)")

    def test_three_tries_then_the_cpu_finishes(self):
        self.plan = [_OomModel("gpu1", self.log, fail_on={2}), RuntimeError(OOM), RuntimeError(OOM),
                     _OomModel("cpu", self.log)]
        r = self.run_file()
        self.assertEqual(self.loads, ["cuda", "cuda", "cuda", "cpu"])
        self.assertEqual(self.log, [("gpu1", 20), ("cpu", 19), ("cpu", 14)])
        self.assertEqual(self.waits, [True, False, True, False])
        self.assertEqual(r["model"], "faster-whisper large-v3 (cuda, float16; cpu, int8 from 00:21)")
        self.assertEqual(len(r["segments"]), 6)
        self.assertEqual(self.warnings, [
            "large-v3: CUDA out of memory (1 of 3 tries); the GPU again in 0 s",
            "large-v3: CUDA out of memory (2 of 3 tries); the GPU again in 0 s",
            "large-v3: CUDA out of memory 3 times; on the CPU (int8) from here"])

    def test_no_gpu_memory_from_the_start(self):
        self.plan = [RuntimeError(OOM), RuntimeError(OOM), RuntimeError(OOM), _OomModel("cpu", self.log)]
        r = self.run_file()
        self.assertEqual(self.loads, ["cuda", "cuda", "cuda", "cpu"])
        self.assertEqual(r["model"], "faster-whisper large-v3 (cpu, int8)")

    def test_other_errors_are_not_retried(self):
        model = _OomModel("gpu1", self.log)
        model.transcribe = mock.Mock(side_effect=RuntimeError("CUDA failed with error an illegal memory access"))
        self.plan = [model]
        with self.assertRaises(RuntimeError):
            self.run_file()
        self.assertEqual(self.loads, ["cuda"])
        self.assertEqual(self.waits, [])

    def test_the_failed_model_is_freed_before_the_next_load(self):
        import weakref

        refs, alive = [], []

        def load(name, device=asr.DEVICE):
            if not refs:
                model = _OomModel("gpu1", self.log, fail_on={1})
                refs.append(weakref.ref(model))
                return model
            alive.append(refs[0]() is not None)      # the broken model's memory is let go before a new one loads
            return _OomModel("gpu2", self.log)

        with mock.patch.object(asr, "load_model", side_effect=load):
            self.run_file()
        self.assertEqual(alive, [False])
        self.assertEqual([tag for tag, _ in self.log], ["gpu2"] * 3)


class TestLoadModel(unittest.TestCase):
    def test_gpu_and_cpu(self):
        seen = []

        def whisper_model(path, **kw):
            seen.append((path, kw))
            return object()

        with mock.patch("faster_whisper.WhisperModel", side_effect=whisper_model), \
                mock.patch.object(asr, "model_path", return_value="C:/models/x"):
            asr.load_model("large-v3")
            asr.load_model("large-v3", asr.CPU_DEVICE)
        self.assertEqual(seen, [("C:/models/x", {"device": "cuda", "compute_type": "float16"}),
                                ("C:/models/x", {"device": "cpu", "compute_type": "int8",
                                                 "cpu_threads": asr.CPU_THREADS})])


class _OomClipModel(_ChunkModel):
    """A dictation model on the GPU that runs out of memory on its next decode, and is broken after."""

    def transcribe(self, clip, **kw):
        raise RuntimeError(OOM)


class TestDictationOutOfMemory(unittest.TestCase):
    """Dictation never waits: a clip the GPU has no memory for runs on the CPU at once."""

    def setUp(self):
        self.loads = []
        self.cpu = _ChunkModel()
        self.gpu_again = _ChunkModel()
        self.gpu_gate = threading.Event()
        self.gpu_gate.set()
        self.gpu_load_error: Exception | None = None
        self.warnings = capture_warnings(self)
        patcher = mock.patch.object(asr, "load_model", side_effect=self.fake_load)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.d = asr.DictationASR("unused", ["en", "af"], [], strip_fillers=False)

    def fake_load(self, name, device=asr.DEVICE):
        self.loads.append(device)
        if device == asr.CPU_DEVICE:
            return self.cpu
        self.gpu_gate.wait(5)
        if self.gpu_load_error is not None:
            raise self.gpu_load_error
        return self.gpu_again

    def clip(self) -> np.ndarray:
        return np.full(16000 * 3, 0.1, np.float32)

    def test_a_clip_out_of_gpu_memory_runs_on_the_cpu_at_once(self):
        self.d.model = _OomClipModel()
        began = time.monotonic()
        self.assertEqual(self.d.transcribe(self.clip()), "<3.0>")
        self.assertLess(time.monotonic() - began, 2.0)      # a meeting would wait GPU_RETRY_S (15 s) here
        self.assertEqual(len(self.cpu.decodes), 1)
        self.assertIsNone(self.d.model)                   # the broken GPU model is dropped
        self.assertEqual(self.d.device, "cpu")
        self.assertEqual(self.loads, ["cpu"])             # no GPU reload within GPU_RETRY_S of the error
        self.d.transcribe(self.clip())
        self.assertEqual(len(self.cpu.decodes), 2)        # the CPU model stays loaded
        self.assertEqual(self.loads, ["cpu"])
        self.assertEqual(self.warnings, ["dictation: CUDA out of memory (clip); on the CPU until the GPU model is back"])

    def test_the_gpu_comes_back_on_its_own_thread(self):
        self.d.model = _OomClipModel()
        with mock.patch.object(asr, "GPU_RETRY_S", 0.0):
            self.d.transcribe(self.clip())
            self.assertTrue(wait_until(lambda: self.d.model is self.gpu_again))
        self.d.transcribe(self.clip())
        self.assertEqual(len(self.gpu_again.decodes), 1)
        self.assertEqual(self.d.device, "cuda")

    def test_a_reload_still_out_of_memory_keeps_the_cpu(self):
        self.d.model = _OomClipModel()
        self.gpu_load_error = RuntimeError(OOM)
        with mock.patch.object(asr, "GPU_RETRY_S", 0.0):
            self.d.transcribe(self.clip())
            self.assertTrue(wait_until(lambda: self.loads.count("cuda") == 1 and not self.d._reloading))
        self.assertEqual(self.d.device, "cpu")
        self.assertEqual(self.d.transcribe(self.clip()), "<3.0>")

    def test_no_gpu_memory_at_load(self):
        self.gpu_load_error = RuntimeError(OOM)
        self.d.load()
        self.assertEqual(self.loads, ["cuda", "cpu"])
        self.assertEqual(self.d.device, "cpu")
        self.assertEqual(self.d.transcribe(self.clip()), "<3.0>")

    def test_other_errors_still_raise(self):
        broken = _ChunkModel()
        broken.transcribe = mock.Mock(side_effect=RuntimeError("cuDNN failed with status CUDNN_STATUS_EXECUTION_FAILED"))
        self.d.model = broken
        with self.assertRaises(RuntimeError):
            self.d.transcribe(self.clip())
        self.assertIs(self.d.model, broken)
        self.assertEqual(self.loads, [])
        self.gpu_load_error = asr.ModelNotCached("model not cached: unused")
        with self.assertRaises(asr.ModelNotCached):
            asr.DictationASR("unused", ["en"], []).load()

    def test_a_reload_finishing_after_unload_is_dropped(self):
        self.d.model = _OomClipModel()
        self.gpu_gate.clear()
        with mock.patch.object(asr, "GPU_RETRY_S", 0.0):
            self.d.transcribe(self.clip())
            self.assertTrue(wait_until(lambda: "cuda" in self.loads))
            self.d.unload()
            self.gpu_gate.set()
            self.assertTrue(wait_until(lambda: not self.d._reloading))
        self.assertIsNone(self.d.model)
        self.assertIsNone(self.d.device)


def wait_until(cond, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return bool(cond())


class TestDetectLanguagePure(unittest.TestCase):
    def test_renormalised_over_allowed(self):
        m = _FakeModel({"nl": 0.6, "af": 0.3, "en": 0.1})
        lang, p = asr.detect_language(m, np.ones(16000 * 40, np.float32), ["en", "af"])
        self.assertEqual(lang, "af")
        self.assertAlmostEqual(p, 0.75)
        self.assertEqual(m.audio_len, 16000 * 30)  # first 30 s only

    def test_single_or_unsupported(self):
        m = _FakeModel({"en": 1.0})
        self.assertEqual(asr.detect_language(m, np.ones(16000, np.float32), ["en"]), ("en", 1.0))
        self.assertEqual(asr.detect_language(m, np.ones(16000, np.float32), ["xx", "af"]), ("af", 1.0))
        self.assertEqual(m.calls, 0)

    def test_zero_mass(self):
        m = _FakeModel({"de": 1.0, "en": 0.0, "af": 0.0})
        self.assertEqual(asr.detect_language(m, np.ones(16000, np.float32), ["en", "af"]), ("en", 0.5))


# ---------------------------------------------------------------- GPU

class _DictationGPU:
    MODEL = ""

    @classmethod
    def setUpClass(cls):
        if not _cuda():
            raise unittest.SkipTest("no CUDA device")
        if not _cached(cls.MODEL):
            raise unittest.SkipTest(f"{cls.MODEL} not complete in the local HF cache")
        cls.asr = asr.DictationASR(cls.MODEL, ["en", "af"], ["Project Planner", "BlueSky", "Claude"])
        cls.asr.load()

    @classmethod
    def tearDownClass(cls):
        cls.asr.unload()
        gc.collect()

    def test_vocabulary_prompt(self):
        text = self.asr.transcribe(_wav("short_dictation.wav"))
        self.assertIn("Project Planner", text)
        self.assertIn("BlueSky", text)

    def test_silence(self):
        self.assertEqual(self.asr.transcribe(np.zeros(16000 * 3, np.float32)), "")

    def test_faint_noise(self):
        rng = np.random.default_rng(1)
        self.assertEqual(self.asr.transcribe((rng.standard_normal(16000 * 3) * 0.001).astype(np.float32)), "")

    def test_detect_language_real(self):
        lang, p = asr.detect_language(self.asr.model, _wav("short_dictation.wav"), ["en", "af"])
        self.assertEqual(lang, "en")
        self.assertGreater(p, 0.9)
        lang, p = asr.detect_language(self.asr.model, _wav("short_dictation.wav"), ["af", "de"])
        self.assertIn(lang, ("af", "de"))
        self.assertTrue(0.5 <= p <= 1.0)


class TestDictationTurbo(_DictationGPU, unittest.TestCase):
    MODEL = "large-v3-turbo"


class TestDictationLargeV3(_DictationGPU, unittest.TestCase):
    MODEL = "large-v3"


class TestTranscribeFileGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _cuda():
            raise unittest.SkipTest("no CUDA device")
        if not _cached("large-v3"):
            raise unittest.SkipTest("large-v3 not complete in the local HF cache")
        cls.fractions = []
        cls.path = FIXTURES / "two_speakers.wav"
        cls.result = asr.transcribe_file(cls.path, "large-v3", ["en", "af"], ["Claude"],
                                         progress=cls.fractions.append)
        cls.duration = len(_wav("two_speakers.wav")) / asr.SAMPLE_RATE

    def test_schema(self):
        r = self.result
        for key in ("source", "model", "language", "language_probability", "languages", "duration",
                    "integrity", "teams_transcript", "segments"):
            self.assertIn(key, r)
        self.assertEqual(r["source"], "two_speakers.wav")
        self.assertEqual(r["integrity"], {"truncated": False})
        self.assertIsNone(r["teams_transcript"])
        self.assertAlmostEqual(r["duration"], self.duration, places=2)
        json.dumps(r)
        for seg in r["segments"]:
            self.assertEqual(set(seg), {"start", "end", "text", "language", "words"})
            for w in seg["words"]:
                self.assertEqual(set(w), {"start", "end", "word"})

    def test_language(self):
        r = self.result
        self.assertEqual(r["language"], "en")
        self.assertGreater(r["language_probability"], 0.9)
        self.assertEqual(set(r["languages"]), {"en", "af"})
        self.assertGreater(r["languages"]["en"], 30.0)
        self.assertEqual(r["languages"]["af"], 0.0)
        self.assertTrue(all(s["language"] == "en" for s in r["segments"]))

    def test_words_monotonic_within_file(self):
        words = [w for s in self.result["segments"] for w in s["words"]]
        self.assertGreater(len(words), 80)
        for w in words:
            self.assertGreaterEqual(w["start"], 0.0)
            self.assertLessEqual(w["end"], self.duration + EPS)
            self.assertLessEqual(w["start"], w["end"])
        for a, b in zip(words, words[1:]):
            self.assertLessEqual(a["start"], b["start"])
            self.assertLessEqual(a["end"], b["start"] + EPS)
        for s in self.result["segments"]:
            self.assertLessEqual(s["start"], s["end"])
        self.assertGreater(words[-1]["end"], 40.0)  # the second chunk was offset, not left at chunk time

    def test_content_and_progress(self):
        text = " ".join(s["text"] for s in self.result["segments"])
        self.assertIn("sourdough", text)
        self.assertIn("price list", text)
        self.assertGreaterEqual(len(self.fractions), 2)
        self.assertEqual(self.fractions, sorted(self.fractions))
        self.assertAlmostEqual(self.fractions[-1], 1.0)


class TestTranscribeFileSilence(unittest.TestCase):
    def test_silent_file(self):
        import tempfile
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "silence.wav"
            sf.write(str(p), np.zeros(16000 * 5, np.float32), 16000, subtype="PCM_16")
            fractions = []
            r = asr.transcribe_file(p, "large-v3", ["en", "af"], [], progress=fractions.append)
        self.assertEqual(r["segments"], [])
        self.assertEqual(r["languages"], {"en": 0.0, "af": 0.0})
        self.assertEqual(fractions, [1.0])


if __name__ == "__main__":
    unittest.main()
