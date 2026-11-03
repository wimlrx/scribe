"""Local speech recognition with faster-whisper on CUDA: a resident model for dictation, and whole-file
transcription for meetings and drop-ins. Nothing here leaves the machine."""
from __future__ import annotations

import gc
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

# Models come only from the local Hugging Face cache, fetched once by hand at install. Set before
# faster_whisper (and huggingface_hub with it) is imported, so nothing can reach huggingface.co.
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402

log = logging.getLogger("scribe.asr")

SAMPLE_RATE = 16000
DEVICE = "cuda"
COMPUTE = "float16"
CPU_DEVICE = "cpu"            # where Whisper goes when the GPU has no memory left for it
CPU_COMPUTE = "int8"
CPU_THREADS = min(8, os.cpu_count() or 4)
GPU_TRIES = 3                 # a file whose GPU runs out of memory gets this many GPU tries in all ...
GPU_RETRY_S = 15.0            # ... this far apart, then moves to the CPU so the job still finishes
LANG_WINDOW_S = 30.0          # Whisper judges language from one 30 s window
CHUNK_MAX_S = 28.0            # under Whisper's 30 s window, so each chunk is decoded in a single pass
CHUNK_MIN_S = 14.0            # prefer cuts that leave a chunk at least this long, for context
VAD_MIN_SILENCE_MS = 500
MIN_RMS = 0.004               # about -48 dBFS on the loudest frames; below it a clip is treated as silence
MIN_SAMPLES = SAMPLE_RATE // 10
NO_SPEECH_DROP = 0.6          # a segment is dropped when no_speech_prob is above this ...
LOGPROB_DROP = -1.0           # ... and avg_logprob is below this
PHANTOM_NO_SPEECH = 0.3       # a stock phrase is dropped on this weaker no-speech signal
PHANTOMS = frozenset({        # what large-v3 says over silence; compared lower-case, punctuation removed
    "you", "thank you", "thank you very much", "thank you so much", "thanks", "thanks for watching",
    "thank you for watching", "thank you so much for watching", "thank you for listening", "bye",
    "please subscribe", "subtitles by the amara org community", "dankie", "baie dankie",
    "dankie vir die kyk", "dankie dat jy gekyk het",
})
FILLER_RE = re.compile(r"^(?:u+m+|u+h+|u+h+m+|e+r+|a+h+|e+h+|h+m+)$", re.I)
CODE_SWITCH_SAMPLE = ("Ja, so ons moet eers die numbers check, and then we send it to the client. "
                      "Dis reg, ek sal die scope note skryf.")
_SENT_END = (".", "?", "!", "…")


# ---------------------------------------------------------------- shared helpers

def fmt_ts(seconds: float) -> str:
    """mm:ss, or h:mm:ss past an hour. Same as the transcribe skill, so loop markers read the same."""
    total = int(round(max(0, seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class ModelNotCached(RuntimeError):
    """The Whisper model is not complete in the local Hugging Face cache. Scribe never downloads one."""


def model_path(model_name: str) -> str:
    """Local folder for a model name, from the HF cache only. A missing or incomplete model raises
    ModelNotCached: models are fetched once by hand at install (README), never silently at run time."""
    from faster_whisper.utils import download_model

    if Path(model_name).is_dir():
        return model_name
    try:
        return download_model(model_name, local_files_only=True)
    except Exception as exc:
        log.error("model not cached: %s (%s)", model_name, type(exc).__name__)
        raise ModelNotCached(f"model not cached: {model_name}") from exc


def load_model(model_name: str, device: str = DEVICE):
    """WhisperModel on CUDA fp16, as DESIGN.md fixes it, or on the CPU (int8) when the GPU has no memory for it.
    A failure raises; it never triggers a download."""
    from faster_whisper import WhisperModel

    compute = COMPUTE if device == DEVICE else CPU_COMPUTE
    extra = {"cpu_threads": CPU_THREADS} if device == CPU_DEVICE else {}
    t0 = time.perf_counter()
    model = WhisperModel(model_path(model_name), device=device, compute_type=compute, **extra)
    log.info("loaded %s on %s %s in %.1f s", model_name, device, compute, time.perf_counter() - t0)
    return model


def out_of_memory(exc: BaseException) -> bool:
    """A CUDA allocation failure. ctranslate2 raises RuntimeError("CUDA failed with error out of memory") (forced
    on this machine, 31 Oct 2026); cuBLAS and cuDNN name theirs ..._STATUS_ALLOC_FAILED."""
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc)
    return "out of memory" in text.lower() or "ALLOC_FAILED" in text


def as_mono_float32(audio: np.ndarray) -> np.ndarray:
    """float32 mono in -1..1, whatever the capture handed over (int16, stereo, float64)."""
    a = np.asarray(audio)
    if a.ndim == 2:
        a = a.mean(axis=1)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    return np.ascontiguousarray(a, dtype=np.float32).reshape(-1)


def speech_rms(audio: np.ndarray, frame: int = 480) -> float:
    """RMS of the loudest 10% of 30 ms frames (at least three). A short phrase inside a long silent hold
    still registers, while a single key click is diluted."""
    a = as_mono_float32(audio)
    if a.size == 0:
        return 0.0
    n = a.size // frame
    if n < 3:
        return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    ms = np.mean(a[: n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1)
    top = max(3, int(math.ceil(n * 0.1)))
    return float(np.sqrt(np.mean(np.sort(ms)[-top:])))


def _phrase_key(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def is_hallucination(no_speech_prob: float, avg_logprob: float, text: str = "") -> bool:
    """Whisper's own silence test (likely no speech and a low-confidence decode), plus the stock phrases
    large-v3 invents on room noise ("Thank you for watching."), which decode confidently: those are
    dropped on a weaker no-speech signal. Measured 31 Oct 2026: room tone at -44 dBFS gave
    'Thank you for watching.' at no_speech 0.82, avg_logprob -0.82."""
    if no_speech_prob > NO_SPEECH_DROP and avg_logprob < LOGPROB_DROP:
        return True
    return no_speech_prob > PHANTOM_NO_SPEECH and _phrase_key(text) in PHANTOMS


def vocab_prompt(vocabulary: list[str]) -> str | None:
    """The vocabulary as preceding text, so Whisper spells the user's names and terms their way."""
    terms = [v.strip() for v in vocabulary if v and v.strip()]
    return ", ".join(terms) + "." if terms else None


def apply_vocabulary(text: str, vocabulary: list[str]) -> str:
    """Respell vocabulary terms the user's way where Whisper heard the words but not the casing
    (large-v3-turbo writes 'project planner' even with the prompt). Whole words only; case and spacing differ."""
    terms = sorted({" ".join(v.split()) for v in vocabulary if v and v.strip()}, key=len, reverse=True)
    for term in terms:
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in term.split()) + r"(?!\w)"
        text = re.sub(pattern, lambda m, t=term: t, text, flags=re.IGNORECASE)
    return text


def file_prompt(vocabulary: list[str]) -> str:
    """Vocabulary plus a short mixed Afrikaans/English line, so code-switching is kept as spoken, not translated."""
    vp = vocab_prompt(vocabulary)
    return f"{vp} {CODE_SWITCH_SAMPLE}" if vp else CODE_SWITCH_SAMPLE


def _cap_first(tok: str) -> str:
    for i, ch in enumerate(tok):
        if ch.isalpha():
            return tok[:i] + ch.upper() + tok[i + 1:]
    return tok


def strip_fillers(text: str) -> str:
    """Drop standalone hesitations (um, uh, uhm, er, ah, eh, hmm and their stretched forms) and repair the
    punctuation around them: no orphan commas, a sentence-ending mark moves to the word before, and the word
    after a dropped sentence-opening filler is capitalised."""
    out: list[str] = []
    cap_next = False
    for tok in text.split():
        m = re.match(r"^([\"'(\[“‘]*)(.*?)([.,!?;:…\"')\]”’]*)$", tok)
        lead, core, trail = m.groups() if m else ("", tok, "")
        if not lead and FILLER_RE.match(core):
            at_start = not out or out[-1].endswith(_SENT_END)
            term = next((c for c in trail if c in ".?!…"), None)
            if term and not at_start:
                prev = out[-1].rstrip(",;:")
                out[-1] = prev if prev.endswith(_SENT_END) else prev + term
                cap_next = True
            elif at_start or term:
                cap_next = True
            elif "," in trail and out and out[-1].endswith(","):
                out[-1] = out[-1][:-1]
            continue
        if cap_next:
            tok = _cap_first(tok)
            cap_next = False
        out.append(tok)
    return " ".join(out)


# ---------------------------------------------------------------- language

def detect_language(model, audio: np.ndarray, allowed: list[str]) -> tuple[str, float]:
    """The allowed language most likely in the first 30 s, with its probability renormalised over the allowed
    set. Left free, Whisper reads accented English or Afrikaans as Dutch; the allowed set rules that out."""
    allowed = list(dict.fromkeys(a for a in allowed if a))
    supported = getattr(model, "supported_languages", None)
    if supported is not None:
        allowed = [a for a in allowed if a in supported]
    if len(allowed) == 1:
        return allowed[0], 1.0
    clip = as_mono_float32(audio)[: int(LANG_WINDOW_S * SAMPLE_RATE)]
    if allowed and clip.size < MIN_SAMPLES:
        return allowed[0], 1.0 / len(allowed)
    top, top_p, all_probs = model.detect_language(audio=clip)
    if not allowed:
        log.warning("no usable allowed languages; using Whisper's own pick %s", top)
        return top, float(top_p)
    probs = dict(all_probs or [])
    scores = [float(probs.get(a, 0.0)) for a in allowed]
    total = sum(scores)
    if total <= 0.0:
        return allowed[0], 1.0 / len(allowed)
    best = max(range(len(allowed)), key=lambda i: scores[i])
    return allowed[best], scores[best] / total


# ---------------------------------------------------------------- dictation

class DictationASR:
    """Resident model for hold-to-talk dictation. Loaded once at startup so release-to-text stays short.

    Dictation never waits for the GPU. A CUDA out-of-memory error, at load or on a clip, sends that clip to a CPU
    model (int8, kept once loaded). The model that failed is unusable afterwards (forced on this machine, 31 Oct
    2026), so it is dropped, and a fresh GPU model is loaded on its own thread after a later clip, at most once
    per GPU_RETRY_S; clips use the CPU until it is in."""

    def __init__(self, model_name: str, languages: list[str], vocabulary: list[str],
                 strip_fillers: bool = True, min_rms: float = MIN_RMS,
                 prompt_terms: Callable[[], list[str]] | None = None):
        self.model_name = model_name
        self.languages = list(languages)
        self.vocabulary = list(vocabulary)
        self.prompt_terms = prompt_terms     # learned spellings: Whisper's prompt only, never forced on the text
        self.strip_fillers = strip_fillers
        self.min_rms = min_rms
        self.model = None                    # on the GPU
        self.cpu_model = None                # loaded the first time the GPU runs out of memory
        self._gpu_lost_at: float | None = None
        self._reloading = False
        self._generation = 0                 # unload() makes a GPU load still under way stale
        self._lock = threading.RLock()

    @property
    def device(self) -> str | None:
        """Where the next clip runs: "cuda", "cpu", or None before load()."""
        return DEVICE if self.model is not None else (CPU_DEVICE if self.cpu_model is not None else None)

    def load(self) -> None:
        """Load the model onto the GPU; it stays resident until unload(). With no GPU memory for it, the CPU
        model is loaded instead."""
        with self._lock:
            if self.model is not None or self.cpu_model is not None:
                return
            try:
                self.model = load_model(self.model_name)
            except Exception as exc:
                if not out_of_memory(exc):
                    raise
            if self.model is None:
                self._gpu_lost("load")
                self._cpu()

    def unload(self) -> None:
        """Free the GPU memory (tests, or the app shutting down)."""
        with self._lock:
            self.model = self.cpu_model = None
            self._generation += 1
        gc.collect()

    def transcribe(self, audio: np.ndarray) -> str:
        """16 kHz mono audio to text. Silence returns "" rather than Whisper's invented "Thank you."."""
        a = as_mono_float32(audio)
        level = speech_rms(a)
        if a.size < MIN_SAMPLES or level < self.min_rms:
            log.info("dictation gated: %.2f s, level %.4f", a.size / SAMPLE_RATE, level)
            return ""
        t0 = time.perf_counter()
        prompt = vocab_prompt(self._prompt_vocabulary())
        with self._lock:
            self.load()
            device, result = self.device, None
            try:
                result = self._run(self.model if self.model is not None else self.cpu_model, a, prompt)
            except Exception as exc:
                if device != DEVICE or not out_of_memory(exc):
                    raise
            if result is None:          # out of GPU memory: this clip on the CPU, at once
                self._gpu_lost("clip")
                device, result = CPU_DEVICE, self._run(self._cpu(), a, prompt)
        if device == CPU_DEVICE:
            self._regain_gpu()
        lang, p, parts = result
        text = " ".join(" ".join(parts).split())
        if self.strip_fillers:
            text = strip_fillers(text)
        text = apply_vocabulary(text, self.vocabulary)
        log.info("dictation %.2f s -> %d chars, %s p=%.2f, %s, %.0f ms",
                 a.size / SAMPLE_RATE, len(text), lang, p, device, (time.perf_counter() - t0) * 1000)
        return text

    def _run(self, model, a: np.ndarray, prompt: str | None) -> tuple[str, float, list[str]]:
        lang, p = detect_language(model, a, self.languages)
        parts = [text for clip, stamps in self._pieces(a) for text in self._decode(model, clip, lang, stamps, prompt)]
        return lang, p, parts

    def _cpu(self):
        if self.cpu_model is None:
            self.cpu_model = load_model(self.model_name, CPU_DEVICE)
        return self.cpu_model

    def _gpu_lost(self, where: str) -> None:
        """The GPU ran out of memory at load or on a clip: that model is dropped; the CPU takes the clips."""
        log.warning("dictation: CUDA out of memory (%s); on the CPU until the GPU model is back", where)
        self.model = None
        self._gpu_lost_at = time.monotonic()
        gc.collect()

    def _regain_gpu(self) -> None:
        """After a clip on the CPU: a fresh GPU model, loaded on its own thread, at most once per GPU_RETRY_S."""
        with self._lock:
            if self.model is not None or self._reloading or (
                    self._gpu_lost_at is not None and time.monotonic() - self._gpu_lost_at < GPU_RETRY_S):
                return
            self._reloading = True
            generation = self._generation
        threading.Thread(target=self._reload, args=(generation,), name="scribe-asr-gpu", daemon=True).start()

    def _reload(self, generation: int) -> None:
        model = None
        try:
            model = load_model(self.model_name)
        except Exception as exc:
            if out_of_memory(exc):
                log.warning("dictation: GPU still out of memory; staying on the CPU")
            else:
                log.exception("dictation: GPU model not reloaded")
        with self._lock:
            self._reloading = False
            if model is None:
                self._gpu_lost_at = time.monotonic()
            elif generation == self._generation and self.model is None:
                self.model = model
                log.info("dictation back on the GPU")
        model = None
        gc.collect()

    def _pieces(self, a: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        """(clip, with timestamps) to decode. A hold over CHUNK_MAX_S is cut into VAD chunks of at most 28 s:
        decoded without timestamps, Whisper steps a blind 30 s per window and loses the speech that crosses
        each boundary. Each chunk also gets the vocabulary prompt, which only reaches a call's first window."""
        if a.size <= CHUNK_MAX_S * SAMPLE_RATE:
            return [(a, False)]
        chunks = plan_chunks(speech_regions(a))
        if not chunks:
            return [(a, True)]          # VAD heard nothing the energy gate let through: let Whisper seek itself
        out = []
        for c0, c1 in chunks:
            clip = a[int(round(c0 * SAMPLE_RATE)):int(round(c1 * SAMPLE_RATE))]
            if clip.size:
                out.append((clip, False))
        log.info("dictation %.1f s decoded in %d chunks", a.size / SAMPLE_RATE, len(out))
        return out

    def _prompt_vocabulary(self) -> list[str]:
        """Learned terms the vocabulary does not already hold, least used first, then the configured vocabulary.
        faster-whisper keeps only the prompt's last 223 tokens, so a long learned list loses its least used
        terms first and never pushes out the configured vocabulary."""
        extra = []
        if self.prompt_terms is not None:
            try:
                extra = list(self.prompt_terms() or [])     # most used first
            except Exception:
                log.exception("learned terms unavailable")
        have = {" ".join(v.lower().split()) for v in self.vocabulary if v}
        learned = [t for t in extra if t and " ".join(t.lower().split()) not in have]
        return learned[::-1] + self.vocabulary

    def _decode(self, model, clip: np.ndarray, lang: str, timestamps: bool, prompt: str | None = None) -> list[str]:
        segments, _ = model.transcribe(
            clip, language=lang, beam_size=5, condition_on_previous_text=False,
            without_timestamps=not timestamps, initial_prompt=prompt)
        return [s.text for s in segments if not is_hallucination(s.no_speech_prob, s.avg_logprob, s.text)]


# ---------------------------------------------------------------- chunking

def _normalise_regions(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted((float(s), float(e)) for s, e in regions if e > s):
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def plan_chunks(regions: list[tuple[float, float]], max_s: float = CHUNK_MAX_S,
                min_s: float = CHUNK_MIN_S) -> list[tuple[float, float]]:
    """Group speech regions (seconds) into contiguous spans of at most max_s. When the next region would
    overflow, the chunk is cut at the longest pause it can end on, choosing among cuts that keep at least
    min_s when any do (ties go to the later cut, for context). A region longer than max_s on its own is cut
    into equal pieces."""
    regs = _normalise_regions(regions)
    out: list[tuple[float, float]] = []
    i, n = 0, len(regs)
    while i < n:
        start = regs[i][0]
        if regs[i][1] - start > max_s:
            s, e = regs[i]
            k = max(1, math.ceil((e - s) / max_s - 1e-9))
            step = (e - s) / k
            out.extend((s + j * step, e if j == k - 1 else s + (j + 1) * step) for j in range(k))
            i += 1
            continue
        j = i
        while j + 1 < n and regs[j + 1][1] - start <= max_s:
            j += 1
        if j == n - 1:
            out.append((start, regs[j][1]))
            break
        cands = list(range(i, j + 1))
        pool = [k for k in cands if regs[k][1] - start >= min_s] or cands
        k = max(pool, key=lambda k: (regs[k + 1][0] - regs[k][1], k))
        out.append((start, regs[k][1]))
        i = k + 1
    return out


def speech_regions(audio: np.ndarray) -> list[tuple[float, float]]:
    """Silero VAD speech regions in seconds. max_speech_duration_s keeps an unbroken monologue from
    becoming one region longer than a chunk: the VAD cuts it in its last short silence instead."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    if audio.size == 0:
        return []
    opts = VadOptions(min_silence_duration_ms=VAD_MIN_SILENCE_MS, max_speech_duration_s=CHUNK_MAX_S)
    return [(r["start"] / SAMPLE_RATE, r["end"] / SAMPLE_RATE)
            for r in get_speech_timestamps(audio, opts, sampling_rate=SAMPLE_RATE)]


def _overlap_seconds(regions: list[tuple[float, float]], a: float, b: float) -> float:
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in regions)


# ---------------------------------------------------------------- loops

def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def collapse_word_loops(words: list[dict]) -> list[dict]:
    """A phrase of 3-15 words repeated three or more times back to back is a whisper loop: keep one copy, mark it.
    Identical to the transcribe skill's rule."""
    out: list[dict] = []
    i = 0
    n_words = len(words)
    while i < n_words:
        hit = None
        for n in range(15, 2, -1):
            if i + 3 * n > n_words:
                continue
            phrase = [_norm(w["word"]) for w in words[i:i + n]]
            reps = 1
            while i + (reps + 1) * n <= n_words and [_norm(w["word"]) for w in words[i + reps * n:i + (reps + 1) * n]] == phrase:
                reps += 1
            if reps >= 3:
                hit = (n, reps)
                break
        if hit:
            n, reps = hit
            out.extend(words[i:i + n])
            last = words[i + reps * n - 1]
            out.append({"start": last["start"], "end": last["end"], "word": f"[repetition loop removed, {reps - 1} repeats to {fmt_ts(last['end'])}]"})
            i += reps * n
        else:
            out.append(words[i])
            i += 1
    return out


def collapse_segment_loops(segments: list[dict]) -> list[dict]:
    """Run the loop rule over the whole word stream (loops can cross segment lines), then rebuild segments:
    a segment that was wholly repeats goes, a trimmed one gets its text and times from the words it kept,
    and the marker stays with the copy that was kept."""
    flat: list[dict] = []
    owner: dict[int, int] = {}
    for si, seg in enumerate(segments):
        for w in seg["words"]:
            owner[id(w)] = si
            flat.append(w)
    collapsed = collapse_word_loops(flat)
    if len(collapsed) == len(flat):
        return segments
    kept: dict[int, list[dict]] = {si: [] for si in range(len(segments))}
    last = 0
    for w in collapsed:
        last = owner.get(id(w), last)
        kept[last].append(w)
    out: list[dict] = []
    for si, seg in enumerate(segments):
        ws = kept[si]
        if not seg["words"]:
            out.append(seg)
        elif ws:
            if len(ws) != len(seg["words"]) or any(a is not b for a, b in zip(ws, seg["words"])):
                seg = {**seg, "words": ws, "text": " ".join(w["word"] for w in ws),
                       "start": ws[0]["start"], "end": ws[-1]["end"]}
            out.append(seg)
    return out


# ---------------------------------------------------------------- files

def _segment_row(seg, c0: float, c1: float, lang: str) -> dict | None:
    """One Whisper segment in file time, clamped to its chunk. None for empty or hallucinated text."""
    text = " ".join((seg.text or "").split())
    if not text or is_hallucination(seg.no_speech_prob, seg.avg_logprob, text):
        return None

    def clamp(t: float) -> float:
        return min(max(c0 + float(t), c0), c1)

    words = []
    for w in seg.words or []:
        word = w.word.strip()
        if word:
            s = clamp(w.start)
            words.append({"start": s, "end": max(clamp(w.end), s), "word": word})
    start = clamp(seg.start)
    return {"start": start, "end": max(clamp(seg.end), start), "text": text, "language": lang, "words": words}


def make_monotonic(segments: list[dict]) -> None:
    """In place: no word starts before the previous one ends, no end before its start, segment times
    follow their words. Whisper's alignment occasionally overlaps by a frame."""
    prev = 0.0
    for seg in segments:
        for w in seg["words"]:
            w["start"] = max(w["start"], prev)
            w["end"] = max(w["end"], w["start"])
            prev = w["end"]
        if seg["words"]:
            seg["start"], seg["end"] = seg["words"][0]["start"], seg["words"][-1]["end"]
        else:
            seg["start"] = max(seg["start"], prev)
            seg["end"] = max(seg["end"], seg["start"])
            prev = seg["end"]


def _round_times(segments: list[dict]) -> None:
    for seg in segments:
        seg["start"], seg["end"] = round(seg["start"], 3), round(seg["end"], 3)
        for w in seg["words"]:
            w["start"], w["end"] = round(w["start"], 3), round(w["end"], 3)


def _report(progress: Callable[[float], None] | None, fraction: float) -> None:
    if progress is None:
        return
    try:
        progress(fraction)
    except Exception:
        log.exception("progress callback failed")


class _FileModel:
    """The model one file is transcribed on. A CUDA out-of-memory error, at load or mid-file, leaves that model
    unusable (forced on this machine, 31 Oct 2026: its next call fails with cudaErrorInvalidDevice, while other
    models in the process carry on), so each new try loads a fresh one: on the GPU up to GPU_TRIES times in all,
    GPU_RETRY_S apart, then on the CPU (int8) for the rest of the file. on_wait(True) and on_wait(False) bracket
    each wait."""

    def __init__(self, model_name: str, on_wait: Callable[[bool], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.name, self.on_wait, self.sleep = model_name, on_wait, sleep
        self.device = DEVICE
        self.model = None
        self.ooms = 0

    def get(self):
        """The model to run on, loaded when there is none."""
        while self.model is None:
            try:
                self.model = load_model(self.name, self.device)
            except Exception as exc:
                if not self.oom(exc):
                    raise
            if self.model is None:
                self.retry()
        return self.model

    def oom(self, exc: BaseException) -> bool:
        """True when exc is the GPU running out of memory: the model is then replaced through retry()."""
        return self.device == DEVICE and out_of_memory(exc)

    def retry(self) -> None:
        """Called after the except block, so its traceback no longer holds the failed model: drop it, then wait
        for another GPU try, or move to the CPU once GPU_TRIES are spent."""
        self.model = None
        gc.collect()
        self.ooms += 1
        if self.ooms >= GPU_TRIES:
            log.warning("%s: CUDA out of memory %d times; on the CPU (%s) from here", self.name, self.ooms,
                        CPU_COMPUTE)
            self.device = CPU_DEVICE
            return
        log.warning("%s: CUDA out of memory (%d of %d tries); the GPU again in %.0f s", self.name, self.ooms,
                    GPU_TRIES, GPU_RETRY_S)
        self._tell(True)
        try:
            self.sleep(GPU_RETRY_S)
        finally:
            self._tell(False)

    def _tell(self, on: bool) -> None:
        if self.on_wait is None:
            return
        try:
            self.on_wait(on)
        except Exception:
            log.exception("on_wait callback failed")


def _model_label(model_name: str, gpu_chunks: int, cpu_from: float | None) -> str:
    """whisper.json's "model": where the file was transcribed, the CPU's start time when it moved part-way."""
    if cpu_from is None:
        return f"faster-whisper {model_name} ({DEVICE}, {COMPUTE})"
    if not gpu_chunks:
        return f"faster-whisper {model_name} ({CPU_DEVICE}, {CPU_COMPUTE})"
    return f"faster-whisper {model_name} ({DEVICE}, {COMPUTE}; {CPU_DEVICE}, {CPU_COMPUTE} from {fmt_ts(cpu_from)})"


def transcribe_file(audio_path: Path, model_name: str, languages: list[str], vocabulary: list[str],
                    progress: Callable[[float], None] | None = None,
                    on_wait: Callable[[bool], None] | None = None) -> dict:
    """A whole recording to the transcribe skill's whisper.json schema, plus per-segment language and seconds
    per language. VAD cuts the file into chunks of at most 28 s at pauses; each chunk gets its own language
    from the allowed set, so a meeting that moves between Afrikaans and English is decoded in the language
    actually spoken. The model is loaded for this call and freed at the end. When the GPU runs out of memory the
    chunk is tried again (_FileModel); on_wait(True) while it waits, on_wait(False) when it goes on."""
    from faster_whisper import decode_audio

    audio_path = Path(audio_path)
    t0 = time.perf_counter()
    audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    duration = audio.shape[0] / SAMPLE_RATE
    regions = speech_regions(audio)
    chunks = plan_chunks(regions)
    log.info("%s: %.1f s, %d speech regions, %d chunks", audio_path.name, duration, len(regions), len(chunks))

    seconds: dict[str, float] = {lang: 0.0 for lang in languages}
    weighted: dict[str, float] = {}
    segments: list[dict] = []
    gpu_chunks, cpu_from = 0, None
    if chunks:
        prompt = file_prompt(vocabulary)
        whisper = _FileModel(model_name, on_wait)
        try:
            for idx, (c0, c1) in enumerate(chunks):
                a0, a1 = int(round(c0 * SAMPLE_RATE)), int(round(c1 * SAMPLE_RATE))
                start, end = a0 / SAMPLE_RATE, a1 / SAMPLE_RATE
                clip = audio[a0:a1]
                while True:
                    model = whisper.get()
                    try:
                        lang, p = detect_language(model, clip, languages)
                        segs, _ = model.transcribe(
                            clip, language=lang, beam_size=5, word_timestamps=True,
                            condition_on_previous_text=False, vad_filter=False, initial_prompt=prompt)
                        rows = [row for row in (_segment_row(seg, start, end, lang) for seg in segs) if row]
                        break
                    except Exception as exc:
                        if not whisper.oom(exc):
                            raise
                    finally:
                        model = segs = None
                    whisper.retry()         # this chunk again, on a fresh model
                if whisper.device == DEVICE:
                    gpu_chunks += 1
                elif cpu_from is None:
                    cpu_from = start
                speech = _overlap_seconds(regions, c0, c1)
                seconds[lang] = seconds.get(lang, 0.0) + speech
                weighted[lang] = weighted.get(lang, 0.0) + p * speech
                segments.extend(rows)
                _report(progress, (idx + 1) / len(chunks))
        finally:
            whisper.model = None
            gc.collect()
    else:
        _report(progress, 1.0)

    make_monotonic(segments)
    segments = collapse_segment_loops(segments)
    _round_times(segments)
    main = max(seconds, key=lambda k: seconds[k]) if any(seconds.values()) else (languages[0] if languages else "en")
    prob = weighted.get(main, 0.0) / seconds[main] if seconds.get(main) else 0.0
    label = _model_label(model_name, gpu_chunks, cpu_from)
    log.info("%s: %d segments, %s, %.0f s, %s", audio_path.name, len(segments),
             ", ".join(f"{k} {v:.0f} s" for k, v in seconds.items()), time.perf_counter() - t0, label)
    return {
        "source": audio_path.name,
        "model": label,
        "language": main,
        "language_probability": round(float(prob), 4),
        "languages": {k: round(v, 1) for k, v in seconds.items()},
        "duration": float(duration),
        "integrity": {"truncated": False},
        "teams_transcript": None,
        "segments": segments,
    }
