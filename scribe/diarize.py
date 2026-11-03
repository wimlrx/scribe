"""Local speaker labels: sherpa-onnx diarisation (pyannote segmentation-3.0 + WeSpeaker ResNet34), voices it merged
parted by pitch, the owner's voiceprint, and word-to-speaker assignment. Runs on CPU so the GPU stays free for
Whisper."""
from __future__ import annotations

import argparse
import bisect
import itertools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("scribe.diarize")

SAMPLE_RATE = 16000
SEGMENTATION_MODEL = Path("sherpa-onnx-pyannote-segmentation-3-0") / "model.onnx"
EMBEDDING_MODEL = "wespeaker_en_voxceleb_resnet34_LM.onnx"
MIN_ENROLL_SECONDS = 2.0      # speech needed before a clip may move the voiceprint
MIN_NAMING_SEGMENT = 1.5      # shorter segments give unstable embeddings
MAX_NAMING_SEGMENTS = 8
MAX_EMBED_SECONDS = 30.0      # a long monologue segment is embedded from its middle 30 s; CPU time stays bounded
MIN_DIARIZE_SECONDS = 0.25    # below this there is nothing to label (and sherpa refuses an empty buffer)
THREADS = max(1, min(8, (os.cpu_count() or 2) // 2))
ENROLL_AGREE = 0.6            # a clip joins the voiceprint only this close to it (owner 0.98, colleague 0.38)
PENDING_MAX = 6               # candidate clips kept while the first three agreeing ones are awaited
FIRST_CLIPS = 3               # the voiceprint is committed once this many candidates agree pairwise
OWNER_BAND = 0.1              # clusters this close to the best match are the owner's voice split by clustering
CLEAR_MATCH = 0.7             # below this, the best match must also beat every other cluster ...
OWNER_MARGIN = 0.2            # ... by this much, or the owner counts as not matched
CHILD_TIMEOUT_S = 1800.0      # the labelling child gets this long plus twice the audio length
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Pitch. On the laptop microphone a man's and a woman's speaker embeddings sat at cosine 0.75-0.83, so sherpa put
# them in one cluster at 0.5; their pitch registers were 126 Hz and 188 Hz (a recorded meeting).
PITCH_HOP_S = 0.01            # one pitch reading every 10 ms ...
PITCH_WINDOW_S = 0.025        # ... over 25 ms (YIN's integration window)
F0_MIN, F0_MAX = 60.0, 400.0  # speaking pitch, deep man to high woman
YIN_THRESHOLD = 0.2           # YIN's aperiodicity cut: a frame below it is voiced (0.15 found 6 frames in her short
                              # reply, 0.2 found 8; the man's median and 95th percentile did not move)
VOICED_GATE = 0.2             # a frame's pitch counts at >= this share of its segment's loud level (90th percentile
                              # RMS); the man's quiet trailing syllables read at 2x and 4x his pitch
PAUSE_GATE = 0.15             # quieter than this share of the loud level ...
MIN_PAUSE_S = 0.15            # ... for this long is a pause, where the turn may pass to someone else
MIN_PART_PITCH_S = 0.05       # a stretch between pauses needs this much voiced speech for a pitch (her reply: 0.07 s)
PITCH_SPLIT = 1.35            # registers this far apart (about 5 semitones) are two voices: the man's stretches lay within
                              # 1.26 of his register, the woman's sat 1.46-1.49 above it
PITCH_GAP = 1.15              # two voices leave a gap this wide between their stretches; one voice's stretches,
                              # sorted, stepped by at most 1.12 (the man's, and both fixture voices)
PITCH_SAME = 1.25             # a stretch this close to a voice's register may be that voice: a short reply read up to
                              # 1.21 from its own voice (fixture)
MIN_VOICE_PITCH_S = 0.5       # a pitch group with this much voiced speech may stand as a voice of its own
SAME_VOICE = 0.9              # a pitch split is refused when the two groups' embeddings agree this well: one voice's
                              # low and high halves 0.93-0.95 on the fixtures (0.87 the man's, on the laptop microphone)


# ---------------------------------------------------------------- models and audio

def segmentation_model(models_dir: Path) -> Path:
    path = Path(models_dir) / SEGMENTATION_MODEL
    if not path.is_file():
        raise FileNotFoundError(f"segmentation model missing: {path}")
    return path


def embedding_model(models_dir: Path) -> Path:
    path = Path(models_dir) / EMBEDDING_MODEL
    if not path.is_file():
        raise FileNotFoundError(f"speaker-embedding model missing: {path}")
    return path


def read_audio(path: Path) -> np.ndarray:
    """Whole file as float32 16 kHz mono. mix.flac already is; anything else is downmixed and resampled."""
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if rate != SAMPLE_RATE:
        import soxr

        audio = soxr.resample(audio, rate, SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _mono(audio: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("degenerate speaker embedding")
    return (vec / norm).astype(np.float32)


def _middle(audio: np.ndarray, seconds: float) -> np.ndarray:
    keep = int(seconds * SAMPLE_RATE)
    if audio.size <= keep:
        return audio
    start = (audio.size - keep) // 2
    return audio[start:start + keep]


# ---------------------------------------------------------------- diarisation

def diarize(audio_path: Path, models_dir: Path, threshold: float = 0.5, num_speakers: int | None = None) -> list[dict]:
    """[{"start", "end", "speaker"}] sorted by start; speakers numbered 0, 1, ... in order of first appearance.
    `threshold` is sherpa's clustering distance: larger merges more. `num_speakers` fixes the count when known."""
    import sherpa_onnx

    seg_model = segmentation_model(models_dir)
    emb_model = embedding_model(models_dir)
    audio = read_audio(Path(audio_path))
    if audio.size < MIN_DIARIZE_SECONDS * SAMPLE_RATE:
        log.info("diarize: %s holds %.2f s of audio, nothing to label", audio_path, audio.size / SAMPLE_RATE)
        return []
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model)),
            num_threads=THREADS,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=THREADS, provider="cpu"),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=int(num_speakers) if num_speakers else -1, threshold=float(threshold)
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError("sherpa-onnx rejected the diarisation config")
    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    if engine.sample_rate != SAMPLE_RATE:
        import soxr

        audio = np.ascontiguousarray(soxr.resample(audio, SAMPLE_RATE, engine.sample_rate), dtype=np.float32)

    last_decile = [-1]

    def progress(done: int, total: int) -> int:
        decile = (10 * done) // max(total, 1)
        if decile != last_decile[0]:
            last_decile[0] = decile
            log.debug("diarize: %d/%d chunks", done, total)
        return 0

    began = time.monotonic()
    result = engine.process(audio, callback=progress).sort_by_start_time()
    order: dict[int, int] = {}
    out = []
    for seg in result:
        speaker = order.setdefault(int(seg.speaker), len(order))
        out.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "speaker": speaker})
    log.info(
        "diarize: %s -> %d segments, %d speakers (threshold %.2f, %s) in %.1f s",
        Path(audio_path).name, len(out), len(order), threshold,
        f"{num_speakers} fixed" if num_speakers else "auto", time.monotonic() - began,
    )
    return out


# ---------------------------------------------------------------- pitch

def pitch_frames(audio: np.ndarray, diar: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(log2 pitch in Hz, RMS) per PITCH_HOP_S frame over the whole recording, computed inside the diarised speech
    only; pitch is NaN where a frame is unvoiced or outside the speech. YIN (de Cheveigne and Kawahara 2002) in
    numpy: no pitch tracker is installed, and torchaudio would add seconds of torch import to the labelling child."""
    audio = _mono(audio)
    hop = int(round(PITCH_HOP_S * SAMPLE_RATE))
    win = int(round(PITCH_WINDOW_S * SAMPLE_RATE))
    tau_min, tau_max = int(SAMPLE_RATE / F0_MAX), int(SAMPLE_RATE / F0_MIN)
    length = win + tau_max
    count = audio.size // hop
    pitch = np.full(count, np.nan)
    rms = np.zeros(count)
    want = np.zeros(count, bool)
    for seg in diar:
        lo, hi = int(float(seg["start"]) / PITCH_HOP_S), int(np.ceil(float(seg["end"]) / PITCH_HOP_S))
        want[max(0, lo):max(0, hi)] = True
    frames = np.flatnonzero(want[:max(0, (audio.size - length) // hop + 1)])
    nfft = 1 << int(np.ceil(np.log2(length)))       # lags 0..tau_max of a win-long window never wrap
    taus = np.arange(tau_max + 1)
    for chunk in np.array_split(frames, max(1, int(np.ceil(frames.size / 2048)))):
        if chunk.size == 0:
            continue
        x = audio[chunk[:, None] * hop + np.arange(length)].astype(np.float64)
        x -= x[:, :win].mean(axis=1, keepdims=True)
        corr = np.fft.irfft(np.conj(np.fft.rfft(x[:, :win], nfft)) * np.fft.rfft(x, nfft), nfft)[:, :tau_max + 1]
        energy = np.concatenate([np.zeros((chunk.size, 1)), np.cumsum(x * x, axis=1)], axis=1)
        e0 = energy[:, win]
        diff = np.maximum(e0[:, None] + energy[:, taus + win] - energy[:, taus] - 2.0 * corr, 0.0)
        running = np.cumsum(diff[:, 1:], axis=1)
        cmnd = np.ones_like(diff)
        np.divide(diff[:, 1:] * taus[1:], running, out=cmnd[:, 1:], where=running > 1e-12)
        below = cmnd < YIN_THRESHOLD
        below[:, :tau_min] = False
        first = np.argmax(below, axis=1)
        voiced = below[np.arange(chunk.size), first]
        # the dip: from the first lag under the threshold to where the curve climbs back over it
        after = taus[None, :] >= first[:, None]
        broken = np.cumsum(after & ~below, axis=1) > 0
        tau = np.argmin(np.where(after & below & ~broken, cmnd, np.inf), axis=1)
        inner = (tau > 0) & (tau < tau_max)
        t = np.clip(tau, 1, tau_max - 1)
        rows = np.arange(chunk.size)
        a, b, c = cmnd[rows, t - 1], cmnd[rows, t], cmnd[rows, t + 1]
        curve = a - 2.0 * b + c
        shift = np.where(inner & (np.abs(curve) > 1e-12), 0.5 * (a - c) / np.where(curve == 0, 1.0, curve), 0.0)
        period = tau + np.clip(shift, -1.0, 1.0)
        rms[chunk] = np.sqrt(e0 / win)
        pitch[chunk] = np.where(voiced & (period > 0), np.log2(SAMPLE_RATE / np.maximum(period, 1e-9)), np.nan)
    return pitch, rms


def _frame_span(start: float, end: float, count: int) -> tuple[int, int]:
    return min(count, max(0, int(round(start / PITCH_HOP_S)))), min(count, max(0, int(round(end / PITCH_HOP_S))))


def _parts(diar: list[dict], frames: tuple[np.ndarray, np.ndarray]) -> list[dict]:
    """Each diarisation segment cut at its pauses: [{"start", "end", "speaker", "seg", "pitch", "voiced", "f0"}],
    "pitch" the median log2 Hz of the loud voiced frames (None below MIN_PART_PITCH_S of them), "voiced" their
    seconds, "f0" the frames themselves."""
    pitch, rms = frames
    out: list[dict] = []
    min_pause = int(round(MIN_PAUSE_S / PITCH_HOP_S))
    min_voiced = int(round(MIN_PART_PITCH_S / PITCH_HOP_S))
    for i, seg in enumerate(diar):
        start, end = float(seg["start"]), float(seg["end"])
        a, b = _frame_span(start, end, rms.size)
        loud = float(np.percentile(rms[a:b], 90)) if b > a else 0.0
        cuts = [start]
        if loud > 0:
            quiet = np.concatenate([[0], (rms[a:b] < PAUSE_GATE * loud).astype(np.int8), [0]])
            edges = np.diff(quiet)
            # cut in the middle of each pause; one at an edge of the segment leaves a quiet stretch with no pitch,
            # which follows its neighbour
            for k, j in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if j - k >= min_pause:
                    cuts.append(round(float((a + (k + j) / 2) * PITCH_HOP_S), 3))
        cuts.append(end)
        for lo, hi in zip(cuts, cuts[1:]):
            fa, fb = _frame_span(lo, hi, rms.size)
            f0 = pitch[fa:fb][(rms[fa:fb] >= VOICED_GATE * loud) & np.isfinite(pitch[fa:fb])] if loud > 0 else pitch[:0]
            out.append({"start": lo, "end": hi, "speaker": int(seg["speaker"]), "seg": i,
                        "pitch": float(np.median(f0)) if f0.size >= min_voiced else None,
                        "voiced": f0.size * PITCH_HOP_S, "f0": f0})
    return out


def _register(parts: list[dict]) -> float | None:
    """Median log2 pitch over the parts' loud voiced frames; None below MIN_PART_PITCH_S of them."""
    f0 = np.concatenate([p["f0"] for p in parts]) if parts else np.zeros(0)
    return float(np.median(f0)) if f0.size * PITCH_HOP_S >= MIN_PART_PITCH_S else None


def pitch_registers(audio: np.ndarray, diar: list[dict], frames=None) -> dict[int, float]:
    """{cluster: median pitch in Hz} for clusters with at least MIN_PART_PITCH_S of loud voiced speech."""
    parts = _parts(diar, frames if frames is not None else pitch_frames(audio, diar))
    out = {}
    for spk in sorted({p["speaker"] for p in parts}):
        reg = _register([p for p in parts if p["speaker"] == spk])
        if reg is not None:
            out[spk] = float(2.0 ** reg)
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2.0)])


def _two_groups(parts: list[dict]) -> tuple[list[dict], list[dict]] | None:
    """The parts with a pitch, cut in two at the widest gap in pitch, when that gap is at least PITCH_GAP and the
    two groups' centres (medians weighted by voiced seconds) are at least PITCH_SPLIT apart; else None. A gap, not
    the least spread: one woman's short reply inside a man's long cluster has almost no weight, and one man's
    utterances spread smoothly over his range with no gap that wide."""
    parts = sorted((p for p in parts if p["pitch"] is not None), key=lambda p: p["pitch"])
    if len(parts) < 2:
        return None
    x = np.array([p["pitch"] for p in parts])
    w = np.array([max(p["voiced"], PITCH_HOP_S) for p in parts])
    best, best_gap = None, np.log2(PITCH_GAP)
    for k in range(1, len(parts)):
        gap = x[k] - x[k - 1]
        if gap < best_gap:
            continue
        if _weighted_median(x[k:], w[k:]) - _weighted_median(x[:k], w[:k]) >= np.log2(PITCH_SPLIT):
            best, best_gap = k, gap
    return (parts[:best], parts[best:]) if best is not None else None


def _clip_of(audio: np.ndarray, parts: list[dict]) -> np.ndarray:
    """The longest parts joined, up to MAX_EMBED_SECONDS: an hour-long cluster is never copied whole."""
    pieces, total = [], 0
    for p in sorted(parts, key=lambda p: p["end"] - p["start"], reverse=True):
        piece = _middle(audio[int(p["start"] * SAMPLE_RATE):int(p["end"] * SAMPLE_RATE)], MAX_EMBED_SECONDS)
        pieces.append(piece[:max(0, int(MAX_EMBED_SECONDS * SAMPLE_RATE) - total)])
        total += pieces[-1].size
        if total >= MAX_EMBED_SECONDS * SAMPLE_RATE:
            break
    return np.concatenate(pieces) if pieces else audio[:0]


def _same_voice(audio: np.ndarray, a: list[dict], b: list[dict], embed) -> float | None:
    """Cosine between the embeddings of two groups of parts; None when either is too short to embed."""
    if embed is None:
        return None
    try:
        return _cosine(embed(_clip_of(audio, a)), embed(_clip_of(audio, b)))
    except ValueError:
        return None


def refine(audio: np.ndarray, diar: list[dict], embed=None, frames=None) -> list[dict]:
    """The diarisation with clusters that mix a low and a high voice taken apart by pitch.

    Each segment is cut at its pauses; a cluster whose parts fall into two pitch groups (_two_groups) holds two
    voices, unless the groups' embeddings agree (SAME_VOICE; `embed` is Voiceprint.embed, or None for no check).
    The group with less voiced speech joins the one other cluster whose register is within PITCH_SAME of it; with
    no such cluster it becomes a cluster of its own when it has MIN_VOICE_PITCH_S of voiced speech, else it stays.
    A part with no pitch follows the pitched parts beside it. Speakers are renumbered by first appearance;
    segments that nothing moved come back as they were."""
    audio = _mono(audio)
    if not diar:
        return []
    parts = _parts(diar, frames if frames is not None else pitch_frames(audio, diar))
    next_id = max(int(s["speaker"]) for s in diar) + 1
    for spk in sorted({p["speaker"] for p in parts}):
        groups = _two_groups([p for p in parts if p["speaker"] == spk])
        if groups is None:
            continue
        low, high = groups
        major, minor = (low, high) if sum(p["voiced"] for p in low) >= sum(p["voiced"] for p in high) else (high, low)
        centre = _register(minor)
        cos = _same_voice(audio, major, minor, embed)
        if cos is not None and cos >= SAME_VOICE:
            log.info("refine: cluster %d pitch %.0f/%.0f Hz kept as one voice (embeddings %.2f)",
                     spk, 2 ** _register(major), 2 ** centre, cos)
            continue
        regs = {k: _register([p for p in parts if p["speaker"] == k]) for k in {p["speaker"] for p in parts} - {spk}}
        near = [k for k, r in regs.items() if r is not None and abs(r - centre) <= np.log2(PITCH_SAME)]
        if len(near) == 1:
            target, why = near[0], f"joins cluster {near[0]}"
        elif sum(p["voiced"] for p in minor) >= MIN_VOICE_PITCH_S:
            target, why = next_id, "new cluster"
            next_id += 1
        else:
            continue
        for p in minor:
            p["speaker"] = target
        log.info("refine: cluster %d pitch %.0f Hz / %.0f Hz split (%s, embeddings %s): %s", spk,
                 2 ** _register(major), 2 ** centre, why, "n/a" if cos is None else f"{cos:.2f}",
                 ", ".join(f"{p['start']:.2f}-{p['end']:.2f}" for p in minor))
    # A stretch with no pitch of its own (a whisper, a fricative, a cough) goes with the pitched stretches beside it
    # in its segment when they agree.
    for _, group in itertools.groupby(parts, key=lambda p: p["seg"]):
        group = list(group)
        pitched = [k for k, p in enumerate(group) if p["pitch"] is not None]
        for k, p in enumerate(group):
            if p["pitch"] is None:
                at = bisect.bisect_left(pitched, k)
                side = {group[q]["speaker"] for q in pitched[max(0, at - 1):at + 1]}
                if len(side) == 1:
                    p["speaker"] = side.pop()
    out: list[dict] = []
    for p in parts:
        last = out[-1] if out else None
        if last is not None and last["seg"] == p["seg"] and last["speaker"] == p["speaker"]:
            last["end"] = p["end"]
        else:
            out.append({"start": p["start"], "end": p["end"], "speaker": p["speaker"], "seg": p["seg"]})
    order: dict[int, int] = {}
    for s in sorted(out, key=lambda s: s["start"]):
        order.setdefault(s["speaker"], len(order))
    return [{"start": s["start"], "end": s["end"], "speaker": order[s["speaker"]]}
            for s in sorted(out, key=lambda s: (s["start"], s["end"]))]


# ---------------------------------------------------------------- voiceprint

def _vector(values) -> np.ndarray | None:
    vec = np.asarray(values, dtype=np.float32).reshape(-1)
    return vec if vec.size and np.all(np.isfinite(vec)) and np.any(vec) else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else -1.0


def agreeing_group(vecs: list[np.ndarray], size: int = FIRST_CLIPS, agree: float = ENROLL_AGREE) -> tuple | None:
    """Indices of `size` embeddings that all agree pairwise (cosine >= agree), the tightest such group;
    None when no group agrees. Recent clips win a tie."""
    best, best_key = None, None
    for group in itertools.combinations(range(len(vecs)), size):
        low = min(_cosine(vecs[i], vecs[j]) for i, j in itertools.combinations(group, 2))
        if low < agree:
            continue
        key = (low, group)
        if best_key is None or key > best_key:
            best, best_key = group, key
    return best


def speech_only(audio: np.ndarray) -> np.ndarray:
    """Just the speech in a clip (Silero VAD from faster-whisper), so pauses do not dilute the voiceprint."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = _mono(audio)
    if audio.size == 0:
        return audio
    spans = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=300, speech_pad_ms=100), sampling_rate=SAMPLE_RATE
    )
    if not spans:
        return audio[:0]
    return np.concatenate([audio[s["start"]:s["end"]] for s in spans])


class Voiceprint:
    """The owner's voice as a running mean of L2-normalised WeSpeaker embeddings, kept in voiceprint.json
    ({"embedding": [...], "count": n, ...}). Dictation clips feed it; meetings use it to name the owner's cluster."""

    def __init__(self, path: Path, models_dir: Path):
        self.path = Path(path)
        self.models_dir = Path(models_dir)
        self._lock = threading.RLock()
        self._extractor = None
        self._mean: np.ndarray | None = None
        self._count = 0
        self._pending: list[np.ndarray] = []
        self._mtime: int | None = None
        with self._lock:
            self._refresh()

    # -- persistence

    def _refresh(self) -> None:
        """Reload when voiceprint.json changed on disk: the app's dictation instance enrols while a job's instance reads."""
        try:
            mtime = self.path.stat().st_mtime_ns
        except OSError:
            self._mean, self._count, self._mtime, self._pending = None, 0, None, []
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("model", EMBEDDING_MODEL) != EMBEDDING_MODEL:
                raise ValueError(f"made with {data.get('model')}, not {EMBEDDING_MODEL}")
            self._pending = [v for v in (_vector(x) for x in data.get("pending") or []) if v is not None]
            if data.get("embedding") is None:
                self._mean, self._count = None, 0          # still collecting the first agreeing clips
                return
            mean = np.asarray(data["embedding"], dtype=np.float32).reshape(-1)
            count = int(data["count"])
            if mean.size == 0 or count < 1 or not np.all(np.isfinite(mean)) or not np.any(mean):
                raise ValueError("empty or malformed embedding")
            self._mean, self._count = mean, count
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.error("voiceprint %s unreadable, treated as not enrolled: %r", self.path, exc)
            self._mean, self._count, self._pending = None, 0, []

    def _save(self, mean: np.ndarray | None, count: int, pending: list[np.ndarray] | None = None) -> None:
        dim = mean.size if mean is not None else (pending[0].size if pending else 0)
        payload = {
            "model": EMBEDDING_MODEL,
            "dim": int(dim),
            "count": int(count),
            "updated": datetime.now().isoformat(timespec="seconds"),
            "embedding": [float(x) for x in mean] if mean is not None else None,
        }
        if pending:
            payload["pending"] = [[float(x) for x in v] for v in pending]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime_ns

    # -- public

    @property
    def enrolled(self) -> bool:
        with self._lock:
            self._refresh()
            return self._mean is not None

    @property
    def count(self) -> int:
        """Clips averaged into the voiceprint so far."""
        with self._lock:
            self._refresh()
            return self._count

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """L2-normalised WeSpeaker embedding of 16 kHz mono float32 audio. ValueError when there is too little."""
        audio = _mono(audio)
        if audio.size == 0:
            raise ValueError("no audio to embed")
        with self._lock:
            if self._extractor is None:
                import sherpa_onnx

                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(embedding_model(self.models_dir)), num_threads=2, provider="cpu"
                )
                if not config.validate():
                    raise ValueError("sherpa-onnx rejected the speaker-embedding config")
                self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            stream = self._extractor.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            stream.input_finished()
            if not self._extractor.is_ready(stream):
                raise ValueError("audio too short to embed")
            vec = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return _unit(vec)

    def similarity(self, embedding: np.ndarray) -> float:
        """Cosine similarity to the voiceprint; -1.0 (never a match) when nothing is enrolled."""
        with self._lock:
            self._refresh()
            mean = self._mean
        if mean is None:
            return -1.0
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.shape != mean.shape:
            log.warning("voiceprint: embedding has %d dims, voiceprint %d", vec.size, mean.size)
            return -1.0
        denom = float(np.linalg.norm(mean) * np.linalg.norm(vec))
        return float(np.dot(mean, vec) / denom) if denom > 0 else -1.0

    @property
    def pending(self) -> int:
        """Candidate clips held while the first agreeing ones are awaited."""
        with self._lock:
            self._refresh()
            return len(self._pending)

    def enroll(self, audio: np.ndarray) -> bool:
        """Offer one dictation clip to the voiceprint. Needs >= 2 s of speech; returns whether it was taken.

        Anyone can hold Ctrl+Win, so no clip is trusted on its own. Before the voiceprint exists, clips wait
        as candidates until three agree pairwise (cosine >= ENROLL_AGREE); those three become the voiceprint
        and the outliers are dropped. Afterwards a clip joins the running mean only if it is that close to it."""
        speech = speech_only(audio)
        seconds = speech.size / SAMPLE_RATE
        if seconds < MIN_ENROLL_SECONDS:
            log.info("voiceprint: %.1f s of speech, below %.1f s, not enrolled", seconds, MIN_ENROLL_SECONDS)
            return False
        try:
            vec = self.embed(speech)
        except ValueError as exc:
            log.warning("voiceprint: clip not enrolled: %s", exc)
            return False
        with self._lock:
            self._refresh()
            if self._mean is not None and self._mean.shape == vec.shape:
                sim = _cosine(self._mean, vec)
                if sim < ENROLL_AGREE:
                    log.warning("voiceprint: clip rejected, similarity %.2f below %.2f (another voice?)",
                                sim, ENROLL_AGREE)
                    return False
                count = self._count + 1
                mean = (self._mean + (vec - self._mean) / count).astype(np.float32)
                self._save(mean, count)
                self._mean, self._count, self._pending = mean, count, []
                log.info("voiceprint: enrolled %.1f s of speech (%d clips, similarity %.2f)", seconds, count, sim)
                return True
            pending = [v for v in self._pending if v.shape == vec.shape] + [vec]
            pending = pending[-PENDING_MAX:]
            group = agreeing_group(pending)
            if group is None:
                self._save(None, 0, pending)
                self._pending = pending
                log.info("voiceprint: candidate %d held until %d clips agree", len(pending), FIRST_CLIPS)
                return True
            mean = np.mean([pending[i] for i in group], axis=0).astype(np.float32)
            dropped = len(pending) - len(group)
            self._save(mean, len(group))
            self._mean, self._count, self._pending = mean, len(group), []
        log.info("voiceprint: committed from %d agreeing clips (%d outliers dropped)", len(group), dropped)
        return True


# ---------------------------------------------------------------- naming

def _cluster_embedding(audio: np.ndarray, segs: list[dict], voiceprint: Voiceprint) -> np.ndarray | None:
    """Mean embedding of up to 8 of the cluster's longest segments of >= 1.5 s; None when it has none."""
    long = sorted(
        (s for s in segs if float(s["end"]) - float(s["start"]) >= MIN_NAMING_SEGMENT),
        key=lambda s: float(s["end"]) - float(s["start"]),
        reverse=True,
    )[:MAX_NAMING_SEGMENTS]
    vecs = []
    for seg in long:
        clip = audio[int(float(seg["start"]) * SAMPLE_RATE):int(float(seg["end"]) * SAMPLE_RATE)]
        if clip.size == 0:
            continue
        try:
            vecs.append(voiceprint.embed(_middle(clip, MAX_EMBED_SECONDS)))
        except ValueError as exc:
            log.debug("naming: segment %.1f-%.1f skipped: %s", seg["start"], seg["end"], exc)
    return np.mean(vecs, axis=0) if vecs else None


def name_speakers(audio: np.ndarray, diar: list[dict], voiceprint: Voiceprint | None, owner_name: str,
                  owner_match: float = 0.6, pitch: dict[int, float] | None = None) -> dict[int, str]:
    """{cluster: label}. The clusters closest to the voiceprint (owner_clusters) are the owner; the rest are
    Speaker 2, 3, ... in order of first appearance. With no owner match: Speaker 1, 2, ...

    A fragment that is not the owner (a cluster with no segment of MIN_NAMING_SEGMENT, too short to score) shares
    the label of the one longer cluster whose pitch register is within PITCH_SAME of it, unless that one is the
    owner's. `pitch` is pitch_registers' {cluster: Hz}, computed here when None."""
    first: dict[int, float] = {}
    for seg in sorted(diar, key=lambda s: float(s["start"])):
        first.setdefault(int(seg["speaker"]), float(seg["start"]))
    order = sorted(first, key=lambda k: first[k])
    samples = _mono(audio)
    scores: dict[int, float] = {}
    if order and voiceprint is not None and voiceprint.enrolled:
        for spk in order:
            vec = _cluster_embedding(samples, [s for s in diar if int(s["speaker"]) == spk], voiceprint)
            if vec is not None:
                scores[spk] = voiceprint.similarity(vec)
        log.info("naming: voiceprint similarity %s (match at %.2f)",
                 ", ".join(f"cluster {k} {v:.2f}" for k, v in scores.items()) or "none", owner_match)
    if pitch is None:
        pitch = pitch_registers(samples, diar) if order else {}
    owners = owner_clusters(scores, owner_match, pitch)
    hosts = fragment_hosts(diar, owners, pitch)
    names: dict[int, str] = {}
    n = 2 if owners else 1
    for spk in order:
        if spk in owners:
            names[spk] = owner_name
            continue
        host = hosts.get(spk, spk)
        if host not in names:
            names[host] = f"Speaker {n}"
            n += 1
        names[spk] = names[host]
    return names


def fragment_hosts(diar: list[dict], owners: set[int], pitch: dict[int, float]) -> dict[int, int]:
    """{fragment: cluster whose label it shares}. A fragment has no segment of MIN_NAMING_SEGMENT, so it has no
    voiceprint score and too little audio for an embedding to say whose it is; its pitch can. It joins the one
    longer cluster within PITCH_SAME of its register, and never the owner's: a stranger's short reply must not
    carry the owner's name."""
    long = {int(s["speaker"]) for s in diar if float(s["end"]) - float(s["start"]) >= MIN_NAMING_SEGMENT}
    hosts: dict[int, int] = {}
    for spk in sorted({int(s["speaker"]) for s in diar} - long - set(owners)):
        if spk not in pitch:
            continue
        near = [k for k in sorted(long) if k in pitch and abs(np.log2(pitch[k] / pitch[spk])) <= np.log2(PITCH_SAME)]
        if len(near) == 1 and near[0] not in owners:
            hosts[spk] = near[0]
            log.info("naming: fragment cluster %d (%.0f Hz) labelled with cluster %d (%.0f Hz)",
                     spk, pitch[spk], near[0], pitch[near[0]])
    return hosts


def _pitch_apart(pitch: dict[int, float] | None, a: int, b: int) -> bool:
    """Both registers known and at least PITCH_SPLIT apart: two voices, whatever the embeddings say."""
    return bool(pitch) and a in pitch and b in pitch and abs(np.log2(pitch[a] / pitch[b])) >= np.log2(PITCH_SPLIT)


def owner_clusters(scores: dict[int, float], owner_match: float, pitch: dict[int, float] | None = None) -> set[int]:
    """The clusters that are the owner's voice. Every cluster within OWNER_BAND of the best match (and above
    owner_match) is the owner's: a long meeting can split one voice into two clusters. A cluster whose pitch register
    (`pitch`, {cluster: Hz}) is PITCH_SPLIT or more from the best match's is another voice, however close its score.
    A best match below CLEAR_MATCH must also beat every other cluster by OWNER_MARGIN, or a same-accent stranger
    could take the owner's name."""
    if not scores:
        return set()
    best = max(scores, key=lambda k: scores[k])
    top = scores[best]
    if top < owner_match:
        return set()
    band = {k for k, v in scores.items() if v >= owner_match and v >= top - OWNER_BAND}
    apart = {k for k in band if _pitch_apart(pitch, k, best)}
    if apart:
        log.info("naming: cluster %s within the owner band but %.0f Hz against %.0f Hz; not the owner",
                 ", ".join(str(k) for k in sorted(apart)), pitch[min(apart)], pitch[best])
        band -= apart
    rival = max((v for k, v in scores.items() if k not in band), default=None)
    if top < CLEAR_MATCH and rival is not None and top - rival < OWNER_MARGIN:
        log.info("naming: best match %.2f is within %.2f of another cluster (%.2f); owner not matched",
                 top, OWNER_MARGIN, rival)
        return set()
    return band


# ---------------------------------------------------------------- words to lines

def _words(segments: list[dict]) -> list[dict]:
    """Whisper words in order; a segment without word timestamps counts as one word spanning the segment."""
    out: list[dict] = []
    for seg in segments:
        words = [w for w in (seg.get("words") or []) if str(w.get("word", "")).strip()]
        if words:
            out.extend({"start": float(w["start"]), "end": float(w["end"]), "word": " ".join(str(w["word"]).split())}
                       for w in words)
        elif str(seg.get("text", "")).strip():
            out.append({"start": float(seg["start"]), "end": float(seg["end"]), "word": " ".join(str(seg["text"]).split())})
    return out


def assign_words(segments: list[dict], diar: list[dict], names: dict[int, str], gap: float = 3.0) -> list[dict]:
    """[{"speaker", "start", "end", "text"}]. Each word goes to the diarisation segment it overlaps most (the nearest
    segment when none overlaps; on an equal overlap the previous word's speaker); a new line starts on a speaker
    change or a pause longer than `gap` seconds."""
    spans = sorted(diar, key=lambda s: (float(s["start"]), float(s["end"])))
    starts = [float(s["start"]) for s in spans]
    ends = [float(s["end"]) for s in spans]
    # Running max of end (and where it sits) over spans sorted by start: bounds the overlap scan, and gives the
    # nearest span before a word that overlaps nothing.
    run_end: list[float] = []
    run_arg: list[int] = []
    best_end, best_i = float("-inf"), -1
    for i, e in enumerate(ends):
        if e > best_end:
            best_end, best_i = e, i
        run_end.append(best_end)
        run_arg.append(best_i)

    def tie_wins(j: int, best: int, prev: int | None) -> bool:
        """Equal overlap: the previous word's speaker keeps the floor (Whisper writes the dominant voice and
        drops the overlapped "ja" or "mm"); without one, the shorter span wins."""
        sj, sb = int(spans[j]["speaker"]), int(spans[best]["speaker"])
        if prev is not None and (sj == prev) != (sb == prev):
            return sj == prev
        return ends[j] - starts[j] < ends[best] - starts[best]

    def speaker_at(ws: float, we: float, prev: int | None = None) -> int | None:
        if not spans:
            return None
        if we <= ws:
            we = ws + 1e-3                       # a zero-length word counts as a point
        idx = bisect.bisect_left(starts, we)     # spans[:idx] start before the word ends
        best, best_ov = -1, 0.0
        j = idx - 1
        while j >= 0 and run_end[j] > ws:
            ov = min(we, ends[j]) - max(ws, starts[j])
            if ov > best_ov + 1e-9 or (best >= 0 and abs(ov - best_ov) <= 1e-9 and tie_wins(j, best, prev)):
                best, best_ov = j, ov
            j -= 1
        if best < 0:
            before = run_arg[idx - 1] if idx > 0 else -1
            gap_before = ws - ends[before] if before >= 0 else float("inf")
            gap_after = starts[idx] - we if idx < len(spans) else float("inf")
            best = before if gap_before <= gap_after else idx
        return int(spans[best]["speaker"])

    unknown: set[int] = set()
    lines: list[dict] = []
    prev: int | None = None
    for w in _words(segments):
        spk = speaker_at(w["start"], w["end"], prev)
        prev = spk
        if spk is None:
            name = ""
        elif spk in names:
            name = names[spk]
        else:
            name = f"Speaker {spk + 1}"
            if spk not in unknown:
                unknown.add(spk)
                log.warning("assign_words: cluster %d has no name, labelled %s", spk, name)
        if lines and lines[-1]["speaker"] == name and w["start"] - lines[-1]["end"] <= gap:
            lines[-1]["text"] += " " + w["word"]
            lines[-1]["end"] = max(lines[-1]["end"], w["end"])
        else:
            lines.append({"speaker": name, "start": w["start"], "end": w["end"], "text": w["word"]})
    return lines


# ---------------------------------------------------------------- the labelling step, in a child process

def label(audio_path: Path, segments: list[dict], models_dir: Path, *, threshold: float, owner_name: str,
          owner_match: float, voiceprint_path: Path | None) -> dict:
    """Diarise, part voices sherpa merged by pitch (refine), name by voiceprint and assign Whisper's words:
    the speakers.json document."""
    diar = diarize(audio_path, models_dir, threshold=threshold)
    names: dict[int, str] = {}
    lines: list[dict] = []
    if diar:
        vp, voiceprint = None, None
        if voiceprint_path is not None:
            try:
                vp = Voiceprint(voiceprint_path, models_dir)
                voiceprint = vp if vp.enrolled else None
            except Exception:
                log.warning("voiceprint unavailable; speakers left unnamed", exc_info=True)
        audio = read_audio(Path(audio_path))
        pitch = None
        try:
            frames = pitch_frames(audio, diar)
            diar = refine(audio, diar, embed=vp.embed if vp is not None else None, frames=frames)
            pitch = pitch_registers(audio, diar, frames)
        except Exception:
            log.warning("pitch refinement failed; sherpa's clusters kept", exc_info=True)
            pitch = {}
        names = name_speakers(audio, diar, voiceprint, owner_name, owner_match=owner_match, pitch=pitch)
        lines = assign_words(segments, diar, names)
    return {
        "owner": owner_name,
        "owner_matched": owner_name in names.values(),
        "names": {str(k): v for k, v in names.items()},
        "diarization": diar,
        "lines": lines,
    }


def label_job(audio_path: Path, whisper_path: Path, out_path: Path, models_dir: Path, *, threshold: float,
              owner_name: str, owner_match: float, voiceprint_path: Path | None,
              log_path: Path | None = None) -> dict:
    """Run `label` in a child Python process and return what it wrote to out_path.

    sherpa-onnx holds the GIL for the whole diarisation (minutes on a long meeting). In Scribe's own process
    that would starve the keyboard hook (Windows then removes it), the recorder's audio callbacks and the
    overlay; a child process cannot touch any of them. The child has no window (CREATE_NO_WINDOW)."""
    audio_path, out_path = Path(audio_path), Path(out_path)
    try:
        import soundfile as sf

        duration = float(sf.info(str(audio_path)).duration)
    except Exception:
        duration = 0.0
    timeout = CHILD_TIMEOUT_S + 2.0 * duration
    cmd = [sys.executable, "-m", "scribe.diarize", "label", "--audio", str(audio_path), "--whisper", str(whisper_path),
           "--out", str(out_path), "--models", str(models_dir), "--threshold", repr(float(threshold)),
           "--owner", owner_name, "--owner-match", repr(float(owner_match))]
    if voiceprint_path is not None:
        cmd += ["--voiceprint", str(voiceprint_path)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(PACKAGE_ROOT), env.get("PYTHONPATH", "")) if p)
    env["PYTHONIOENCODING"] = "utf-8"
    out_path.unlink(missing_ok=True)
    log_path = Path(log_path) if log_path else out_path.with_name(out_path.stem + ".log")
    began = time.monotonic()
    with open(log_path, "wb") as fh:
        try:
            proc = subprocess.run(cmd, cwd=str(PACKAGE_ROOT), env=env, stdin=subprocess.DEVNULL, stdout=fh,
                                  stderr=subprocess.STDOUT, timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"labelling timed out after {timeout:.0f} s") from None
    if proc.returncode != 0 or not out_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        raise RuntimeError(f"labelling exited {proc.returncode}: {' | '.join(tail)}")
    log.info("labelling child finished in %.1f s", time.monotonic() - began)
    return json.loads(out_path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    """python -m scribe.diarize label --audio A --whisper W --out O --models M ...: the child half of label_job."""
    parser = argparse.ArgumentParser(prog="scribe.diarize")
    sub = parser.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label")
    lab.add_argument("--audio", type=Path, required=True)
    lab.add_argument("--whisper", type=Path, required=True)
    lab.add_argument("--out", type=Path, required=True)
    lab.add_argument("--models", type=Path, required=True)
    lab.add_argument("--threshold", type=float, default=0.5)
    lab.add_argument("--owner", required=True)
    lab.add_argument("--owner-match", type=float, default=0.6)
    lab.add_argument("--voiceprint", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr or open(os.devnull, "w"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    whisper = json.loads(args.whisper.read_text(encoding="utf-8"))
    result = label(args.audio, whisper.get("segments", []), args.models, threshold=args.threshold,
                   owner_name=args.owner, owner_match=args.owner_match, voiceprint_path=args.voiceprint)
    _write_json_atomic(args.out, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
