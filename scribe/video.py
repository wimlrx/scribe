"""Video and media files: probe, audio extraction, slide keyframes and on-screen notes.

ffmpeg/ffprobe do the media work (hidden processes, no window). Keyframes are the frames where the
screen changed (scene cut) plus a heartbeat frame every max_gap seconds, deduped with a numpy dHash
so a re-shown slide is described once. Qwen vision via the local Ollama server, never through a proxy,
describes what each kept frame shows; camera-only frames come back NONE and are dropped.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from PIL import Image

log = logging.getLogger("scribe.video")

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
FRAME_WIDTH = 1280
DHASH_SIZE = 8
DHASH_MAX_DISTANCE = 5
OLLAMA_TIMEOUT = (10, 600)  # (connect, read) seconds; the first frame also pays the model load
NO_PROXY = {"http": None, "https": None, "all": None}   # the server is local: no system or HTTP(S)/ALL_PROXY
                                                        # (a copy per call: requests adds to the dict it gets)

# Describe first, NONE last and narrow: asked "a blank screen or nothing readable" first, qwen3.6 answered NONE
# for a slide holding only a title on a plain background.
DESCRIBE_PROMPT = (
    "This is a frame from a recorded meeting or screen share. "
    "Describe what is on screen for someone reading the meeting transcript:\n"
    "- the slide, document or window title;\n"
    "- the key text, quoted as written;\n"
    "- every number that matters, with its label;\n"
    "- table contents as rows;\n"
    "- the takeaway of any chart.\n"
    "Plain text, concise, no preamble.\n"
    "Answer exactly NONE only when nothing on it can be read: only people, a camera feed or an empty screen. "
    "A slide with only a title is not empty: give the title."
)

_SHOWINFO = re.compile(r"Parsed_showinfo\S*\s.*?\bn:\s*(\d+)\b.*?\bpts_time:\s*(\S+)")
_OUT_TIME = re.compile(r"^out_time_us=(\d+)\s*$", re.M)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


class VideoError(RuntimeError):
    """ffmpeg, ffprobe or the vision model could not do the job."""


def _run(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a media tool hidden: no console window, no stdin, text output."""
    try:
        return subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except FileNotFoundError as exc:
        raise VideoError(f"{args[0]} not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoError(f"{args[0]} timed out after {timeout:.0f} s") from exc


def _tail(text: str, lines: int = 5) -> str:
    return " | ".join(text.strip().splitlines()[-lines:])


def _float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _scan_duration(path: Path) -> float:
    """Duration by remuxing to null: for files whose header carries none (browser webm recordings)."""
    res = _run([FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-nostats", "-i", str(path),
                "-map", "0", "-c", "copy", "-f", "null", "-", "-progress", "pipe:1"], timeout=1800)
    times = [int(m) for m in _OUT_TIME.findall(res.stdout)]
    return max(times) / 1e6 if times else 0.0


def probe(path: Path) -> dict:
    """{"duration", "has_video", "has_audio"}. Cover art (attached pictures) does not count as video."""
    path = Path(path)
    res = _run([FFPROBE, "-v", "error", "-of", "json",
                "-show_entries", "format=duration:stream=codec_type,duration:stream_disposition=attached_pic",
                str(path)], timeout=120)
    if res.returncode != 0:
        raise VideoError(f"ffprobe failed on {path.name}: {_tail(res.stderr)}")
    try:
        info = json.loads(res.stdout or "{}")
    except ValueError as exc:
        raise VideoError(f"ffprobe gave unreadable output for {path.name}") from exc
    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
                    for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = _float(info.get("format", {}).get("duration"))
    if duration is None:
        durations = [d for d in (_float(s.get("duration")) for s in streams) if d is not None]
        duration = max(durations) if durations else None
    if duration is None:
        duration = _scan_duration(path)
        log.info("probe %s: no duration in header, scanned %.1f s", path.name, duration)
    return {"duration": float(duration), "has_video": has_video, "has_audio": has_audio}


def extract_audio(path: Path, out_flac: Path) -> Path:
    """First audio track -> 16 kHz mono 16-bit FLAC, aligned to the file's clock.

    A late-starting or gappy track is padded with silence (aresample async) so transcript times
    line up with keyframe times. Written to a .part file and renamed, so a crash never leaves a
    truncated FLAC that a resumed job would trust.
    """
    path, out_flac = Path(path), Path(out_flac)
    out_flac.parent.mkdir(parents=True, exist_ok=True)
    part = out_flac.with_name(out_flac.stem + ".part.flac")
    res = _run([FFMPEG, "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(path),
                "-map", "0:a:0", "-vn", "-sn", "-dn",
                "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", "16000",
                "-sample_fmt", "s16", "-c:a", "flac", "-f", "flac", str(part)], timeout=3600)
    if res.returncode != 0 or not part.exists() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise VideoError(f"audio extraction failed for {path.name}: {_tail(res.stderr)}")
    part.replace(out_flac)
    log.info("extract_audio %s -> %s", path.name, out_flac.name)
    return out_flac


def dhash(image_path: Path, size: int = DHASH_SIZE) -> np.ndarray:
    """Difference hash: (size x size) bools, True where a cell is brighter than its left neighbour.

    Built on gradients, so it ignores uniform brightness but changes when the layout or text moves.
    Cells are plain area means (BOX): Lanczos ringing turns flat neighbours into near-ties whose
    sign flips with brightness and JPEG noise (measured 7 bits on a re-shown slide vs 0 with BOX).
    """
    with Image.open(image_path) as im:
        small = im.convert("L").resize((size + 1, size), Image.Resampling.BOX)
    a = np.asarray(small, dtype=np.int16)
    return a[:, 1:] > a[:, :-1]


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _dedupe(frames: list[dict], max_gap: float) -> list[dict]:
    """Drop near-duplicates of the last kept frame, unless max_gap has passed since it.

    The exception keeps the heartbeat: a long static screen still gets a frame every max_gap s.
    Dropped files are deleted.
    """
    kept: list[dict] = []
    last_hash = None
    for f in frames:
        h = dhash(f["file"])
        if last_hash is not None:
            dist = hamming(h, last_hash)
            if dist <= DHASH_MAX_DISTANCE and f["t"] - kept[-1]["t"] < max_gap - 0.5:
                log.debug("keyframes: drop near-duplicate t=%.2f (hamming %d vs t=%.2f)",
                          f["t"], dist, kept[-1]["t"])
                Path(f["file"]).unlink(missing_ok=True)
                continue
        kept.append(f)
        last_hash = h
    return kept


def _thin(frames: list[dict], max_frames: int) -> list[dict]:
    """Evenly spaced subset of at most max_frames, first frame always kept; the rest deleted."""
    if len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        idx = {0}
    else:
        idx = {round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)}
    out = []
    for i, f in enumerate(frames):
        if i in idx:
            out.append(f)
        else:
            Path(f["file"]).unlink(missing_ok=True)
    log.warning("keyframes: %d frames thinned to %d", len(frames), len(out))
    return out


def keyframes(path: Path, out_dir: Path, scene: float = 0.3, max_gap: float = 120.0,
              max_frames: int = 200) -> list[dict]:
    """[{"t": seconds, "file": Path}] — scene cuts plus a frame at least every max_gap s.

    One ffmpeg pass: select (first frame | scene > threshold | max_gap since last pick), scale to
    at most 1280 wide, showinfo for the timestamps (stderr pts_time), JPEGs to out_dir. Then
    near-duplicates are dropped with a numpy dHash and the list is capped at max_frames.
    Times are relative to the start of the file, the same clock as extract_audio.
    """
    path, out_dir = Path(path), Path(out_dir)
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    info = probe(path)
    if not info["has_video"]:
        log.info("keyframes %s: no video stream", path.name)
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame_*.jpg"):
        stale.unlink()

    select = f"select='isnan(prev_selected_t)+gt(scene,{scene!r})+gte(t-prev_selected_t,{max_gap!r})'"
    vf = f"{select},scale='min({FRAME_WIDTH},iw)':-2,showinfo"
    res = _run([FFMPEG, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", "-y",
                "-i", str(path), "-map", "0:V:0", "-an", "-sn", "-dn", "-vf", vf,
                "-fps_mode", "vfr", "-q:v", "2", "-start_number", "0", "-f", "image2",
                str(out_dir / "frame_%05d.jpg")],
               timeout=max(900.0, 3 * info["duration"]))
    if res.returncode != 0:
        raise VideoError(f"keyframe extraction failed for {path.name}: {_tail(res.stderr)}")

    candidates: list[dict] = []
    for m in _SHOWINFO.finditer(res.stderr):
        n, t = int(m.group(1)), _float(m.group(2))
        file = out_dir / f"frame_{n:05d}.jpg"
        if t is None or not file.exists():
            log.warning("keyframes %s: frame %d has no usable time or file, skipped", path.name, n)
            continue
        candidates.append({"t": round(max(t, 0.0), 3), "file": file})
    candidates.sort(key=lambda f: f["t"])

    unique = _dedupe(candidates, max_gap)
    kept = _thin(unique, max_frames)
    log.info("keyframes %s: %d candidates, %d near-duplicates dropped, %d kept",
             path.name, len(candidates), len(candidates) - len(unique), len(kept))
    return kept


def _is_none(text: str) -> bool:
    """True for the model's 'nothing to read here' answer (NONE, NONE., NONE - camera only, empty)."""
    t = text.strip().lstrip("*_`\"' ")
    return not t or bool(re.match(r"NONE\b", t)) or t.rstrip(" .!").lower() == "none"


def _chat(url: str, body: dict) -> str:
    r = requests.post(url, json=body, timeout=OLLAMA_TIMEOUT, proxies=dict(NO_PROXY))
    if r.status_code == 404:
        raise VideoError(f"ollama: model {body.get('model')!r} not found")
    r.raise_for_status()
    content = (r.json().get("message") or {}).get("content") or ""
    return _THINK.sub("", content).strip()


def _unload(url: str, model: str) -> None:
    """Free the GPU for the next job: an empty chat with keep_alive 0 unloads the model."""
    try:
        requests.post(url, json={"model": model, "messages": [], "keep_alive": 0}, timeout=(10, 60),
                      proxies=dict(NO_PROXY))
        log.info("ollama: unloaded %s", model)
    except requests.RequestException as exc:
        log.warning("ollama: unload of %s failed: %r", model, exc)


def describe_frames(frames: list[dict], ollama_url: str, model: str,
                    progress: Callable[[float], None] | None = None) -> list[dict]:
    """[{"t", "text"}] — what each keyframe shows, from the local vision model.

    Frames answered NONE (camera feed, people only) are dropped. A frame whose request fails twice
    is skipped; an unreachable server, a missing model, or every frame failing raises VideoError.
    The model is unloaded at the end either way.
    """
    if not frames:
        return []
    url = ollama_url.rstrip("/") + "/api/chat"
    out: list[dict] = []
    failed = 0
    try:
        for i, frame in enumerate(frames, 1):
            image = base64.b64encode(Path(frame["file"]).read_bytes()).decode("ascii")
            body = {
                "model": model,
                "messages": [{"role": "user", "content": DESCRIBE_PROMPT, "images": [image]}],
                "stream": False,
                "think": False,
                "options": {"num_ctx": 8192, "temperature": 0},
            }
            text = None
            for attempt in (1, 2):
                try:
                    text = _chat(url, body)
                    break
                except requests.ConnectionError as exc:
                    raise VideoError(f"ollama unreachable at {ollama_url}: {exc!r}") from exc
                except (requests.RequestException, ValueError) as exc:
                    log.warning("ollama: frame t=%.1f attempt %d failed: %r", frame["t"], attempt, exc)
            if text is None:
                failed += 1
            elif _is_none(text):
                log.debug("describe: t=%.1f NONE, dropped", frame["t"])
            else:
                out.append({"t": frame["t"], "text": text})
            if progress:
                progress(i / len(frames))
        if failed == len(frames):
            raise VideoError(f"ollama: all {failed} frames failed")
    finally:
        _unload(url, model)
    log.info("describe: %d frames, %d described, %d NONE, %d failed",
             len(frames), len(out), len(frames) - len(out) - failed, failed)
    return out
