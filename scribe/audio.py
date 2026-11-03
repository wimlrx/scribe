"""Audio capture and audio files: the dictation mic, the two-track meeting recorder, the mix, the shelved m4a.

Everything runs locally. The meeting recorder writes mic.flac and system.flac as it goes, so a 2-hour call
never sits in RAM and a crash loses at most a couple of seconds. Both tracks are kept on the wall clock:
WASAPI loopback delivers nothing while nothing plays, and devices drop out, so gaps are filled with silence
by time rather than trusted to the stream.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pyaudiowpatch as pyaudio
import soundfile as sf
import soxr

log = logging.getLogger("scribe.audio")

SR = 16000                 # every track and the mix: 16 kHz mono
FLUSH_S = 2.0              # FLAC flush + recording.json heartbeat (spec: at least every 5 s)
TICK_S = 0.1               # recorder writer loop period
CHUNK_S = 0.05             # PortAudio buffer length
LATE_S = 0.04              # a track this far behind the wall clock gets zeros (two drifting clocks stay < 100 ms apart)
EARLY_S = 0.04             # a track this far ahead has its next audio trimmed
WINDOW_S = 2.0             # alignment trusts the least-delayed chunk of this window: callback jitter only delays
BREAK_S = 0.5              # a pause between chunks longer than this is a gap, not jitter
HOLD_S = 1.0               # a track silent this long is padded up to the wall clock minus HOLD_S
STALL_S = 3.0              # a stream expected to run continuously that is silent this long is reopened
DEVICE_POLL_S = 2.0        # default-device and stream-health check interval
REOPEN_GAP_S = 5.0         # minimum time between full reopens (default device changed); first track back-off step
BACKOFF_MAX_S = 120.0      # a track that keeps stalling is reattached at most this rarely
MISSING_RETRY_S = 10.0     # a track with no device retries this often ...
MISSING_REFRESH_S = 30.0   # ... re-enumerating devices at most this often (doubling up to MISSING_REFRESH_MAX_S)
MISSING_REFRESH_MAX_S = 600.0
QUEUE_MAX_S = 30.0         # audio buffered per track ahead of the writer; beyond it chunks are dropped (then padded)
MAX_EVENTS = 200
PEAK_DBFS = -1.0
MIN_FREE_START = 1 << 30   # a recording does not start with less free disk than this
LOW_DISK_BYTES = 15 * 60 * 2 * SR * 2   # about 15 minutes of both tracks, uncompressed: warn below it
VOICE_RMS = 10 ** (-50 / 20)            # a chunk louder than -50 dBFS counts as someone talking
SILENCE_STOP_S = 20 * 60.0              # both tracks this long without voice: the notetaker was forgotten
MAX_RECORDING_S = 6 * 3600.0            # hard cap on one recording
RESUME_GAP_S = 30.0                     # a gap this long between writer ticks means the machine slept
NO_WINDOW = subprocess.CREATE_NO_WINDOW
_ZERO_BLOCK = SR * 10      # zeros are written in blocks of this many frames


class AudioError(RuntimeError):
    """A device could not be opened, or an audio file could not be produced."""


# --------------------------------------------------------------------------------------------------
# Wall-clock alignment: pure functions, no I/O. The recorder feeds them chunk arrival times.
# --------------------------------------------------------------------------------------------------

def chunk_offset(arrival_s: float, written: int, frames: int, rate: int = SR) -> int:
    """Frames by which the wall clock leads the track once a chunk ending at arrival_s is written whole."""
    return int(round(arrival_s * rate)) - (written + frames)


def correction(offsets, rate: int = SR, late_s: float = LATE_S, early_s: float = EARLY_S) -> int:
    """Signed fix from recent offsets: > 0 zeros to insert, < 0 frames to trim, 0 inside tolerance.
    Uses the minimum because a late callback only ever makes a chunk look later than it was."""
    m = min(offsets)
    if m > late_s * rate or m < -early_s * rate:
        return m
    return 0


def idle_padding(written: int, now_s: float, last_arrival_s: float | None, rate: int = SR,
                 hold_s: float = HOLD_S) -> int:
    """Zeros owed by a track that has delivered nothing for hold_s (loopback in silence, a dead device).
    Keeps the file growing so an interrupted recording still has aligned tracks."""
    if last_arrival_s is not None and now_s - last_arrival_s < hold_s:
        return 0
    return max(0, int(round((now_s - hold_s) * rate)) - written)


def frames_owed(written: int, end_s: float, rate: int = SR) -> int:
    """Frames still needed (> 0) or written past (< 0) for the track to end exactly at end_s."""
    return int(round(end_s * rate)) - written


class TrackClock:
    """Where one track stands against the wall clock, and what to pad or trim. Bookkeeping only."""

    def __init__(self, rate: int = SR, window_s: float = WINDOW_S, late_s: float = LATE_S,
                 early_s: float = EARLY_S, hold_s: float = HOLD_S, break_s: float = BREAK_S):
        self.rate, self.window_s, self.late_s, self.early_s = rate, window_s, late_s, early_s
        self.hold_s, self.break_s = hold_s, break_s
        self.written = 0
        self.last_arrival: float | None = None
        self._recent: deque[tuple[float, int]] = deque()

    def place(self, arrival_s: float, frames: int, end_s: float | None = None) -> tuple[int, int, int]:
        """(zeros to write first, frames to skip at the chunk head, frames to keep) for a chunk of
        `frames` that arrived at arrival_s. With end_s, nothing is placed past end_s."""
        if self.last_arrival is None or arrival_s - self.last_arrival > self.break_s:
            self._recent.clear()  # the stream paused: older chunks say nothing about this one
        self.last_arrival = arrival_s
        self._recent.append((arrival_s, chunk_offset(arrival_s, self.written, frames, self.rate)))
        while self._recent[0][0] < arrival_s - self.window_s:
            self._recent.popleft()
        fix = correction([o for _, o in self._recent], self.rate, self.late_s, self.early_s)
        pad = max(fix, 0)
        skip = min(max(-fix, 0), frames)
        if pad or skip:  # the fix moves the whole trajectory; recent offsets move with it
            self._recent = deque((a, o - pad + skip) for a, o in self._recent)
        keep = frames - skip
        if end_s is not None:
            room = max(0, frames_owed(self.written, end_s, self.rate))
            pad = min(pad, room)
            keep = min(keep, room - pad)
        self.written += pad + keep
        return pad, skip, keep

    def idle(self, now_s: float) -> int:
        """Zeros to write now for a track that has gone quiet."""
        pad = idle_padding(self.written, now_s, self.last_arrival, self.rate, self.hold_s)
        if pad:
            self.written += pad
            self._recent.clear()
        return pad

    def finish(self, end_s: float) -> int:
        """Zeros to append so the track ends at end_s (0 if it is already there or past it)."""
        pad = max(0, frames_owed(self.written, end_s, self.rate))
        self.written += pad
        return pad


# --------------------------------------------------------------------------------------------------
# PortAudio: one process-wide instance. PortAudio enumerates devices only at init and is refcounted,
# so a new headset is invisible until every instance is terminated and re-created.
# --------------------------------------------------------------------------------------------------

_pa_lock = threading.RLock()       # serialises open/close/start/stop/terminate; never taken in callbacks
_pa: pyaudio.PyAudio | None = None
_pa_gen = 0
_captures_running = 0              # dictation captures in progress: a refresh would cut them off


def _portaudio() -> tuple[pyaudio.PyAudio, int]:
    """The shared PortAudio instance and its generation."""
    global _pa
    with _pa_lock:
        if _pa is None:
            _pa = pyaudio.PyAudio()
        return _pa, _pa_gen


def _refresh_portaudio() -> None:
    """Re-enumerate devices. Closes every stream in the process; owners see the generation change and reopen.
    The old instance is dropped and the generation bumped before the re-init, so a failed Pa_Initialize
    leaves nothing dead behind: the next _portaudio() call simply tries again."""
    global _pa, _pa_gen
    with _pa_lock:
        old, _pa = _pa, None
        _pa_gen += 1
        if old is not None:
            try:
                old.terminate()
            except Exception as exc:  # a half-dead device can make terminate complain; the re-init still helps
                log.warning("PortAudio terminate: %r", exc)
        try:
            _pa = pyaudio.PyAudio()
        except Exception as exc:
            log.error("PortAudio re-init failed (retried on next use): %r", exc)


def _close_stream(stream) -> None:
    """Stop and close a stream that may already be dead (closed by a refresh, device unplugged)."""
    if stream is None:
        return
    for step in (stream.stop_stream, stream.close):
        try:
            step()
        except Exception:
            pass


# --------------------------------------------------------------------------------------------------
# Default endpoints via IMMDeviceEnumerator. PortAudio has no device-change notification, so the
# recorder polls these ids to notice a headset being plugged in or the default being switched.
# --------------------------------------------------------------------------------------------------

class _GUID(ctypes.Structure):
    _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD), ("d4", ctypes.c_ubyte * 8)]

    @classmethod
    def of(cls, text: str) -> "_GUID":
        u = uuid.UUID(text)
        g = cls()
        g.d1, g.d2, g.d3 = u.fields[0], u.fields[1], u.fields[2]
        g.d4[:] = list(u.bytes[8:])
        return g


_ole32 = ctypes.WinDLL("ole32")
_ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
_ole32.CoInitializeEx.restype = ctypes.c_long
_ole32.CoUninitialize.argtypes = []
_ole32.CoUninitialize.restype = None
_ole32.CoCreateInstance.argtypes = [ctypes.POINTER(_GUID), ctypes.c_void_p, wintypes.DWORD,
                                    ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
_ole32.CoCreateInstance.restype = ctypes.c_long
_ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
_ole32.CoTaskMemFree.restype = None

_CLSID_ENUMERATOR = _GUID.of("BCDE0395-E52F-467C-8E3D-C4579291692E")
_IID_ENUMERATOR = _GUID.of("A95664D2-9614-4F35-A746-DE8DB63617E6")
_COINIT_MULTITHREADED = 0x0
_CLSCTX_INPROC_SERVER = 0x1
_RPC_E_CHANGED_MODE = 0x80010106
_E_RENDER, _E_CAPTURE, _E_CONSOLE = 0, 1, 0
# vtable slots: IUnknown 0-2; IMMDeviceEnumerator.GetDefaultAudioEndpoint 4; IMMDevice.GetId 5
_GET_DEFAULT = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_void_p))
_GET_ID = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
_RELEASE = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)


def _method(obj: ctypes.c_void_p, slot: int, proto):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return proto(vtable[slot])


class _Endpoints:
    """Default render and capture endpoint ids (eConsole). One per thread: COM objects are per apartment."""

    def __init__(self) -> None:
        self._enum = ctypes.c_void_p()
        self._uninit = False
        hr = _ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
        if hr in (0, 1):
            self._uninit = True
        elif hr & 0xFFFFFFFF != _RPC_E_CHANGED_MODE:  # already initialised another way is fine
            raise OSError(f"CoInitializeEx 0x{hr & 0xFFFFFFFF:08X}")
        hr = _ole32.CoCreateInstance(ctypes.byref(_CLSID_ENUMERATOR), None, _CLSCTX_INPROC_SERVER,
                                     ctypes.byref(_IID_ENUMERATOR), ctypes.byref(self._enum))
        if hr != 0 or not self._enum.value:
            self.close()
            raise OSError(f"MMDeviceEnumerator 0x{hr & 0xFFFFFFFF:08X}")

    def ids(self) -> tuple[str | None, str | None]:
        return self._default_id(_E_RENDER), self._default_id(_E_CAPTURE)

    def _default_id(self, flow: int) -> str | None:
        dev = ctypes.c_void_p()
        hr = _method(self._enum, 4, _GET_DEFAULT)(self._enum, flow, _E_CONSOLE, ctypes.byref(dev))
        if hr != 0 or not dev.value:
            return None  # E_NOTFOUND: no device of that kind
        try:
            sid = ctypes.c_void_p()
            if _method(dev, 5, _GET_ID)(dev, ctypes.byref(sid)) != 0 or not sid.value:
                return None
            try:
                return ctypes.wstring_at(sid.value)
            finally:
                _ole32.CoTaskMemFree(sid)
        finally:
            _method(dev, 2, _RELEASE)(dev)

    def close(self) -> None:
        if self._enum.value:
            _method(self._enum, 2, _RELEASE)(self._enum)
            self._enum = ctypes.c_void_p()
        if self._uninit:
            _ole32.CoUninitialize()
            self._uninit = False


_UNKNOWN = object()


def _capture_endpoint():
    """The default capture endpoint id, or _UNKNOWN when COM is unavailable. Initialises and releases COM
    within the call (~2 ms) so no apartment is left on the caller's thread (it may be the Tk thread)."""
    try:
        ep = _Endpoints()
    except OSError as exc:
        log.debug("endpoint query unavailable: %r", exc)
        return _UNKNOWN
    try:
        return ep.ids()[1]
    finally:
        ep.close()


# --------------------------------------------------------------------------------------------------
# Dictation microphone
# --------------------------------------------------------------------------------------------------

def _mapper_input(pa: pyaudio.PyAudio) -> dict:
    """The MME sound mapper: whatever the default microphone is when the stream opens, converted by
    Windows to the rate asked for. Falls back to the host default input."""
    try:
        mme = pa.get_host_api_info_by_type(pyaudio.paMME)
        info = pa.get_device_info_by_host_api_device_index(mme["index"], 0)
        if info["maxInputChannels"] > 0:
            return info
    except Exception:
        pass
    return pa.get_default_input_device_info()


class MicCapture:
    """Dictation mic: the default input at 16 kHz mono int16. The stream stays open but stopped between
    dictations, because opening costs ~300 ms and that would clip the first word; start() costs ~1 ms."""

    def __init__(self, max_seconds: float = 600.0):
        self.max_seconds = max_seconds
        self._stream = None
        self._gen = -1
        self._endpoint = None
        self._rate = SR
        self._chunks: list[bytes] = []
        self._frames = 0
        self._running = False
        self._ms: deque[float] = deque(maxlen=2)  # mean squares of the last two 50 ms chunks

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            if in_data and self._frames < self.max_seconds * self._rate:
                self._chunks.append(in_data)
                self._frames += frame_count
            if in_data:
                x = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                self._ms.append(float(np.mean(x * x)) if x.size else 0.0)
        except Exception:  # never let an exception abort the stream
            pass
        return (None, pyaudio.paContinue)

    def _usable(self, endpoint) -> bool:
        if self._stream is None or self._gen != _pa_gen:
            return False
        return endpoint is _UNKNOWN or endpoint == self._endpoint

    def _open(self, endpoint) -> None:
        _close_stream(self._stream)
        self._stream = None
        pa, gen = _portaudio()
        try:
            dev = _mapper_input(pa)
        except Exception as exc:
            raise AudioError(f"no microphone: {exc!r}") from exc
        last: Exception | None = None
        for rate in dict.fromkeys((SR, int(dev["defaultSampleRate"]))):
            try:
                self._stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                                       input_device_index=dev["index"], frames_per_buffer=int(rate * CHUNK_S),
                                       stream_callback=self._callback, start=False)
                self._rate, self._gen, self._endpoint = rate, gen, endpoint
                log.info("dictation mic armed: %s @ %d Hz", dev["name"], rate)
                return
            except Exception as exc:
                last = exc
        raise AudioError(f"microphone would not open: {last!r}")

    def start(self) -> None:
        """Begin capturing. Raises AudioError when there is no usable microphone."""
        endpoint = _capture_endpoint()
        with _pa_lock:
            if self._running:
                return
            self._chunks, self._frames = [], 0
            self._ms.clear()
            if not self._usable(endpoint):
                self._open(endpoint)
            try:
                self._stream.start_stream()
            except Exception:  # stale handle (device gone, PortAudio refreshed): reopen once
                self._open(endpoint)
                try:
                    self._stream.start_stream()
                except Exception as exc:
                    raise AudioError(f"microphone would not start: {exc!r}") from exc
            self._running = True
            _count_capture(+1)

    def stop(self) -> np.ndarray:
        """Stop capturing; the audio as float32, 16 kHz mono. Empty when nothing was captured."""
        with _pa_lock:
            if not self._running:
                return np.zeros(0, dtype=np.float32)
            self._running = False
            _count_capture(-1)
            try:
                self._stream.stop_stream()
            except Exception:
                _close_stream(self._stream)
                self._stream = None
        data, self._chunks, self._frames = b"".join(self._chunks), [], 0
        self._ms.clear()
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if self._rate != SR and audio.size:
            audio = soxr.resample(audio, self._rate, SR).astype(np.float32)
        return audio

    def level(self) -> float:
        """RMS 0..1 of the last ~100 ms, for the UI."""
        ms = list(self._ms)
        return min(1.0, float(np.sqrt(sum(ms) / len(ms)))) if (self._running and ms) else 0.0

    def close(self) -> None:
        """Release the device (app quit)."""
        with _pa_lock:
            if self._running:
                _count_capture(-1)
            self._running = False
            _close_stream(self._stream)
            self._stream = None


def _count_capture(delta: int) -> None:
    global _captures_running
    with _pa_lock:
        _captures_running = max(0, _captures_running + delta)


# --------------------------------------------------------------------------------------------------
# Meeting recorder
# --------------------------------------------------------------------------------------------------

def _to_int16(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def _silence_callback(channels: int):
    """Output callback that plays zeros: keeps the audio engine rendering so loopback keeps delivering."""
    def callback(in_data, frame_count, time_info, status):
        return (bytes(frame_count * channels * 2), pyaudio.paContinue)
    return callback


class _Source:
    """One open input stream feeding a track. Its callback only timestamps and enqueues."""

    def __init__(self, track: "_Track", rate: int, channels: int, t0: float):
        self.track, self.rate, self.channels, self.t0 = track, rate, channels, t0
        self.resampler = soxr.ResampleStream(rate, SR, 1, dtype="float32") if rate != SR else None
        self.stream = None

    def callback(self, in_data, frame_count, time_info, status):
        try:
            arrival = time.monotonic() - self.t0
            tr = self.track
            tr.last_callback = arrival
            if status:
                tr.overflows += 1
            if in_data:
                try:
                    tr.queue.put_nowait((arrival, in_data, self))
                except queue.Full:
                    tr.dropped += frame_count
                x = np.frombuffer(in_data, dtype=np.float32)
                tr.level = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        except Exception:  # never let an exception abort the stream
            pass
        return (None, pyaudio.paContinue)

    def active(self) -> bool:
        try:
            return bool(self.stream is not None and self.stream.is_active())
        except Exception:
            return False


class _Track:
    """One FLAC file on the wall clock, plus the stream (if any) feeding it."""

    def __init__(self, name: str, path: Path):
        self.name, self.path = name, path
        self.file = sf.SoundFile(str(path), "w", SR, 1, format="FLAC", subtype="PCM_16")
        self.clock = TrackClock()
        self.queue: queue.Queue = queue.Queue(maxsize=int(QUEUE_MAX_S / CHUNK_S))
        self.source: _Source | None = None
        self.continuous = False      # expected to deliver even in silence (mic; loopback with keep-alive)
        self.opened_at = 0.0
        self.last_callback: float | None = None
        self.level = 0.0
        self.overflows = 0
        self.dropped = 0
        self.reported = (0, 0)       # (overflows, dropped) already logged as events
        self.reported_at = -1e9
        self.backoff = REOPEN_GAP_S  # wait before the next reattach of a stalled stream; doubles while it repeats
        self.next_retry = 0.0
        self.last_voice = 0.0        # recording time of the last chunk above VOICE_RMS
        self.fault: str | None = None  # the file can no longer be written (disk full): set once, never cleared

    def write_zeros(self, n: int) -> None:
        while n > 0:
            k = min(n, _ZERO_BLOCK)
            self.file.write(np.zeros(k, dtype=np.int16))
            n -= k


class MeetingRecorder:
    """Records the default microphone and the WASAPI loopback of the default output into
    <folder>/mic.flac and <folder>/system.flac (16 kHz mono), aligned by wall clock, for any length."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._t0 = 0.0
        self._started = ""
        self._end_s: float | None = None
        self._seconds = 0.0
        self._tracks: list[_Track] = []
        self._keepalive = None
        self._events: list[dict] = []
        self._events_lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: dict | None = None
        self._endpoints: _Endpoints | None = None
        self._ids = None
        self._last_flush = self._last_check = self._last_missing = self._last_tick = 0.0
        self._last_reopen = -REOPEN_GAP_S  # the first reopen is never held back
        self._last_refresh = 0.0
        self._missing_refresh_s = MISSING_REFRESH_S
        self._alerts: set[str] = set()
        self._alerts_lock = threading.Lock()

    # ---- public ----

    def start(self) -> None:
        """Open both devices and begin writing. Raises AudioError if neither track can be opened."""
        if self._thread is not None:
            raise AudioError("recorder already started")
        self.folder.mkdir(parents=True, exist_ok=True)
        free = _free_bytes(self.folder)
        if free is not None and free < MIN_FREE_START:
            raise AudioError(f"disk nearly full: {free / 1e6:.0f} MB free")
        self.mic = _Track("mic", self.folder / "mic.flac")
        self.system = _Track("system", self.folder / "system.flac")
        self._tracks = [self.mic, self.system]
        self._t0 = time.monotonic()
        self._started = datetime.now().isoformat(timespec="seconds")
        try:
            ids = self._probe_ids()  # before the refresh: a default that changes meanwhile shows on the next poll
            with _pa_lock:
                # The shared PortAudio instance may date from the first dictation, hours ago: a headset
                # plugged in since is not in its list. Re-enumerate, unless a dictation is capturing now.
                if _captures_running:
                    log.info("dictation in progress; recorder opens on the current device list")
                else:
                    _refresh_portaudio()
                self._last_refresh = 0.0
                self._open_streams()
            self._ids = ids
            if not any(tr.source for tr in self._tracks):
                raise AudioError("no microphone and no system audio device")
        except BaseException:
            with _pa_lock:
                self._close_streams()
            for tr in self._tracks:
                tr.file.close()
                tr.path.unlink(missing_ok=True)
            raise
        self._heartbeat("recording", 0.0)
        self._thread = threading.Thread(target=self._run, name="scribe-recorder", daemon=True)
        self._thread.start()
        log.info("recording to %s", self.folder)

    def stop(self) -> dict:
        """Stop, pad both tracks to the same end, close the files. {"mic", "system", "seconds", "events"}."""
        if self._result is not None:
            return self._result
        if self._thread is None:
            raise AudioError("recorder not started")
        self._end_s = self._now()
        self._stop_evt.set()
        self._thread.join(timeout=120)
        if self._thread.is_alive():
            log.error("recorder thread did not finish")
        if self._result is None:  # the writer died; report what is on disk
            self._seconds = self._end_s
            self._result = {"mic": self.mic.path, "system": self.system.path, "seconds": self._end_s,
                            "events": list(self._events)}
        return self._result

    def elapsed(self) -> float:
        if self._thread is None:
            return 0.0
        return self._seconds if self._result is not None else self._now()

    def alerts(self) -> set[str]:
        """What the app must act on: "fault" (a track can no longer be written: stop and tell the user),
        "disk_low" (tell), "silence" or "cap" (stop: a forgotten notetaker). Sticky once raised."""
        with self._alerts_lock:
            return set(self._alerts)

    def _alert(self, name: str, detail: str = "") -> None:
        with self._alerts_lock:
            if name in self._alerts:
                return
            self._alerts.add(name)
        self._event("all", "auto_stop" if name in ("silence", "cap") else name, detail)

    def levels(self) -> tuple[float, float]:
        """(mic, system) RMS 0..1 of the last chunk; 0 when a track has gone quiet."""
        if self._thread is None or self._result is not None:
            return 0.0, 0.0
        now = self._now()
        out = []
        for tr in self._tracks:
            fresh = tr.last_callback is not None and now - tr.last_callback < 0.3
            out.append(min(1.0, tr.level) if fresh else 0.0)
        return out[0], out[1]

    # ---- internals ----

    def _now(self) -> float:
        return time.monotonic() - self._t0

    def _event(self, track: str, kind: str, detail: str = "") -> None:
        ev = {"t": round(self._now(), 2), "track": track, "kind": kind, "detail": detail}
        log.info("recorder %s %s %s", track, kind, detail)
        with self._events_lock:
            if len(self._events) < MAX_EVENTS:
                self._events.append(ev)
            elif self._events[-1].get("kind") != "events_truncated":
                self._events.append({**ev, "track": "all", "kind": "events_truncated", "detail": ""})

    def _probe_ids(self):
        try:
            ep = _Endpoints()
        except OSError as exc:
            log.warning("default-device watch unavailable: %r", exc)
            return None
        try:
            return ep.ids()
        finally:
            ep.close()

    def _open_streams(self) -> None:
        """Open keep-alive, loopback and mic on the current defaults. Caller holds _pa_lock."""
        try:
            pa, _ = _portaudio()
        except Exception as exc:  # Pa_Initialize failed: both tracks stay missing and are retried
            self._event("all", "open_failed", f"PortAudio: {exc!r}")
            return
        if self.system.source is None:
            try:
                out = pa.get_default_wasapi_device(d_out=True)
                if self._keepalive is None:
                    try:
                        ch = max(1, int(out["maxOutputChannels"]))  # shared mode wants the mix's own layout
                        rate = int(out["defaultSampleRate"])
                        self._keepalive = pa.open(format=pyaudio.paInt16, channels=ch, rate=rate, output=True,
                                                  output_device_index=out["index"],
                                                  frames_per_buffer=int(rate * CHUNK_S),
                                                  stream_callback=_silence_callback(ch))
                    except Exception as exc:
                        self._event("system", "keepalive_failed", repr(exc))
                self._attach(self.system, pa, pa.get_wasapi_loopback_analogue_by_dict(out),
                             continuous=self._keepalive is not None)
            except Exception as exc:
                self._event("system", "open_failed", repr(exc))
        if self.mic.source is None:
            try:
                try:
                    dev = pa.get_default_wasapi_device(d_in=True)
                except Exception:
                    dev = pa.get_default_input_device_info()
                self._attach(self.mic, pa, dev, continuous=True)
            except Exception as exc:
                self._event("mic", "open_failed", repr(exc))

    def _attach(self, tr: _Track, pa: pyaudio.PyAudio, dev: dict, continuous: bool) -> None:
        rate = int(dev["defaultSampleRate"])
        ch = max(1, int(dev["maxInputChannels"]))  # every channel, averaged to mono in the writer
        src = _Source(tr, rate, ch, self._t0)
        src.stream = pa.open(format=pyaudio.paFloat32, channels=ch, rate=rate, input=True,
                             input_device_index=dev["index"], frames_per_buffer=int(rate * CHUNK_S),
                             stream_callback=src.callback)
        tr.source, tr.continuous = src, continuous
        tr.opened_at, tr.last_callback = self._now(), None
        self._event(tr.name, "open", f"{dev['name']} @ {rate} Hz x{ch}")

    def _close_streams(self) -> None:
        """Caller holds _pa_lock."""
        for tr in self._tracks:
            if tr.source is not None:
                _close_stream(tr.source.stream)
                tr.source = None
        _close_stream(self._keepalive)
        self._keepalive = None

    def _reopen(self, reason: str, now: float) -> None:
        """Close every stream, re-enumerate devices and open the current defaults. The ids are read before
        the refresh: a second default change during the ~1 s reopen then differs on the next poll."""
        self._last_reopen = self._last_refresh = now
        self._event("all", "reopen", reason)
        ids = self._ids
        if self._endpoints is not None:
            try:
                ids = self._endpoints.ids()
            except OSError as exc:
                log.warning("endpoint query failed: %r", exc)
        with _pa_lock:
            self._close_streams()
            _refresh_portaudio()
            self._open_streams()
        self._ids = ids

    def _reattach(self, tr: _Track, reason: str, now: float) -> None:
        """One stalled or stopped track: close and reopen only its own stream (and the keep-alive for the
        system track), without a PortAudio refresh, so the healthy track keeps recording. While the same
        fault repeats the retries back off: 5, 10, 20 ... 120 s, one event per step."""
        step = tr.backoff
        tr.next_retry = now + step
        tr.backoff = min(step * 2, BACKOFF_MAX_S)
        self._event(tr.name, "reattach", f"{reason}; next try in {step:.0f} s")
        with _pa_lock:
            if tr.source is not None:
                _close_stream(tr.source.stream)
                tr.source = None
            if tr is self.system:
                _close_stream(self._keepalive)
                self._keepalive = None
            self._open_streams()

    def _drain(self, tr: _Track, end_s: float | None = None) -> None:
        while True:
            try:
                arrival, data, src = tr.queue.get_nowait()
            except queue.Empty:
                return
            x = np.frombuffer(data, dtype=np.float32)
            if src.channels > 1:
                x = x[: x.size - x.size % src.channels].reshape(-1, src.channels).mean(axis=1)
            y = src.resampler.resample_chunk(x) if src.resampler is not None else x
            if x.size and float(np.sqrt(np.mean(x * x))) > VOICE_RMS:
                tr.last_voice = arrival
            if tr.fault is not None:
                continue                  # the file is dead: keep the queue empty, write nothing
            pad, skip, keep = tr.clock.place(arrival, int(y.size), end_s)
            tr.write_zeros(pad)
            if keep > 0:
                tr.file.write(_to_int16(y[skip:skip + keep]))

    def _tick(self) -> None:
        now = self._now()
        gap = now - self._last_tick
        if self._last_tick and gap > RESUME_GAP_S:
            self._event("all", "resume", f"{gap:.0f} s without a tick (sleep or a stall)")
        self._last_tick = now
        for tr in self._tracks:
            # Each track on its own: a write error on one (disk full) never stops the other being drained.
            try:
                self._drain(tr)
                if tr.fault is None:
                    tr.write_zeros(tr.clock.idle(now))
            except Exception as exc:
                self._track_fault(tr, exc)
        if now - self._last_flush >= FLUSH_S:
            self._flush(now)
        if now - self._last_check >= DEVICE_POLL_S:
            self._last_check = now
            self._check_devices(now)
        quiet = now - max(tr.last_voice for tr in self._tracks)
        if quiet >= SILENCE_STOP_S:
            self._alert("silence", f"no voice on either track for {quiet / 60:.0f} min")
        if now >= MAX_RECORDING_S:
            self._alert("cap", f"{MAX_RECORDING_S / 3600:.0f} h cap reached")

    def _track_fault(self, tr: _Track, exc: BaseException) -> None:
        """A track's file cannot be written (libFLAC's error state is permanent): close it, log once, and
        raise the "fault" alert so the app stops the recording cleanly and tells the user."""
        if tr.fault is not None:
            return
        tr.fault = repr(exc)
        log.error("recorder %s: write failed, track closed: %r", tr.name, exc)
        free = _free_bytes(self.folder)
        detail = f"{exc!r}" + (f"; {free / 1e6:.0f} MB free" if free is not None else "")
        self._event(tr.name, "write_failed", detail)
        try:
            tr.file.close()
        except Exception:
            pass
        self._alert("fault", f"{tr.name}: {detail}")

    def _flush(self, now: float) -> None:
        self._last_flush = now
        for tr in self._tracks:
            if tr.fault is not None:
                continue
            try:
                tr.file.flush()
            except Exception as exc:
                self._track_fault(tr, exc)
                continue
            if (tr.overflows, tr.dropped) != tr.reported and now - tr.reported_at >= 60.0:
                rate = tr.source.rate if tr.source is not None else SR
                self._event(tr.name, "overrun", f"overflows {tr.overflows}, dropped {tr.dropped / rate:.2f} s")
                tr.reported, tr.reported_at = (tr.overflows, tr.dropped), now
        free = _free_bytes(self.folder)
        if free is not None and free < LOW_DISK_BYTES:
            self._alert("disk_low", f"{free / 1e6:.0f} MB free")
        self._heartbeat("recording", now)

    def _check_devices(self, now: float) -> None:
        if self._endpoints is not None:
            try:
                if self._endpoints.ids() != self._ids:
                    if now - self._last_reopen >= REOPEN_GAP_S:
                        self._reopen("default device changed", now)
                    return
            except OSError as exc:
                log.warning("endpoint query failed: %r", exc)
        for tr in self._tracks:
            src = tr.source
            if src is None:
                continue
            since = tr.last_callback if tr.last_callback is not None else tr.opened_at
            reason = None
            if not src.active():
                reason = "stream stopped"
            elif tr.continuous and now - since > STALL_S:
                reason = "stalled"
            if reason is None:
                if tr.last_callback is not None and now - tr.opened_at > 2 * STALL_S:
                    tr.backoff = REOPEN_GAP_S          # delivering again: the next fault starts the ladder afresh
                continue
            if now >= tr.next_retry:
                self._reattach(tr, reason, now)
        if any(tr.source is None for tr in self._tracks) and now - self._last_missing >= MISSING_RETRY_S:
            self._last_missing = now
            if now - self._last_refresh >= self._missing_refresh_s:
                # A device may have appeared since the last enumeration: only a refresh can see it.
                self._missing_refresh_s = min(self._missing_refresh_s * 2, MISSING_REFRESH_MAX_S)
                self._reopen("missing track", now)
            else:
                with _pa_lock:
                    self._open_streams()

    def _heartbeat(self, status: str, seconds: float) -> None:
        with self._events_lock:
            events = list(self._events)
        doc = {"status": status, "started": self._started,
               "last_flush": datetime.now().isoformat(timespec="milliseconds"),
               "seconds": round(seconds, 3), "events": events}
        path = self.folder / "recording.json"
        tmp = path.with_name("recording.json.tmp")
        for attempt in range(5):
            try:
                # flushed and fsynced before the rename, so a power cut never leaves a renamed empty file
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(doc, indent=2))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
                return
            except OSError as exc:  # a reader holding the file open on Windows; try again shortly
                if attempt == 4:
                    log.error("recording.json not written: %r", exc)
                time.sleep(0.05)

    def _run(self) -> None:
        try:
            try:
                self._endpoints = _Endpoints()
            except OSError as exc:
                log.warning("default-device watch unavailable: %r", exc)
            while not self._stop_evt.wait(TICK_S):
                try:
                    self._tick()
                except Exception as exc:
                    log.exception("recorder tick")
                    self._event("all", "error", repr(exc))
        finally:
            self._finish()

    def _finish(self) -> None:
        end = self._end_s if self._end_s is not None else self._now()
        with _pa_lock:
            self._close_streams()
        for tr in self._tracks:
            if tr.fault is not None:
                continue
            try:
                self._drain(tr, end_s=end)
                tr.write_zeros(tr.clock.finish(end))
                over = -frames_owed(tr.clock.written, end)
                if over > 0:
                    self._event(tr.name, "long", f"{over / SR:.3f} s past the end")
            except Exception as exc:
                log.exception("finishing %s", tr.name)
                self._event(tr.name, "error", repr(exc))
            finally:
                try:
                    tr.file.close()
                except Exception as exc:
                    log.error("closing %s: %r", tr.path, exc)
        if self._endpoints is not None:
            self._endpoints.close()
            self._endpoints = None
        self._seconds = end
        self._heartbeat("stopped", end)
        self._result = {"mic": self.mic.path, "system": self.system.path, "seconds": end,
                        "events": list(self._events)}
        log.info("recording stopped: %.1f s, %d events", end, len(self._events))


# --------------------------------------------------------------------------------------------------
# Files: mix and m4a
# --------------------------------------------------------------------------------------------------

def _free_bytes(folder: Path) -> int | None:
    try:
        return shutil.disk_usage(folder).free
    except OSError:
        return None


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioError("ffmpeg not on PATH")
    return exe


class _TrackReader:
    """Reads a track as float32 16 kHz mono blocks. soundfile for a finished file; ffmpeg for anything
    else, including a FLAC left unfinalised by a crash (its header claims 2**63 frames)."""

    def __init__(self, path: Path):
        self.path = path
        self._sf = None
        self._proc = None
        try:
            info = sf.info(str(path))
            sane = info.samplerate == SR and info.channels == 1 and 0 <= info.frames < 2 ** 40
        except Exception:
            sane = False
        if sane:
            self._sf = sf.SoundFile(str(path))
        else:
            self._proc = subprocess.Popen(
                [_ffmpeg(), "-v", "error", "-nostdin", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)

    def read(self, n: int) -> np.ndarray:
        if self._sf is not None:
            try:
                return self._sf.read(n, dtype="float32", always_2d=False)
            except Exception as exc:  # a damaged tail: keep what was read
                log.warning("%s unreadable past %d frames: %r", self.path.name, self._sf.tell(), exc)
                return np.zeros(0, dtype=np.float32)
        buf = bytearray()
        while len(buf) < 2 * n:
            part = self._proc.stdout.read(2 * n - len(buf))
            if not part:
                break
            buf += part
        buf = buf[: len(buf) - len(buf) % 2]
        return np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0

    def close(self) -> None:
        if self._sf is not None:
            self._sf.close()
        if self._proc is not None:
            self._proc.stdout.close()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def _mixed_blocks(paths: list[Path], block: int) -> Iterator[np.ndarray]:
    """Sum of the tracks, block by block; a shorter track is treated as silence once it ends."""
    readers = [_TrackReader(p) for p in paths]
    try:
        while True:
            parts = [r.read(block) for r in readers]
            n = max((len(p) for p in parts), default=0)
            if n == 0:
                return
            out = np.zeros(n, dtype=np.float32)
            for p in parts:
                out[: len(p)] += p
            yield out
    finally:
        for r in readers:
            r.close()


def mix_tracks(folder: Path) -> Path:
    """mic.flac + system.flac -> mix.flac (16 kHz mono, peak -1 dBFS). Two streaming passes (peak, then
    write) so RAM stays flat for any length. A missing, empty or crash-truncated track is tolerated."""
    folder = Path(folder)
    paths = [p for p in (folder / "mic.flac", folder / "system.flac") if p.exists() and p.stat().st_size > 0]
    if not paths:
        raise AudioError(f"no tracks in {folder}")
    block = SR * 60
    peak, frames = 0.0, 0
    for b in _mixed_blocks(paths, block):
        peak = max(peak, float(np.max(np.abs(b))))
        frames += len(b)
    if frames == 0:
        raise AudioError(f"tracks in {folder} hold no audio")
    gain = (10 ** (PEAK_DBFS / 20)) / peak if peak > 0 else 1.0
    out, tmp = folder / "mix.flac", folder / "mix.flac.part"
    with sf.SoundFile(str(tmp), "w", SR, 1, format="FLAC", subtype="PCM_16") as f:
        for b in _mixed_blocks(paths, block):
            f.write(_to_int16(b * gain))
    os.replace(tmp, out)
    log.info("mixed %s: %.1f s from %s, gain %.2f", out, frames / SR, [p.name for p in paths], gain)
    return out


def to_m4a(src: Path, dst: Path) -> Path:
    """ffmpeg AAC 48 kbps mono, for shelving beside the transcript. Written to a .part file, then renamed,
    so a half-written m4a never looks finished."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    cmd = [_ffmpeg(), "-v", "error", "-nostdin", "-y", "-i", str(src), "-vn", "-ac", "1",
           "-c:a", "aac", "-b:a", "48k", "-movflags", "+faststart", "-f", "ipod", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW)
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise AudioError(f"ffmpeg m4a failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[-400:]}")
    os.replace(tmp, dst)
    return dst
