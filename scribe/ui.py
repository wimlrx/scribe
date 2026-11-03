"""The widget, vault picker and tray icon.

Tk owns the main thread and is not thread-safe, so every public Overlay method may be called from any
thread: off the Tk thread the call is queued and run by a `root.after` poll. The widget and picker are
borderless, topmost layered windows that never take focus (WS_EX_NOACTIVATE); their pixels are glass
frames drawn by glass.py and handed to UpdateLayeredWindow, so the user's cursor and focus stay where
they were.
"""
from __future__ import annotations

import ctypes
import functools
import logging
import math
import queue
import threading
import time
from collections import deque
from ctypes import wintypes
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from . import config, glass

log = logging.getLogger("scribe.ui")

# ---------------------------------------------------------------- look

PILL_BG = "#1F2328"
PILL_FG = "#E6EDF3"
RED = "#E5534B"
GREEN = "#3FB950"
AMBER = "#D29922"
GREY = "#9AA0A6"
STATE_COLOURS = {"idle": GREY, "dictating": GREEN, "recording": RED, "working": AMBER}
STATES = tuple(STATE_COLOURS)
ICON_SIZE = 64                  # tray image; Windows scales it to the notification-area size
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

# ---------------------------------------------------------------- timing

POLL_MS = 30                    # cross-thread call queue
FLASH_MS = 1200                 # error symbol
OFFSCREEN = -32000

# ---------------------------------------------------------------- pure helpers (tested)


def fmt_elapsed(seconds: float) -> str:
    """Recording clock: m:ss under an hour, h:mm:ss past it."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def level_to_unit(rms: float, floor_db: float = -55.0) -> float:
    """Mic RMS (0..1) to 0..1 on a dB scale, so quiet speech still moves the bars."""
    if rms <= 0.0:
        return 0.0
    db = 20.0 * math.log10(min(1.0, rms))
    return max(0.0, min(1.0, (db - floor_db) / -floor_db))


def bar_shape(n: int) -> list[float]:
    """Relative bar heights: tallest in the middle."""
    if n <= 1:
        return [1.0] * max(n, 0)
    c = (n - 1) / 2.0
    return [1.0 - 0.45 * abs(i - c) / c for i in range(n)]


def level_bars(rms: float, n: int = 5, floor: float = 0.15) -> list[float]:
    """Bar heights 0..1 for a level; never below `floor`, so the bars read as bars when quiet."""
    u = level_to_unit(rms)
    return [max(floor, u * s) for s in bar_shape(n)]


def smooth(prev: float, target: float, attack: float = 0.6, release: float = 0.2) -> float:
    """Fast rise, slow fall: bars follow speech without flicker."""
    k = attack if target > prev else release
    return prev + (target - prev) * k


class Countdown:
    """Seconds left on the vault picker. Pure: the clock is injectable."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.deadline = clock() + max(0.0, float(seconds))

    def remaining(self) -> int:
        """Whole seconds left, rounded up; 0 once expired."""
        return max(0, math.ceil(self.deadline - self.clock() - 1e-9))

    @property
    def expired(self) -> bool:
        return self.clock() >= self.deadline


def pick_state(dictating: bool, recording: bool, jobs: int) -> tuple[str, int]:
    """Tray state from what is happening: dictating > recording > working > idle. The count rides along."""
    count = max(0, int(jobs))
    if dictating:
        return "dictating", count
    if recording:
        return "recording", count
    if count:
        return "working", count
    return "idle", 0


def tray_title(state: str, count: int = 0, waiting: bool = False) -> str:
    """Tray tooltip: a label and a count, nothing else; "⏳ GPU" while a job waits for the GPU."""
    title = f"Scribe · {count}" if count else "Scribe"
    return f"{title} · ⏳ GPU" if waiting else title


def toggle_text(recording: bool) -> str:
    return "Stop notetaker" if recording else "Start notetaker"


def menu_labels(recording: bool) -> list[str]:
    """Visible tray menu, top to bottom ("-" is a separator). Retry shows only while a job has failed;
    Send lists this session's finished Local jobs, each with the vaults to send it to."""
    return [toggle_text(recording), "Add file…", "Retry", "Send", "-", "Recordings", "Inbox", "-", "Quit"]


UNATTENDED = "Local"            # what the vault picker settles on when nobody clicks: nothing leaves the machine


def unattended_pick(names: list[str]) -> str:
    """The picker's choice on expiry. Never a vault: a meeting nobody chose a vault for stays Local, so a
    personal recording can never be shelved into a work vault by a timeout."""
    return UNATTENDED if UNATTENDED in names else names[-1]


def layout_row(widths: list[float], start: float, gap: float) -> list[tuple[float, float]]:
    """(x0, x1) of items laid left to right from `start` with `gap` between them."""
    out, x = [], start
    for w in widths:
        out.append((x, x + w))
        x += w + gap
    return out


def bottom_centre(work: tuple[int, int, int, int], w: int, h: int, margin: int) -> tuple[int, int]:
    """Top-left corner for a window centred above the bottom edge of the work area (l, t, r, b)."""
    left, _top, right, bottom = work
    return (left + right - w) // 2, bottom - h - margin


def bottom_right(work: tuple[int, int, int, int], w: int, h: int, margin: int) -> tuple[int, int]:
    """Top-left corner for a window in the bottom right corner of the work area, above the clock."""
    _left, _top, right, bottom = work
    return right - w - margin, bottom - h - margin


def top_right(work: tuple[int, int, int, int], w: int, h: int, margin: int, row: int = 0,
              row_h: int = 0, gap: int = 0) -> tuple[int, int]:
    """Top-left corner for a window at the top right of the work area, `row` rows down."""
    left, top, right, _bottom = work
    return right - w - margin, top + margin + row * (row_h + gap)


# ---------------------------------------------------------------- icon (Pillow; also used by install.ps1)


def _font(px: int):
    try:
        return ImageFont.truetype("segoeui.ttf", px)
    except OSError:
        try:
            return ImageFont.load_default(px)
        except TypeError:
            return ImageFont.load_default()


def draw_icon(state: str = "idle", count: int = 0, size: int = ICON_SIZE) -> Image.Image:
    """Mic glyph in the state's colour on a transparent square, with a job-count badge when count > 0.
    Drawn at 4x and downsampled so the edges stay smooth at 16 px."""
    colour = STATE_COLOURS.get(state, GREY)
    k = 4
    s = size * k
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = s / 2
    # capsule
    cw, ct, cb = 0.34 * s, 0.06 * s, 0.58 * s
    d.rounded_rectangle((cx - cw / 2, ct, cx + cw / 2, cb), radius=cw / 2, fill=colour)
    # holder: the lower half of a ring around the capsule
    lw = max(k, int(0.075 * s))
    d.arc((0.19 * s, 0.24 * s, 0.81 * s, 0.74 * s), start=0, end=180, fill=colour, width=lw)
    # stem and base
    d.rectangle((cx - lw / 2, 0.73 * s, cx + lw / 2, 0.86 * s), fill=colour)
    d.rounded_rectangle((0.31 * s, 0.84 * s, 0.69 * s, 0.84 * s + lw), radius=lw / 2, fill=colour)
    if count > 0:
        r = 0.25 * s
        bx, by = s - r - k, s - r - k
        d.ellipse((bx - r, by - r, bx + r, by + r), fill=PILL_BG, outline=colour, width=max(k, int(0.04 * s)))
        label = str(count) if count < 10 else "9+"
        font = _font(int(r * (1.25 if len(label) == 1 else 0.95)))
        d.text((bx, by), label, fill=PILL_FG, font=font, anchor="mm")
    return img.resize((size, size), Image.Resampling.LANCZOS)


def write_ico(path: Path | str, state: str = "recording") -> Path:
    """scribe.ico for the shortcuts, from the tray glyph (each size drawn at its own resolution)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [draw_icon(state, 0, n) for n in ICO_SIZES]
    images[-1].save(str(path), format="ICO", sizes=[(n, n) for n in ICO_SIZES], append_images=images[:-1])
    return path


# ---------------------------------------------------------------- cross-thread calls


class CallQueue:
    """Thread-safe queue of calls for the Tk thread. A failing call is logged and never stops the rest."""

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()

    def put(self, fn: Callable, *args, **kwargs) -> None:
        self._q.put((fn, args, kwargs))

    def run_pending(self, limit: int = 200) -> int:
        """Run up to `limit` queued calls in order; returns how many ran."""
        n = 0
        while n < limit:
            try:
                fn, args, kwargs = self._q.get_nowait()
            except queue.Empty:
                break
            n += 1
            try:
                fn(*args, **kwargs)
            except Exception:
                log.exception("ui call %s failed", getattr(fn, "__name__", fn))
        return n


def on_tk(method):
    """Run the method inline on the owner's Tk thread; from any other thread, queue it for that thread."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if threading.current_thread() is self._thread:
            return method(self, *args, **kwargs)
        self._calls.put(method, self, *args, **kwargs)
        return None

    return wrapper


# ---------------------------------------------------------------- Win32

_user32 = ctypes.WinDLL("user32", use_last_error=True)
LONG_PTR = ctypes.c_ssize_t


def _fn(dll, name: str, restype, *argtypes):
    f = getattr(dll, name)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f


_GetWindowLongPtrW = _fn(_user32, "GetWindowLongPtrW", LONG_PTR, wintypes.HWND, ctypes.c_int)
_SetWindowLongPtrW = _fn(_user32, "SetWindowLongPtrW", LONG_PTR, wintypes.HWND, ctypes.c_int, LONG_PTR)
_SetWindowPos = _fn(_user32, "SetWindowPos", wintypes.BOOL, wintypes.HWND, wintypes.HWND,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT)
_ShowWindow = _fn(_user32, "ShowWindow", wintypes.BOOL, wintypes.HWND, ctypes.c_int)
_GetAncestor = _fn(_user32, "GetAncestor", wintypes.HWND, wintypes.HWND, wintypes.UINT)
_IsWindow = _fn(_user32, "IsWindow", wintypes.BOOL, wintypes.HWND)
_SystemParametersInfoW = _fn(_user32, "SystemParametersInfoW", wintypes.BOOL,
                             wintypes.UINT, wintypes.UINT, ctypes.c_void_p, wintypes.UINT)

WIN32_FUNCTIONS = (_GetWindowLongPtrW, _SetWindowLongPtrW, _SetWindowPos, _ShowWindow, _GetAncestor, _IsWindow,
                   _SystemParametersInfoW)

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
SWP_FRAMECHANGED, SWP_SHOWWINDOW = 0x0020, 0x0040
SW_HIDE = 0
GA_ROOT = 2
SPI_GETWORKAREA = 0x0030


def pill_exstyle(click_through: bool) -> int:
    """Extended style for a pill: never activated, no taskbar button, topmost, layered; click-through if asked."""
    style = WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_LAYERED
    return style | WS_EX_TRANSPARENT if click_through else style


def enable_dpi_awareness() -> None:
    """System DPI awareness for this process, before Tk starts, so text is crisp on a scaled display."""
    try:
        shcore = ctypes.WinDLL("shcore")
        f = _fn(shcore, "SetProcessDpiAwareness", ctypes.c_long, ctypes.c_int)
        f(1)
    except (OSError, AttributeError):
        log.debug("DPI awareness not set")


def work_area() -> tuple[int, int, int, int]:
    """Primary monitor's work area (left, top, right, bottom), i.e. the screen minus the taskbar."""
    rect = wintypes.RECT()
    if _SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
        return rect.left, rect.top, rect.right, rect.bottom
    return 0, 0, 1920, 1040


# ---------------------------------------------------------------- layered windows

_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
_shell32 = ctypes.WinDLL("shell32", use_last_error=True)


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class _BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


_UpdateLayeredWindow = _fn(_user32, "UpdateLayeredWindow", wintypes.BOOL, wintypes.HWND, wintypes.HDC,
                           ctypes.POINTER(wintypes.POINT), ctypes.POINTER(wintypes.SIZE), wintypes.HDC,
                           ctypes.POINTER(wintypes.POINT), wintypes.COLORREF, ctypes.POINTER(_BLENDFUNCTION),
                           wintypes.DWORD)
_GetCursorPos = _fn(_user32, "GetCursorPos", wintypes.BOOL, ctypes.POINTER(wintypes.POINT))
_GetDC = _fn(_user32, "GetDC", wintypes.HDC, wintypes.HWND)
_ReleaseDC = _fn(_user32, "ReleaseDC", ctypes.c_int, wintypes.HWND, wintypes.HDC)
_CreateCompatibleDC = _fn(_gdi32, "CreateCompatibleDC", wintypes.HDC, wintypes.HDC)
_DeleteDC = _fn(_gdi32, "DeleteDC", wintypes.BOOL, wintypes.HDC)
_CreateDIBSection = _fn(_gdi32, "CreateDIBSection", wintypes.HBITMAP, wintypes.HDC,
                        ctypes.POINTER(_BITMAPINFOHEADER), wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                        wintypes.HANDLE, wintypes.DWORD)
_SelectObject = _fn(_gdi32, "SelectObject", wintypes.HGDIOBJ, wintypes.HDC, wintypes.HGDIOBJ)
_DeleteObject = _fn(_gdi32, "DeleteObject", wintypes.BOOL, wintypes.HGDIOBJ)
_SHQueryUserNotificationState = _fn(_shell32, "SHQueryUserNotificationState", ctypes.c_long,
                                    ctypes.POINTER(ctypes.c_int))

WIN32_FUNCTIONS = WIN32_FUNCTIONS + (_UpdateLayeredWindow, _GetCursorPos, _GetDC, _ReleaseDC, _CreateCompatibleDC,
                                     _DeleteDC, _CreateDIBSection, _SelectObject, _DeleteObject,
                                     _SHQueryUserNotificationState)

ULW_ALPHA, AC_SRC_ALPHA, BI_RGB, DIB_RGB_COLORS = 2, 1, 0, 0
ERROR_INVALID_PARAMETER = 87
QUNS_FULL_SCREEN = (2, 3, 4)        # busy (full-screen app), Direct3D full screen, presentation mode


def fullscreen_busy() -> bool:
    """True while a full-screen app, game or presentation has the screen: the resting pill keeps out of it."""
    state = ctypes.c_int(0)
    try:
        if _SHQueryUserNotificationState(ctypes.byref(state)) != 0:
            return False
    except OSError:
        return False
    return state.value in QUNS_FULL_SCREEN


def cursor_pos() -> tuple[int, int] | None:
    pt = wintypes.POINT()
    return (pt.x, pt.y) if _GetCursorPos(ctypes.byref(pt)) else None


class Clicks:
    """A click on a control: the button goes down over a control and comes up over the same one. Fed with
    each press and release the window itself receives (see bind_button). Pure; tested."""

    def __init__(self):
        self.was_down = False
        self.pressed: object = None

    def feed(self, down: bool, over: object) -> object:
        """The control clicked on this frame, or None."""
        clicked = None
        if down and not self.was_down:
            self.pressed = over
        elif not down and self.was_down:
            if self.pressed is not None and over == self.pressed:
                clicked = self.pressed
            self.pressed = None
        self.was_down = down
        return clicked


def bind_button(top, on_button: Callable[[bool, tuple[int, int]], None]) -> None:
    """Hand a toplevel's own left-button press and release to on_button(down, screen point). These are the
    window's messages, not a polled button state, so a tap whose press and release both fall between two
    frames still arrives as both. A failing handler is logged and never reaches Tk."""

    def handler(down: bool):
        def run(event) -> None:
            try:
                on_button(down, (int(event.x_root), int(event.y_root)))
            except Exception:
                log.exception("overlay click failed")
        return run

    top.bind("<ButtonPress-1>", handler(True))
    top.bind("<ButtonRelease-1>", handler(False))


def _native_hwnd(top) -> int:
    """The top-level HWND Tk wraps a toplevel in (its frame), found from the inner window."""
    hwnd = _GetAncestor(top.winfo_id(), GA_ROOT)
    if hwnd and _IsWindow(hwnd):
        return hwnd
    return int(top.wm_frame(), 16)


class _Layered:
    """A borderless, topmost, never-activated Tk toplevel whose pixels come from UpdateLayeredWindow, so every
    pixel carries its own alpha: round antialiased edges, a soft shadow, glass the desktop shows through.
    Its own button messages are bound by the overlay (see bind_button); fully transparent pixels let clicks
    through to what is below."""

    def __init__(self, root, click_through: bool = False):
        import tkinter as tk

        top = tk.Toplevel(root, bd=0, highlightthickness=0)
        top.withdraw()
        top.overrideredirect(True)
        top.geometry(f"1x1+{OFFSCREEN}+{OFFSCREEN}")
        top.deiconify()
        top.update_idletasks()
        if not top.winfo_ismapped():
            top.update()  # the native window is created when Tk first maps the toplevel
        self.top = top
        self.hwnd = _native_hwnd(top)
        style = (_GetWindowLongPtrW(self.hwnd, GWL_EXSTYLE) | pill_exstyle(click_through)) & ~WS_EX_APPWINDOW
        _SetWindowLongPtrW(self.hwnd, GWL_EXSTYLE, style)
        _SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                      SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED)
        _ShowWindow(self.hwnd, SW_HIDE)
        self.visible = False
        self.x = self.y = OFFSCREEN
        self.size = (0, 0)
        self._mem = None            # (memory DC, DIB section, previous object, bits pointer, size)
        self._failed = False

    def _surface(self, size: tuple[int, int]):
        if self._mem is not None and self._mem[4] == size:
            return self._mem
        self._free()
        screen = _GetDC(None)
        try:
            hdc = _CreateCompatibleDC(screen)
            header = _BITMAPINFOHEADER(ctypes.sizeof(_BITMAPINFOHEADER), size[0], -size[1], 1, 32, BI_RGB,
                                       0, 0, 0, 0, 0)
            bits = ctypes.c_void_p()
            hbm = _CreateDIBSection(screen, ctypes.byref(header), DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
        finally:
            _ReleaseDC(None, screen)
        if not hdc or not hbm or not bits.value:
            if hbm:
                _DeleteObject(hbm)
            if hdc:
                _DeleteDC(hdc)
            raise OSError("layered surface not created")
        old = _SelectObject(hdc, hbm)
        self._mem = (hdc, hbm, old, bits, size)
        return self._mem

    def _free(self) -> None:
        if self._mem is not None:
            hdc, hbm, old, _bits, _size = self._mem
            _SelectObject(hdc, old)
            _DeleteObject(hbm)
            _DeleteDC(hdc)
            self._mem = None

    def present(self, img: Image.Image, x: int, y: int) -> bool:
        """Put the frame on screen at (x, y) and show the window (without activating it)."""
        size = img.size
        try:
            hdc, _hbm, _old, bits, _size = self._surface(size)
        except OSError:
            if not self._failed:
                log.warning("overlay: layered surface not created")
                self._failed = True
            return False
        data = glass.premultiplied_bgra(img)
        ctypes.memmove(bits, data, len(data))
        ok = self._update(hdc, x, y, size)
        if not ok and ctypes.get_last_error() == ERROR_INVALID_PARAMETER:
            # Tk layers its wrapper through SetLayeredWindowAttributes after creating it, which locks out
            # UpdateLayeredWindow. Taking WS_EX_LAYERED off and putting it back releases it, for good.
            style = _GetWindowLongPtrW(self.hwnd, GWL_EXSTYLE)
            _SetWindowLongPtrW(self.hwnd, GWL_EXSTYLE, style & ~WS_EX_LAYERED)
            _SetWindowLongPtrW(self.hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED)
            ok = self._update(hdc, x, y, size)
        if not ok:
            if not self._failed:
                log.warning("overlay: UpdateLayeredWindow failed (%d)", ctypes.get_last_error())
                self._failed = True
            return False
        if (x, y, size) != (self.x, self.y, self.size):
            self.x, self.y, self.size = x, y, size
            self.top.geometry(f"{size[0]}x{size[1]}+{x}+{y}")   # Tk's own idea of the window, for its events
        if not self.visible:
            _SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                          SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
            self.visible = True
        return True

    def _update(self, hdc, x: int, y: int, size: tuple[int, int]) -> bool:
        blend = _BLENDFUNCTION(0, 0, 255, AC_SRC_ALPHA)
        return bool(_UpdateLayeredWindow(self.hwnd, None, ctypes.byref(wintypes.POINT(x, y)),
                                         ctypes.byref(wintypes.SIZE(*size)), hdc,
                                         ctypes.byref(wintypes.POINT(0, 0)), 0, ctypes.byref(blend), ULW_ALPHA))

    def hide(self) -> None:
        if self.visible:
            _ShowWindow(self.hwnd, SW_HIDE)
            self.visible = False

    def contains(self, pt: tuple[int, int] | None) -> tuple[float, float] | None:
        """The screen point in window coordinates, if it falls inside the visible window."""
        if pt is None or not self.visible:
            return None
        x, y = pt[0] - self.x, pt[1] - self.y
        return (x, y) if 0 <= x < self.size[0] and 0 <= y < self.size[1] else None

    def destroy(self) -> None:
        self._free()
        try:
            self.top.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------- overlay

FRAME_MS = 16                   # while anything moves
SLOW_MS = 60                    # resting pill breathing its recording dot
IDLE_MS = 90                    # nothing moving: pointer watch only
CONFIRM_S = 2.5                 # the recording view shows this long after a start, then folds away
FULLSCREEN_CHECK_S = 1.0
PICKER_GAP = 13                 # logical px between the picker and the widget at its tallest: clears its shadow at any scale
CLICK_GUARD_S = 0.6             # a second click on the same control this soon is ignored
LIVE_CONTENT = 0.9              # a control takes the pointer only once its symbol is (nearly) drawn
BAND_W, BAND_UP, BAND_DOWN = 34, 22, 6   # logical px around the resting pill's bottom centre that catch the pointer


def pulse(phase: float, period: float = 1.6) -> float:
    """0..1 breathing for the recording dot."""
    return 0.5 + 0.5 * math.sin(phase * 2 * math.pi / period)


class Overlay:
    """The widget (bottom right above the clock, or bottom centre) and the vault picker just above it, both
    glass layered windows.

    The widget rests as a small faint pill. The pointer over it turns it into a start button for the
    notetaker (or, while recording, the elapsed time and a stop button); a recording start shows the time
    for a moment and folds back to the resting pill with a breathing copper dot. Ctrl+Win dictation grows it
    into level bars, then a working wave. Construct on the Tk thread. Every public method is safe from any
    thread."""

    def __init__(self, root, *, idle_pill: bool = True, anchor: str = "right"):
        self.root = root
        self.anchor = anchor                   # "right": above the clock; "centre": bottom centre
        self._thread = threading.current_thread()
        self._calls = CallQueue()
        self.scale = max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
        self.idle_pill = idle_pill
        self.on_toggle: Callable[[], None] | None = None     # the widget's start button (the app's toggle)
        self._win: _Layered | None = None
        self._pick_win: _Layered | None = None
        self._mode: str | None = None          # dictation: "listening" | "working" | "error" | None
        self._token = 0                        # bumps on every dictation change
        self._level_fn: Callable[[], float] | None = None
        self._level = 0.0
        self._error_until = 0.0
        self._recording = False
        self._elapsed_fn: Callable[[], float] | None = None
        self._on_stop: Callable[[], None] | None = None
        self._confirm_until = 0.0
        self._view = "rest"                    # what the widget shows (or last showed, while fading out)
        self._w, self._h = (float(v) for v in glass.SIZES["rest"])
        self._content = 1.0
        self._opacity = 0.0
        self._over = False                     # pointer over the widget
        self._hover: str | None = None         # control under the pointer
        self._last_click = 0.0
        self._clicks = Clicks()
        self._pick_clicks = Clicks()
        self._drawn = None                     # what the last widget frame showed
        self._phase = 0.0
        self._last = time.monotonic()
        self._fullscreen = False
        self._fs_checked = 0.0
        self._picker: dict | None = None       # the active picker request
        self._picks: deque = deque()           # picker requests waiting their turn
        self._pick_hover: int | None = None
        self._pick_drawn = None
        self._closed = False
        root.after(POLL_MS, self._poll)
        root.after(FRAME_MS, self._frame)

    # -- plumbing

    def call(self, fn: Callable, *args, **kwargs) -> None:
        """Run fn on the Tk thread (inline when already there)."""
        if threading.current_thread() is self._thread:
            fn(*args, **kwargs)
        else:
            self._calls.put(fn, *args, **kwargs)

    def _poll(self) -> None:
        self._calls.run_pending()
        if not self._closed:
            self.root.after(POLL_MS, self._poll)

    def _px(self, v: float) -> int:
        return int(round(v * self.scale))

    def _widget_window(self) -> _Layered:
        if self._win is None:
            self._win = _Layered(self.root)
            bind_button(self._win.top, self._widget_button)
        return self._win

    @staticmethod
    def _spawn(fn: Callable[[], None] | None, what: str) -> None:
        """Run an app callback off the Tk thread: it may open devices or wait on a lock."""
        if fn is None:
            return

        def run():
            try:
                fn()
            except Exception:
                log.exception("%s failed", what)

        threading.Thread(target=run, name=f"scribe-{what}", daemon=True).start()

    # -- dictation

    @on_tk
    def show_listening(self, level_fn: Callable[[], float]) -> None:
        self._level_fn, self._level = level_fn, 0.0
        self._mode = "listening"
        self._token += 1

    @on_tk
    def show_working(self) -> None:
        self._mode = "working"
        self._token += 1

    @on_tk
    def hide(self) -> None:
        self._mode = None
        self._token += 1

    @on_tk
    def flash_error(self) -> None:
        self._mode = "error"
        self._error_until = time.monotonic() + FLASH_MS / 1000
        self._token += 1

    # -- recording

    @on_tk
    def show_recording(self, elapsed_fn: Callable[[], float], on_stop: Callable[[], None]) -> None:
        self._recording = True
        self._elapsed_fn, self._on_stop = elapsed_fn, on_stop
        self._confirm_until = time.monotonic() + CONFIRM_S

    @on_tk
    def hide_recording(self) -> None:
        self._recording = False
        self._elapsed_fn = self._on_stop = None
        self._confirm_until = 0.0

    def set_toggle(self, fn: Callable[[], None] | None) -> None:
        """The callback for the widget's start button (the app's notetaker toggle)."""
        self.on_toggle = fn

    # -- the frame loop

    def _frame(self) -> None:
        if self._closed:
            return
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - self._last))
        self._last = now
        self._phase += dt
        if now - self._fs_checked >= FULLSCREEN_CHECK_S:
            self._fs_checked = now
            self._fullscreen = fullscreen_busy()
        try:
            pt = cursor_pos()           # hover only: clicks come from the windows' own messages
            delay = min(self._widget_step(now, dt, pt), self._picker_step(now, pt))
        except Exception:
            log.exception("overlay frame failed")
            delay = IDLE_MS
        self.root.after(delay, self._frame)

    def _target_view(self, now: float) -> str | None:
        if self._mode == "error" and now >= self._error_until:
            self._mode = None
        if self._mode:
            return self._mode
        if self._recording and (now < self._confirm_until or self._over):
            return "rec"
        if self._over and not self._recording and self.on_toggle is not None:
            return "start"
        resting = (self.idle_pill or self._recording or self._picker is not None) and not self._fullscreen
        return "rest" if resting else None

    def _pill_screen_rect(self, slack: float) -> tuple[float, float, float, float] | None:
        """The pill's rectangle on screen, grown by `slack` logical px, for the pointer test."""
        win = self._win
        if win is None or not win.visible:
            return None
        x0, y0, x1, y1 = glass.widget_rect(self.scale, self._w, self._h)
        s = slack * self.scale
        return win.x + x0 - s, win.y + y0 - s, win.x + x1 + s, win.y + y1 + s

    def _band_screen_rect(self) -> tuple[float, float, float, float] | None:
        """The band that catches the pointer around the resting pill: the pill is small. It hangs off the
        resting rect, never the animated one, whose centre shifts half a pixel as its width rounds, so the
        band is the same in every view."""
        win = self._win
        if win is None or not win.visible:
            return None
        x0, _y0, x1, y1 = glass.widget_rect(self.scale, *glass.SIZES["rest"])
        cx, bottom, s = win.x + (x0 + x1) / 2, win.y + y1, self.scale
        return cx - BAND_W * s, bottom - BAND_UP * s, cx + BAND_W * s, bottom + BAND_DOWN * s

    def _track_pointer(self, pt: tuple[int, int] | None) -> None:
        """Whether the pointer is over the widget, and the control under it. Each frame with the polled
        cursor, and with each press and release at the event's own point."""
        if pt is None or self._mode is not None:
            self._over, self._hover = False, None
            return
        zones = [self._pill_screen_rect(8 if self._over else 2)]
        if self._view in ("rest", "start", "rec"):
            # the band also holds in the views it opens, so a point that opens start or rec keeps it open
            zones.append(self._band_screen_rect())
        self._over = any(r is not None and r[0] <= pt[0] <= r[2] and r[1] <= pt[1] <= r[3] for r in zones)
        local = self._win.contains(pt) if (self._over and self._win is not None) else None
        # no control before its symbol is drawn, and none under a picker chip: one click, one action
        live = local is not None and self._content >= LIVE_CONTENT and self._chip_at(pt) is None
        self._hover = glass.widget_hit(self.scale, self._w, self._h, self._view, *local) if live else None

    def _widget_button(self, down: bool, pt: tuple[int, int]) -> None:
        """A press or release on the widget window, hit-tested at its own point exactly as hover is."""
        self._track_pointer(pt)
        clicked = self._clicks.feed(down, self._hover)
        if clicked:
            self._click(clicked)

    def _widget_step(self, now: float, dt: float, pt) -> int:
        self._track_pointer(pt)
        view = self._target_view(now)
        if view is not None and view != self._view:
            self._view, self._content = view, 0.0
        tw, th = glass.SIZES[self._view]
        target_op = 0.0 if view is None else (glass.REST_OPACITY if self._view == "rest" else 1.0)
        k = 1.0 - math.exp(-dt * 16)
        self._w += (tw - self._w) * k
        self._h += (th - self._h) * k
        self._opacity += (target_op - self._opacity) * (1.0 - math.exp(-dt * 12))
        near = abs(tw - self._w) < 1.5 and abs(th - self._h) < 1.5
        if near:
            self._w, self._h = float(tw), float(th)
            self._content = min(1.0, self._content + dt * 7)
        if abs(target_op - self._opacity) < 0.01:
            self._opacity = target_op
        settled = near and self._content >= 1.0 and self._opacity == target_op

        if view is None and self._opacity <= 0.0:
            if self._win is not None:
                self._win.hide()
            self._drawn = None
            return IDLE_MS

        heights, clock, p = None, "", pulse(self._phase)
        if self._view == "listening":
            try:
                raw = float(self._level_fn()) if self._level_fn else 0.0
            except Exception:
                raw = 0.0
            self._level = smooth(self._level, level_to_unit(raw))
            heights = glass.listening_heights(self._level, self._phase)
        elif self._view == "working":
            heights = glass.working_heights(self._phase)
        elif self._view == "rec":
            try:
                clock = fmt_elapsed(float(self._elapsed_fn()) if self._elapsed_fn else 0.0)
            except Exception:
                clock = "0:00"
        moving = self._view in ("listening", "working", "rec")
        breathing = self._view == "rest" and self._recording
        area = work_area()              # in the state: a new screen layout moves even the idle pill
        state = (self._view, round(self._w, 1), round(self._h, 1), round(self._opacity, 2),
                 round(self._content, 2), self._hover, clock, self._recording, area)
        if moving or breathing or state != self._drawn:
            frame = glass.widget(self.scale, self._w, self._h, self._view, content=self._content,
                                 opacity=self._opacity, heights=heights, pulse=p, clock=clock,
                                 recording=self._recording, hover=self._hover)
            win = self._widget_window()
            x, y = self._widget_origin(area, frame.size[0], frame.size[1])
            win.present(frame, x, y)
            self._drawn = state
        if not settled or moving or self._over:
            return FRAME_MS
        return SLOW_MS if breathing else IDLE_MS

    def _click(self, control: str) -> None:
        now = time.monotonic()
        if now - self._last_click < CLICK_GUARD_S:
            return
        self._last_click = now
        if control == "start" and not self._recording:
            self._spawn(self.on_toggle, "notetaker")
        elif control == "stop" and self._recording:
            self._spawn(self._on_stop, "notetaker")

    # -- vault picker

    @on_tk
    def pick_vault(self, vaults: list[str], default: str, timeout_s: int, on_pick: Callable[[str], None]) -> None:
        names = list(dict.fromkeys(vaults))
        if not names:
            names = ["Local"]
        if default not in names:
            default = names[0]
        self._picks.append({"names": names, "default": default, "timeout": timeout_s, "on_pick": on_pick})
        if self._picker is None:
            self._next_picker()

    def _next_picker(self) -> None:
        if not self._picks:
            self._picker = None
            if self._pick_win is not None:
                self._pick_win.hide()
            self._pick_drawn = None
            return
        req = self._picks.popleft()
        req["countdown"] = Countdown(req["timeout"])
        self._picker = req
        self._pick_hover = None
        self._pick_drawn = None
        self._pick_clicks = Clicks()            # a press made for the last request never picks for this one

    def _picker_window(self) -> _Layered:
        if self._pick_win is None:
            self._pick_win = _Layered(self.root)
            bind_button(self._pick_win.top, self._picker_button)
        return self._pick_win

    def _chip_at(self, pt) -> int | None:
        """The vault chip at a screen point, while the picker is up."""
        req, win = self._picker, self._pick_win
        if req is None or win is None:
            return None
        local = win.contains(pt)
        return glass.hit_chip(self.scale, req["names"], *local) if local else None

    def _picker_button(self, down: bool, pt: tuple[int, int]) -> None:
        """A press or release on the picker window."""
        clicked = self._pick_clicks.feed(down, self._chip_at(pt))
        req = self._picker
        if clicked is not None and req is not None and clicked < len(req["names"]):
            self._resolve(req["names"][clicked])

    def _picker_step(self, now: float, pt) -> int:
        req = self._picker
        if req is None:
            return IDLE_MS
        cd: Countdown = req["countdown"]
        if cd.expired:
            self._resolve(unattended_pick(req["names"]))
            return IDLE_MS
        win = self._picker_window()
        self._pick_hover = self._chip_at(pt)
        total = max(0.001, float(req["timeout"]))
        fraction = max(0.0, cd.deadline - cd.clock()) / total
        state = (self._pick_hover, cd.remaining(), round(fraction, 3))
        if state != self._pick_drawn:
            frame = glass.picker(self.scale, req["names"], req["default"], self._pick_hover, cd.remaining(),
                                 fraction)
            left, _top, right, bottom = work_area()
            # above the widget's tallest pill, not the resting one: the start and stop buttons the pointer
            # opens on the way up must never reach under a chip, nor under the shadow that would take their click
            pill_bottom = bottom - (glass.MARGIN + glass.WIDGET_H + PICKER_GAP) * self.scale
            y = int(pill_bottom - (glass.PICK_H + glass.MARGIN) * self.scale)
            fw = frame.size[0]
            if self.anchor == "right":            # over the widget, kept on screen
                wx, _wy = self._widget_origin((left, _top, right, bottom), *glass.widget_canvas(self.scale))
                x = max(left, min(right - fw, wx + glass.widget_canvas(self.scale)[0] // 2 - fw // 2))
            else:
                x = (left + right - fw) // 2
            win.present(frame, x, y)
            self._pick_drawn = state
        return 33

    def _widget_origin(self, area: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int]:
        place = bottom_right if getattr(self, "anchor", "right") == "right" else bottom_centre
        return place(area, w, h, 0)

    def _resolve(self, name: str) -> None:
        req = self._picker
        if req is None:
            return
        self._picker = None
        if self._pick_win is not None:
            self._pick_win.hide()
        try:
            req["on_pick"](name)
        except Exception:
            log.exception("vault pick callback failed")
        self._next_picker()

    # -- file dialog (tray "Add file…")

    @on_tk
    def ask_files(self, on_files: Callable[[list[str]], None]) -> None:
        from tkinter import filedialog

        exts = " ".join(f"*{e}" for e in sorted(config.MEDIA_EXTS))
        paths = filedialog.askopenfilenames(parent=self.root, title="Add file",
                                            filetypes=[("Media", exts), ("All", "*.*")])
        if paths:
            try:
                on_files(list(paths))
            except Exception:
                log.exception("add-file callback failed")

    # -- teardown

    @on_tk
    def close(self) -> None:
        self._closed = True
        self._token += 1
        self._picker = None
        self._picks.clear()
        for w in (self._win, self._pick_win):
            if w is not None:
                w.destroy()
        self._win = self._pick_win = None


# ---------------------------------------------------------------- tray


class Tray:
    """Notification-area icon (pystray, run detached on its own thread). Left click toggles the notetaker.

    Every callback runs on pystray's thread; the app routes them on from there."""

    NOTICE_QUEUE = 10

    def __init__(self, on_toggle_notetaker: Callable[[], None], on_add_file: Callable[[], None],
                 on_open_recordings: Callable[[], None], on_open_inbox: Callable[[], None],
                 on_quit: Callable[[], None], *, is_recording: Callable[[], bool] | None = None,
                 on_retry: Callable[[], None] | None = None, has_failed: Callable[[], bool] | None = None,
                 sendable: Callable[[], list] | None = None, vaults: Callable[[], list[str]] | None = None,
                 on_send: Callable[[Path, str], None] | None = None):
        self.on_toggle_notetaker = on_toggle_notetaker
        self.on_add_file = on_add_file
        self.on_open_recordings = on_open_recordings
        self.on_open_inbox = on_open_inbox
        self.on_quit = on_quit
        self.on_retry, self._has_failed = on_retry, has_failed
        self._sendable, self._vaults, self.on_send = sendable, vaults, on_send
        self._is_recording = is_recording
        self._state, self._count, self._waiting = "idle", 0, False
        self._icon = None
        self._shown = False                       # the icon is in the notification area (NIM_ADD done)
        self._notes: list[tuple[str, str]] = []   # notices that arrived before that, sent once it is
        self._lock = threading.Lock()

    def _recording(self) -> bool:
        if self._is_recording is not None:
            try:
                return bool(self._is_recording())
            except Exception:
                return False
        return self._state == "recording"

    @staticmethod
    def _act(fn: Callable[[], None]):
        def action(_icon, _item):
            try:
                fn()
            except Exception:
                log.exception("tray action failed")
        return action

    def _failed(self) -> bool:
        try:
            return bool(self.on_retry) and bool(self._has_failed and self._has_failed())
        except Exception:
            return False

    def send_items(self) -> list[tuple[str, Path, list[str]]]:
        """(job label, folder, vaults) for the Send menu."""
        if self._sendable is None or self.on_send is None:
            return []
        try:
            vaults = list(self._vaults()) if self._vaults else []
            return [(label, folder, vaults) for label, folder in self._sendable()] if vaults else []
        except Exception:
            log.exception("tray send menu failed")
            return []

    def _send(self, folder: Path, vault: str):
        return self._act(lambda: self.on_send(folder, vault))

    def start(self) -> None:
        """Show the icon. pystray runs its message loop on its own thread; notices wait until it is up."""
        import pystray

        item, menu = pystray.MenuItem, pystray.Menu
        toggle = self._act(self.on_toggle_notetaker)
        labels = menu_labels(False)

        def send_menu():
            for label, folder, vaults in self.send_items():
                yield item(label, menu(*[item(v, self._send(folder, v)) for v in vaults]))

        self._icon = pystray.Icon(
            "Scribe", draw_icon(self._state, self._count), tray_title(self._state, self._count, self._waiting),
            menu(
                item("Notetaker", toggle, default=True, visible=False),  # left click, without a bold menu line
                item(lambda _i: toggle_text(self._recording()), toggle),
                item(labels[1], self._act(self.on_add_file)),
                item(labels[2], self._act(lambda: self.on_retry and self.on_retry()),
                     visible=lambda _i: self._failed()),
                item(labels[3], menu(send_menu), visible=lambda _i: bool(self.send_items())),
                menu.SEPARATOR,
                item(labels[5], self._act(self.on_open_recordings)),
                item(labels[6], self._act(self.on_open_inbox)),
                menu.SEPARATOR,
                item(labels[8], self._act(self.on_quit)),
            ),
        )
        self._icon.run_detached(setup=self._setup)

    def _setup(self, icon) -> None:
        """pystray's setup thread, once its message loop runs: show the icon, then send what was queued.
        Shell_NotifyIcon(NIM_MODIFY, NIF_INFO) before NIM_ADD fails without a word, so nothing is sent earlier."""
        try:
            icon.visible = True
        except Exception:
            log.exception("tray icon not shown")
            return
        with self._lock:
            self._shown = True
            notes, self._notes = self._notes, []
        for title, message in notes:
            self._send_notice(icon, title, message)

    def set_state(self, state: str, count: int = 0, waiting: bool = False) -> None:
        """'idle' | 'dictating' | 'recording' | 'working', plus the job count for the badge and whether a job
        waits for the GPU."""
        if state not in STATE_COLOURS:
            state = "idle"
        with self._lock:  # one update at a time: pystray rebuilds the native menu on each call
            changed = (state, count, waiting) != (self._state, self._count, self._waiting)
            self._state, self._count, self._waiting = state, count, waiting
            icon = self._icon
            if icon is None:
                return
            try:
                if changed:
                    icon.icon = draw_icon(state, count)
                    icon.title = tray_title(state, count, waiting)
                icon.update_menu()
            except Exception:
                log.exception("tray update failed")

    def notify(self, title: str, message: str) -> None:
        """Show a notice. Before the icon is visible (or before start) it is kept and sent once it is."""
        with self._lock:
            icon = self._icon
            if icon is None or not self._shown:
                self._notes = (self._notes + [(title, message)])[-self.NOTICE_QUEUE:]
                return
        self._send_notice(icon, title, message)

    @staticmethod
    def _send_notice(icon, title: str, message: str) -> None:
        try:
            icon.notify(message, title)
        except Exception:
            log.exception("tray notification failed")

    def stop(self) -> None:
        with self._lock:
            self._shown = False
        icon, self._icon = self._icon, None
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                log.exception("tray stop failed")
