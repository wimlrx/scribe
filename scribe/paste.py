"""Paste dictated text where the cursor is: swap it onto the clipboard, inject Ctrl+V, put the user's clipboard back.

Pure ctypes. Every clipboard write also carries CanUploadToCloudClipboard = 0, so dictation and restored
contents never sync to Microsoft's cloud clipboard.

The restore is the dangerous half: put the old clipboard back before the target app has read the dictation
and the app pastes the old contents instead (a password, client data). So the dictation is offered by
delayed rendering from a hidden message-only window: the target asks for the text when it handles Ctrl+V,
and the old contents go back only after that request. When the request cannot be seen, Scribe waits for
the target's message queue instead; when nothing reads the text at all, or Ctrl+V could not be delivered,
the dictation stays on the clipboard for a manual paste.
"""
from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Callable

from . import winkeys

log = logging.getLogger("scribe.paste")

CF_TEXT, CF_BITMAP, CF_METAFILEPICT, CF_OEMTEXT, CF_DIB = 1, 2, 3, 7, 8
CF_PALETTE, CF_UNICODETEXT, CF_ENHMETAFILE, CF_LOCALE, CF_DIBV5 = 9, 13, 14, 16, 17
CF_OWNERDISPLAY, CF_DSPENHMETAFILE = 0x80, 0x8E
CF_PRIVATEFIRST, CF_GDIOBJLAST = 0x200, 0x3FF
GMEM_MOVEABLE = 0x0002
OPEN_TRIES = 10
OPEN_SLEEP_S = 0.02
PASTE_SETTLE_S = 0.25          # pause after the target has read the text, before the old contents go back
RENDER_WAIT_S = 2.0            # how long the target may take to ask for the text after Ctrl+V
FALLBACK_SETTLE_S = 0.5        # extra wait when the target's read could not be seen
QUEUE_WAIT_MS = 2000           # SendMessageTimeout on the target's window: is its UI thread pumping again?
HOLD_WAIT_S = 600.0            # a hold Scribe started is waited out (MicCapture's own cap)
SNAPSHOT_MAX_BYTES = 256 << 20  # the clipboard snapshot never holds more than this
CTRL_V = ((winkeys.VK_CONTROL, True), (winkeys.VK_V, True), (winkeys.VK_V, False), (winkeys.VK_CONTROL, False))

WM_NULL, WM_RENDERFORMAT, WM_RENDERALLFORMATS, WM_DESTROYCLIPBOARD = 0x0000, 0x0305, 0x0306, 0x0307
PM_REMOVE = 0x0001
QS_ALLINPUT = 0x04FF
SMTO_ABORTIFHUNG = 0x0002
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_INTEGRITY_LEVEL = 25
HWND_MESSAGE = wintypes.HWND(-3)

LRESULT = ctypes.c_ssize_t
WPARAM = ctypes.c_size_t
LPARAM = ctypes.c_ssize_t

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)


def _fn(dll, name: str, restype, *argtypes):
    f = getattr(dll, name)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f


WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, WPARAM, LPARAM)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HICON)]


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", SID_AND_ATTRIBUTES)]


_OpenClipboard = _fn(_user32, "OpenClipboard", wintypes.BOOL, wintypes.HWND)
_CloseClipboard = _fn(_user32, "CloseClipboard", wintypes.BOOL)
_EmptyClipboard = _fn(_user32, "EmptyClipboard", wintypes.BOOL)
_IsClipboardFormatAvailable = _fn(_user32, "IsClipboardFormatAvailable", wintypes.BOOL, wintypes.UINT)
_GetClipboardData = _fn(_user32, "GetClipboardData", wintypes.HANDLE, wintypes.UINT)
_SetClipboardData = _fn(_user32, "SetClipboardData", wintypes.HANDLE, wintypes.UINT, wintypes.HANDLE)
_CountClipboardFormats = _fn(_user32, "CountClipboardFormats", ctypes.c_int)
_EnumClipboardFormats = _fn(_user32, "EnumClipboardFormats", wintypes.UINT, wintypes.UINT)
_RegisterClipboardFormatW = _fn(_user32, "RegisterClipboardFormatW", wintypes.UINT, wintypes.LPCWSTR)
_GetClipboardFormatNameW = _fn(_user32, "GetClipboardFormatNameW", ctypes.c_int,
                               wintypes.UINT, wintypes.LPWSTR, ctypes.c_int)
_GetClipboardOwner = _fn(_user32, "GetClipboardOwner", wintypes.HWND)
_GetOpenClipboardWindow = _fn(_user32, "GetOpenClipboardWindow", wintypes.HWND)
_GetForegroundWindow = _fn(_user32, "GetForegroundWindow", wintypes.HWND)
_GetWindowThreadProcessId = _fn(_user32, "GetWindowThreadProcessId", wintypes.DWORD,
                                wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
_SendMessageTimeoutW = _fn(_user32, "SendMessageTimeoutW", LRESULT, wintypes.HWND, wintypes.UINT, WPARAM, LPARAM,
                           wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t))
_RegisterClassExW = _fn(_user32, "RegisterClassExW", wintypes.ATOM, ctypes.POINTER(WNDCLASSEXW))
_CreateWindowExW = _fn(_user32, "CreateWindowExW", wintypes.HWND, wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
                       wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                       wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p)
_DestroyWindow = _fn(_user32, "DestroyWindow", wintypes.BOOL, wintypes.HWND)
_DefWindowProcW = _fn(_user32, "DefWindowProcW", LRESULT, wintypes.HWND, wintypes.UINT, WPARAM, LPARAM)
_PeekMessageW = _fn(_user32, "PeekMessageW", wintypes.BOOL, ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                    wintypes.UINT, wintypes.UINT, wintypes.UINT)
_TranslateMessage = _fn(_user32, "TranslateMessage", wintypes.BOOL, ctypes.POINTER(wintypes.MSG))
_DispatchMessageW = _fn(_user32, "DispatchMessageW", LRESULT, ctypes.POINTER(wintypes.MSG))
_MsgWaitForMultipleObjects = _fn(_user32, "MsgWaitForMultipleObjects", wintypes.DWORD, wintypes.DWORD,
                                 ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD, wintypes.DWORD)
_GlobalAlloc = _fn(_kernel32, "GlobalAlloc", wintypes.HGLOBAL, wintypes.UINT, ctypes.c_size_t)
_GlobalLock = _fn(_kernel32, "GlobalLock", ctypes.c_void_p, wintypes.HGLOBAL)
_GlobalUnlock = _fn(_kernel32, "GlobalUnlock", wintypes.BOOL, wintypes.HGLOBAL)
_GlobalFree = _fn(_kernel32, "GlobalFree", wintypes.HGLOBAL, wintypes.HGLOBAL)
_GlobalSize = _fn(_kernel32, "GlobalSize", ctypes.c_size_t, wintypes.HGLOBAL)
_GetModuleHandleW = _fn(_kernel32, "GetModuleHandleW", wintypes.HMODULE, wintypes.LPCWSTR)
_OpenProcess = _fn(_kernel32, "OpenProcess", wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_GetCurrentProcess = _fn(_kernel32, "GetCurrentProcess", wintypes.HANDLE)
_CloseHandle = _fn(_kernel32, "CloseHandle", wintypes.BOOL, wintypes.HANDLE)
_OpenProcessToken = _fn(_advapi32, "OpenProcessToken", wintypes.BOOL, wintypes.HANDLE, wintypes.DWORD,
                        ctypes.POINTER(wintypes.HANDLE))
_GetTokenInformation = _fn(_advapi32, "GetTokenInformation", wintypes.BOOL, wintypes.HANDLE, ctypes.c_int,
                           ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
_GetSidSubAuthorityCount = _fn(_advapi32, "GetSidSubAuthorityCount", ctypes.POINTER(ctypes.c_ubyte), ctypes.c_void_p)
_GetSidSubAuthority = _fn(_advapi32, "GetSidSubAuthority", ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
                          wintypes.DWORD)
_GetEnhMetaFileBits = _fn(_gdi32, "GetEnhMetaFileBits", wintypes.UINT, wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p)
_SetEnhMetaFileBits = _fn(_gdi32, "SetEnhMetaFileBits", wintypes.HANDLE, wintypes.UINT, ctypes.c_char_p)
_DeleteEnhMetaFile = _fn(_gdi32, "DeleteEnhMetaFile", wintypes.BOOL, wintypes.HANDLE)

WIN32_FUNCTIONS = (_OpenClipboard, _CloseClipboard, _EmptyClipboard, _IsClipboardFormatAvailable, _GetClipboardData,
                   _SetClipboardData, _CountClipboardFormats, _EnumClipboardFormats, _RegisterClipboardFormatW,
                   _GetClipboardFormatNameW, _GetClipboardOwner, _GetOpenClipboardWindow, _GetForegroundWindow,
                   _GetWindowThreadProcessId, _SendMessageTimeoutW, _RegisterClassExW, _CreateWindowExW,
                   _DestroyWindow, _DefWindowProcW, _PeekMessageW, _TranslateMessage, _DispatchMessageW,
                   _MsgWaitForMultipleObjects, _GlobalAlloc, _GlobalLock, _GlobalUnlock, _GlobalFree, _GlobalSize,
                   _GetModuleHandleW, _OpenProcess, _GetCurrentProcess, _CloseHandle, _OpenProcessToken,
                   _GetTokenInformation, _GetSidSubAuthorityCount, _GetSidSubAuthority, _GetEnhMetaFileBits,
                   _SetEnhMetaFileBits, _DeleteEnhMetaFile)

CF_CLOUD = _RegisterClipboardFormatW("CanUploadToCloudClipboard")
CF_HISTORY = _RegisterClipboardFormatW("CanIncludeInClipboardHistory")
CF_NO_MONITOR = _RegisterClipboardFormatW("ExcludeClipboardContentFromMonitorProcessing")
CF_OLE_PRIVATE = _RegisterClipboardFormatW("Ole Private Data")
_DWORD_ZERO = (0).to_bytes(4, "little")

_PASTE_LOCK = threading.Lock()


class ClipboardBusy(OSError):
    """Another app held the clipboard open through every retry."""


@contextlib.contextmanager
def _clipboard(owner=None):
    """OpenClipboard with up to OPEN_TRIES attempts, OPEN_SLEEP_S apart; always closed on exit.
    With `owner` (a window handle) that window becomes the clipboard owner on EmptyClipboard."""
    for attempt in range(OPEN_TRIES):
        if _OpenClipboard(owner):
            break
        if attempt < OPEN_TRIES - 1:
            time.sleep(OPEN_SLEEP_S)
    else:
        raise ClipboardBusy(f"OpenClipboard failed {OPEN_TRIES} times (error {ctypes.get_last_error()})")
    try:
        yield
    finally:
        _CloseClipboard()


def _read_text_open() -> str | None:
    """CF_UNICODETEXT from the open clipboard, or None if it holds no text."""
    if not _IsClipboardFormatAvailable(CF_UNICODETEXT):
        return None
    handle = _GetClipboardData(CF_UNICODETEXT)
    if not handle:
        return None
    ptr = _GlobalLock(handle)
    if not ptr:
        return None
    try:
        raw = ctypes.wstring_at(ptr, _GlobalSize(handle) // ctypes.sizeof(ctypes.c_wchar))
    finally:
        _GlobalUnlock(handle)
    return raw.split("\0", 1)[0]


def _global(data: bytes):
    """A movable global block holding `data`, or None."""
    handle = _GlobalAlloc(GMEM_MOVEABLE, max(len(data), 1))
    if not handle:
        return None
    ptr = _GlobalLock(handle)
    if not ptr:
        _GlobalFree(handle)
        return None
    ctypes.memmove(ptr, data, len(data))
    _GlobalUnlock(handle)
    return handle


def _set_global(fmt: int, data: bytes) -> bool:
    """Copy bytes into a movable global block and hand it to the open clipboard (which then owns it)."""
    handle = _global(data)
    if not handle:
        return False
    if not _SetClipboardData(fmt, handle):
        log.warning("SetClipboardData(%d) failed (error %d)", fmt, ctypes.get_last_error())
        _GlobalFree(handle)
        return False
    return True


def _flags_open(history: bool, monitors: bool = True) -> None:
    """The privacy markers: never to the cloud clipboard; out of Win+V history (and monitors) when asked."""
    _set_global(CF_CLOUD, _DWORD_ZERO)
    if not history:
        _set_global(CF_HISTORY, _DWORD_ZERO)
    if not monitors:
        _set_global(CF_NO_MONITOR, _DWORD_ZERO)


def _write_text_open(text: str, history: bool) -> bool:
    """Replace the open clipboard's contents with `text`; kept out of the cloud clipboard, and out of history if asked."""
    if not _EmptyClipboard():
        return False
    if not _set_global(CF_UNICODETEXT, (text + "\0").encode("utf-16-le")):
        return False
    _flags_open(history)
    return True


def get_clipboard_text() -> str | None:
    """The clipboard's text, or None if it holds none. Raises ClipboardBusy."""
    with _clipboard():
        return _read_text_open()


def set_clipboard_text(text: str, history: bool = True) -> bool:
    """Put `text` on the clipboard as CF_UNICODETEXT. history=False keeps it out of Win+V history. Raises ClipboardBusy."""
    with _clipboard():
        return _write_text_open(text.replace("\0", ""), history)


def clipboard_formats() -> list[int]:
    """Format ids currently on the clipboard. Raises ClipboardBusy."""
    with _clipboard():
        return _formats_open()


def _formats_open() -> list[int]:
    out: list[int] = []
    fmt = _EnumClipboardFormats(0)
    while fmt:
        out.append(fmt)
        fmt = _EnumClipboardFormats(fmt)
    return out


def format_name(fmt: int) -> str:
    """Registered name of a clipboard format, or its number for the predefined ones."""
    buf = ctypes.create_unicode_buffer(256)
    n = _GetClipboardFormatNameW(fmt, buf, len(buf))
    return buf.value[:n] if n else str(fmt)


# ---------------------------------------------------------------- the user's clipboard, every format


@dataclass
class Snapshot:
    """Every format the clipboard held, copied out in its own order (apps pick the first they understand).
    `skipped` lists formats that could not be copied, so the restore is partial."""

    items: list[tuple[int, str, bytes]] = field(default_factory=list)   # (format, "global" | "emf", bytes)
    skipped: list[int] = field(default_factory=list)

    def text(self) -> str | None:
        for fmt, kind, data in self.items:
            if fmt == CF_UNICODETEXT and kind == "global":
                return data.decode("utf-16-le", errors="replace").split("\0", 1)[0]
        return None


def _snapshot_open() -> Snapshot | None:
    """Copy every format off the open clipboard; None when it is empty. Formats Windows synthesises from
    another one (CF_TEXT from CF_UNICODETEXT, CF_BITMAP and CF_PALETTE from CF_DIB, CF_METAFILEPICT from
    CF_ENHMETAFILE) are skipped: they come back by themselves. GDI and private handles cannot be copied."""
    formats = _formats_open()
    if not formats:
        return None
    present = set(formats)
    snap = Snapshot()
    total = 0
    for fmt in formats:
        if fmt in (CF_CLOUD, CF_HISTORY, CF_OLE_PRIVATE):
            continue                                   # rewritten by Scribe; OLE's own bookkeeping
        if fmt in (CF_TEXT, CF_OEMTEXT) and CF_UNICODETEXT in present:
            continue
        if fmt in (CF_BITMAP, CF_PALETTE) and present & {CF_DIB, CF_DIBV5}:
            continue
        if fmt == CF_METAFILEPICT and CF_ENHMETAFILE in present:
            continue
        handle = _GetClipboardData(fmt)
        if not handle:
            snap.skipped.append(fmt)
            continue
        if fmt == CF_ENHMETAFILE:
            n = _GetEnhMetaFileBits(handle, 0, None)
            buf = ctypes.create_string_buffer(n) if n else None
            if not n or total + n > SNAPSHOT_MAX_BYTES or _GetEnhMetaFileBits(handle, n, buf) != n:
                snap.skipped.append(fmt)
                continue
            snap.items.append((fmt, "emf", buf.raw))
            total += n
            continue
        if fmt in (CF_BITMAP, CF_PALETTE, CF_METAFILEPICT) or CF_OWNERDISPLAY <= fmt <= CF_DSPENHMETAFILE \
                or CF_PRIVATEFIRST <= fmt <= CF_GDIOBJLAST:
            snap.skipped.append(fmt)
            continue
        size = _GlobalSize(handle)
        if not size or total + size > SNAPSHOT_MAX_BYTES:
            snap.skipped.append(fmt)
            continue
        ptr = _GlobalLock(handle)
        if not ptr:
            snap.skipped.append(fmt)
            continue
        try:
            data = ctypes.string_at(ptr, size)
        finally:
            _GlobalUnlock(handle)
        snap.items.append((fmt, "global", data))
        total += size
    return snap


def _write_snapshot_open(snap: Snapshot) -> bool:
    """Put a snapshot back on the open clipboard, kept out of the cloud clipboard and of Win+V history."""
    if not _EmptyClipboard():
        return False
    ok = True
    for fmt, kind, data in snap.items:
        if kind == "emf":
            handle = _SetEnhMetaFileBits(len(data), data)
            if not handle or not _SetClipboardData(fmt, handle):
                if handle:
                    _DeleteEnhMetaFile(handle)
                ok = False
        elif not _set_global(fmt, data):
            ok = False
    _flags_open(history=False)
    return ok


# ---------------------------------------------------------------- the focused window


@dataclass
class Target:
    """The window Ctrl+V will reach. ok=False: Scribe must not send it there."""

    hwnd: int | None
    pid: int | None
    ok: bool
    reason: str = ""


def _integrity(process) -> int | None:
    """Mandatory integrity RID of a process handle's token (0x2000 medium, 0x3000 high), or None if denied."""
    token = wintypes.HANDLE()
    if not _OpenProcessToken(process, TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        need = wintypes.DWORD()
        _GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, None, 0, ctypes.byref(need))
        if not need.value:
            return None
        buf = ctypes.create_string_buffer(need.value)
        if not _GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, buf, need, ctypes.byref(need)):
            return None
        sid = TOKEN_MANDATORY_LABEL.from_buffer(buf).Label.Sid
        count = _GetSidSubAuthorityCount(sid).contents.value
        if not count:
            return None
        return int(_GetSidSubAuthority(sid, count - 1).contents.value)
    finally:
        _CloseHandle(token)


def process_integrity(pid: int | None = None) -> int | None:
    """Integrity level of a process (this one when pid is None); None when Windows will not say."""
    if pid is None or pid == os.getpid():
        return _integrity(_GetCurrentProcess())
    handle = _OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        return _integrity(handle)
    finally:
        _CloseHandle(handle)


def _window_pid(hwnd) -> int | None:
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    _GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value or None


def _foreground_target() -> Target:
    """Where Ctrl+V would land. UIPI drops injected input into a higher-integrity window without telling
    SendInput, so an elevated (or unreadable) target is refused up front."""
    hwnd = _GetForegroundWindow()
    if not hwnd:
        return Target(None, None, False, "no foreground window")
    pid = _window_pid(hwnd)
    if pid is None:
        return Target(hwnd, None, False, "foreground process unknown")
    if pid == os.getpid():
        return Target(hwnd, pid, True)
    theirs, ours = process_integrity(pid), process_integrity()
    if theirs is None:
        return Target(hwnd, pid, False, "foreground token unreadable")
    if ours is not None and theirs > ours:
        return Target(hwnd, pid, False, "foreground runs elevated")
    return Target(hwnd, pid, True)


def _settle(hwnd) -> bool:
    """Wait until the target's UI thread pumps messages again (it then reaches the queued Ctrl+V), plus a
    margin for it to handle the paste. False when the target is hung."""
    if hwnd:
        result = ctypes.c_size_t()
        if not _SendMessageTimeoutW(hwnd, WM_NULL, 0, 0, SMTO_ABORTIFHUNG, QUEUE_WAIT_MS, ctypes.byref(result)):
            return False
    time.sleep(FALLBACK_SETTLE_S)
    return True


# ---------------------------------------------------------------- delayed rendering

_OWNERS: dict[int, "_Owner"] = {}
_CLASS_NAME = "ScribeClipboardOwner"
_class_lock = threading.Lock()
_class_atom = 0


def _wndproc(hwnd, msg, wparam, lparam):
    owner = _OWNERS.get(hwnd or 0)
    try:
        if owner is not None and msg == WM_RENDERFORMAT:
            owner.render(int(wparam))
            return 0
        if owner is not None and msg == WM_RENDERALLFORMATS:
            owner.render_all()
            return 0
        if owner is not None and msg == WM_DESTROYCLIPBOARD:
            owner.lost = True              # someone else emptied the clipboard: our text is gone
    except Exception:  # never let an exception escape into user32
        log.exception("clipboard render failed")
        return 0
    return _DefWindowProcW(hwnd, msg, wparam, lparam)


_WNDPROC = WNDPROC(_wndproc)  # module-level: the window class calls it for the life of the process


def _register_class() -> bool:
    global _class_atom
    with _class_lock:
        if _class_atom:
            return True
        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.lpfnWndProc = _WNDPROC
        wc.hInstance = _GetModuleHandleW(None)
        wc.lpszClassName = _CLASS_NAME
        _class_atom = _RegisterClassExW(ctypes.byref(wc))
        if not _class_atom:
            log.warning("clipboard owner class not registered (error %d)", ctypes.get_last_error())
        return bool(_class_atom)


class _Owner:
    """A hidden message-only window (never shown, HWND_MESSAGE parent) that offers the dictation by delayed
    rendering, so Scribe learns when, and by whom, the text is read."""

    def __init__(self, text: str, target_pid: int | None):
        self.data = (text + "\0").encode("utf-16-le")
        self.target_pid = target_pid
        self.armed = False             # set just before Ctrl+V is injected
        self.renders = 0
        self.target_read = False
        self.lost = False
        self.hwnd = None
        if _register_class():
            self.hwnd = _CreateWindowExW(0, _CLASS_NAME, None, 0, 0, 0, 0, 0, HWND_MESSAGE, None,
                                         _GetModuleHandleW(None), None) or None
            if self.hwnd:
                _OWNERS[self.hwnd] = self

    def render(self, fmt: int) -> None:
        """WM_RENDERFORMAT: hand over the text; note whether the reader is the paste target."""
        if fmt != CF_UNICODETEXT:
            return
        handle = _global(self.data)
        if not handle or not _SetClipboardData(CF_UNICODETEXT, handle):
            if handle:
                _GlobalFree(handle)
            log.warning("delayed render failed (error %d)", ctypes.get_last_error())
            return
        self.renders += 1
        reader = _window_pid(_GetOpenClipboardWindow())
        # A reader that opened the clipboard without a window cannot be named; after Ctrl+V it is
        # almost always the target (monitors are told to stay away by CF_NO_MONITOR).
        if self.armed and (reader is None or reader == self.target_pid):
            self.target_read = True
        log.debug("dictation rendered for pid %s (target %s)", reader, self.target_pid)

    def render_all(self) -> None:
        """WM_RENDERALLFORMATS (the window is being destroyed while still owning unrendered text)."""
        if not _OpenClipboard(self.hwnd):
            return
        try:
            if _GetClipboardOwner() == self.hwnd and self.renders == 0:
                self.render(CF_UNICODETEXT)
        finally:
            _CloseClipboard()

    def pump(self, until: Callable[[], bool], timeout: float) -> bool:
        """Process this thread's messages (the renders arrive as sent messages) until `until()` or timeout."""
        deadline = time.monotonic() + timeout
        msg = wintypes.MSG()
        while True:
            while _PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                _TranslateMessage(ctypes.byref(msg))
                _DispatchMessageW(ctypes.byref(msg))
            if until():
                return True
            left = deadline - time.monotonic()
            if left <= 0:
                return until()
            _MsgWaitForMultipleObjects(0, None, False, max(1, min(50, int(left * 1000))), QS_ALLINPUT)

    def close(self) -> None:
        """Destroy the window; Windows asks it to render anything still pending first."""
        if self.hwnd:
            _DestroyWindow(self.hwnd)
            _OWNERS.pop(self.hwnd, None)
            self.hwnd = None


# ---------------------------------------------------------------- paste


def _send_paste() -> bool:
    return winkeys.send_keys(CTRL_V)


def _restore(previous: Snapshot, ours: str) -> None:
    """Put the user's clipboard back, but only if nobody replaced our text in the meantime."""
    try:
        with _clipboard():
            if _read_text_open() != ours:
                log.info("clipboard changed during paste; not restored")
                return
            if not _write_snapshot_open(previous):
                log.warning("clipboard restore incomplete")
    except ClipboardBusy as exc:
        log.warning("clipboard restore skipped: %s", exc)
        return
    if previous.skipped:
        log.info("clipboard restored without %d format(s): %s", len(previous.skipped),
                 ", ".join(format_name(f) for f in previous.skipped[:8]))


def _leave(text: str, only_if_ours: bool = True) -> None:
    """Leave the dictation on the clipboard as an ordinary entry (Win+V history included) for a manual paste.
    only_if_ours: once Scribe has put the text there, a copy the user made since is never overwritten."""
    try:
        with _clipboard():
            if only_if_ours and _read_text_open() != text:
                return                                  # the user copied something else meanwhile
            _write_text_open(text, history=True)
    except ClipboardBusy as exc:
        log.error("clipboard busy: %s", exc)


def _wait_for_keys(hold_active: Callable[[], bool] | None) -> bool:
    """Wait out a hold Scribe itself is running (no 2 s limit), then for every modifier to be released."""
    if hold_active is not None:
        deadline = time.monotonic() + HOLD_WAIT_S
        while time.monotonic() < deadline:
            try:
                if not hold_active():
                    break
            except Exception:
                break
            time.sleep(0.02)
    return winkeys.wait_modifiers_released()


def paste_text(text: str, hold_active: Callable[[], bool] | None = None) -> bool:
    """Type `text` into the focused app via the clipboard and Ctrl+V. True once Ctrl+V was delivered and the
    target took the text (or, when its read cannot be seen, answered its message queue).

    `hold_active` (the app's "is a dictation hold down?") is waited out without a limit, so a second hold
    started while this clip was transcribing never times the paste out. Any other modifier still held after
    2 s would turn Ctrl+V into another shortcut, so the text is left on the clipboard and False returned.
    The same happens when the target window runs elevated (Ctrl+V would be dropped silently) or SendInput
    is refused (lock screen, UAC), or the target never reads the text within RENDER_WAIT_S: the user
    can paste by hand, and False makes the app show its error mark.
    """
    text = text.replace("\0", "") if text else ""
    if not text:
        return False
    with _PASTE_LOCK:
        if not _wait_for_keys(hold_active):
            log.warning("modifiers still held; text left on the clipboard, no Ctrl+V sent")
            _leave(text, only_if_ours=False)
            return False
        target = _foreground_target()
        if not target.ok:
            log.warning("no paste: %s; text left on the clipboard", target.reason)
            _leave(text, only_if_ours=False)
            return False
        owner = _Owner(text, target.pid)
        try:
            return _paste_with(owner, text, target)
        finally:
            owner.close()


def _paste_with(owner: _Owner, text: str, target: Target) -> bool:
    delayed = False
    try:
        with _clipboard(owner.hwnd):
            previous = _snapshot_open()
            if previous is None:
                log.debug("clipboard empty; nothing to restore")
            elif owner.hwnd and _EmptyClipboard():
                # Offer the text without data: Windows asks this window for it when someone reads it.
                _SetClipboardData(CF_UNICODETEXT, None)
                _flags_open(history=False, monitors=False)
                delayed = True
            if not delayed and not _write_text_open(text, history=previous is None):
                log.error("could not put text on the clipboard (error %d)", ctypes.get_last_error())
                return False
    except ClipboardBusy as exc:
        log.error("clipboard busy: %s", exc)
        return False
    owner.armed = True
    sent = _send_paste()
    if not sent:
        log.warning("Ctrl+V not delivered; text left on the clipboard")
        owner.close()
        _leave(text)
        return False
    if previous is None:
        return True
    if delayed:
        owner.pump(lambda: owner.renders > 0 or owner.lost, RENDER_WAIT_S)
        if owner.lost and owner.renders == 0:
            log.info("clipboard replaced before the text was read; nothing restored")
            return True
        if owner.target_read:
            owner.pump(lambda: False, PASTE_SETTLE_S)
        elif owner.renders == 0:
            log.warning("target never read the text in %.1f s; left on the clipboard", RENDER_WAIT_S)
            owner.close()
            _leave(text)
            return False
        elif not _settle(target.hwnd):
            log.warning("target not responding; text left on the clipboard")
            owner.close()
            _leave(text)
            return False
    elif not _settle(target.hwnd):
        log.warning("target not responding; text left on the clipboard")
        _leave(text)
        return False
    _restore(previous, text)
    return True
