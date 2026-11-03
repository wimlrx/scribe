"""Hold Ctrl+Win to dictate: a low-level keyboard hook, the chord state machine, and SendInput.

The decisions (press, release, cancel, mask) live in HoldChord, a pure class fed (vk, is_down, injected)
events, so they can be tested without a hook. HoldHotkey wires it to WH_KEYBOARD_LL on its own thread
and hands callbacks to a worker thread so the hook procedure returns in microseconds; Windows silently
removes a low-level hook that is slow to return.

Hotkey is the other kind of shortcut: a press-once combination such as Ctrl+Alt+N (the notetaker), held by
RegisterHotKey on its own thread. parse_hotkey refuses any combination with both Ctrl and Win, so a Hotkey
can never start a dictation.
"""
from __future__ import annotations

import ctypes
import logging
import queue
import threading
import time
from ctypes import wintypes
from typing import Callable, Iterable

log = logging.getLogger("scribe.winkeys")

# --- Win32 types and constants ------------------------------------------------------------------

LRESULT = ctypes.c_ssize_t
WPARAM = ctypes.c_size_t
LPARAM = ctypes.c_ssize_t
ULONG_PTR = ctypes.c_size_t

WH_KEYBOARD_LL = 13
HC_ACTION = 0
WM_QUIT = 0x0012
WM_USER = 0x0400
PM_NOREMOVE = 0x0000
LLKHF_EXTENDED = 0x01
LLKHF_INJECTED = 0x10
LLKHF_UP = 0x80
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
MAPVK_VK_TO_VSC = 0

VK_SHIFT, VK_CONTROL, VK_MENU = 0x10, 0x11, 0x12
VK_LWIN, VK_RWIN = 0x5B, 0x5C
VK_LSHIFT, VK_RSHIFT = 0xA0, 0xA1
VK_LCONTROL, VK_RCONTROL = 0xA2, 0xA3
VK_LMENU, VK_RMENU = 0xA4, 0xA5
VK_MASK = 0xE8          # unassigned: injected between Win down and Win up so the Start menu stays shut
VK_V = 0x56

CTRL_KEYS = frozenset({VK_CONTROL, VK_LCONTROL, VK_RCONTROL})
WIN_KEYS = frozenset({VK_LWIN, VK_RWIN})
CHORD_KEYS = CTRL_KEYS | WIN_KEYS
MODIFIER_KEYS = (VK_LCONTROL, VK_RCONTROL, VK_LWIN, VK_RWIN, VK_LSHIFT, VK_RSHIFT, VK_LMENU, VK_RMENU)
_EXTENDED_VKS = frozenset({VK_RCONTROL, VK_RMENU, VK_LWIN, VK_RWIN, 0x21, 0x22, 0x23, 0x24, 0x25,
                           0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5D, 0x6F, 0x90})

# Every event Scribe injects carries this tag; the hook never feeds its own events to the chord.
SCRIBE_TAG = 0x53435242  # "SCRB"


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    # MOUSEINPUT is the largest member; without it the union is too small and SendInput rejects cbSize.
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, WPARAM, LPARAM)

# Private DLL instances so argtypes set here never collide with another module's use of windll.
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def _fn(dll, name: str, restype, *argtypes):
    f = getattr(dll, name)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f


_SetWindowsHookExW = _fn(_user32, "SetWindowsHookExW", wintypes.HHOOK,
                         ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
_UnhookWindowsHookEx = _fn(_user32, "UnhookWindowsHookEx", wintypes.BOOL, wintypes.HHOOK)
_CallNextHookEx = _fn(_user32, "CallNextHookEx", LRESULT, wintypes.HHOOK, ctypes.c_int, WPARAM, LPARAM)
_GetMessageW = _fn(_user32, "GetMessageW", wintypes.BOOL,
                   ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
_PeekMessageW = _fn(_user32, "PeekMessageW", wintypes.BOOL, ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                    wintypes.UINT, wintypes.UINT, wintypes.UINT)
_TranslateMessage = _fn(_user32, "TranslateMessage", wintypes.BOOL, ctypes.POINTER(wintypes.MSG))
_DispatchMessageW = _fn(_user32, "DispatchMessageW", LRESULT, ctypes.POINTER(wintypes.MSG))
_PostThreadMessageW = _fn(_user32, "PostThreadMessageW", wintypes.BOOL,
                          wintypes.DWORD, wintypes.UINT, WPARAM, LPARAM)
_GetAsyncKeyState = _fn(_user32, "GetAsyncKeyState", ctypes.c_short, ctypes.c_int)
_SendInput = _fn(_user32, "SendInput", wintypes.UINT, wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
_MapVirtualKeyW = _fn(_user32, "MapVirtualKeyW", wintypes.UINT, wintypes.UINT, wintypes.UINT)
_GetCurrentThreadId = _fn(_kernel32, "GetCurrentThreadId", wintypes.DWORD)
_GetModuleHandleW = _fn(_kernel32, "GetModuleHandleW", wintypes.HMODULE, wintypes.LPCWSTR)

WIN32_FUNCTIONS = (_SetWindowsHookExW, _UnhookWindowsHookEx, _CallNextHookEx, _GetMessageW, _PeekMessageW,
                   _TranslateMessage, _DispatchMessageW, _PostThreadMessageW, _GetAsyncKeyState, _SendInput,
                   _MapVirtualKeyW, _GetCurrentThreadId, _GetModuleHandleW)

# Live hook procedures, keyed by HoldHotkey id: ctypes callbacks must outlive the hook or Windows calls freed code.
_LIVE_PROCS: dict[int, object] = {}

# --- Key state and injection ---------------------------------------------------------------------


def _key_down(vk: int) -> bool:
    """True if the key is down right now (GetAsyncKeyState high bit)."""
    return bool(_GetAsyncKeyState(vk) & 0x8000)


def modifiers_down() -> bool:
    """Any of Ctrl/Win/Shift/Alt down. Paste waits for this to clear so Ctrl+V is not read as Ctrl+Win+V."""
    return any(_key_down(vk) for vk in MODIFIER_KEYS)


def wait_modifiers_released(timeout: float = 2.0) -> bool:
    """Poll every 10 ms until no modifier is down. False if one is still held after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while modifiers_down():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def build_inputs(events: Iterable[tuple[int, bool]]) -> ctypes.Array:
    """INPUT array for (vk, is_down) events, tagged SCRIBE_TAG, with real scan codes for apps that read them."""
    events = list(events)
    arr = (INPUT * len(events))()
    for i, (vk, is_down) in enumerate(events):
        flags = 0 if is_down else KEYEVENTF_KEYUP
        if vk in _EXTENDED_VKS:
            flags |= KEYEVENTF_EXTENDEDKEY
        arr[i].type = INPUT_KEYBOARD
        arr[i].ki = KEYBDINPUT(wVk=vk, wScan=_MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) & 0xFFFF, dwFlags=flags,
                               time=0, dwExtraInfo=SCRIBE_TAG)
    return arr


def send_keys(events: Iterable[tuple[int, bool]]) -> bool:
    """Inject key events in one SendInput call. True if every event was accepted."""
    arr = build_inputs(events)
    if not len(arr):
        return True
    sent = _SendInput(len(arr), arr, ctypes.sizeof(INPUT))
    if sent != len(arr):
        log.warning("SendInput accepted %d of %d events (error %d)", sent, len(arr), ctypes.get_last_error())
    return sent == len(arr)


def send_mask() -> bool:
    """Mask key down+up: makes Windows see a key between Win down and Win up, so the Start menu stays shut."""
    return send_keys(((VK_MASK, True), (VK_MASK, False)))


# --- The chord state machine ----------------------------------------------------------------------

PRESS, RELEASE, CANCEL, MASK = "press", "release", "cancel", "mask"
IDLE, ARMED, HELD, SPENT = "idle", "armed", "held", "spent"


class HoldChord:
    """Pure Ctrl+Win hold logic. feed() takes one key event and returns the actions it triggers, in order.

    press   both a Ctrl and a Win key are down, and no other key was down from the first of them on
    release a Ctrl or Win family empties after press
    cancel  any other key goes down after press (Ctrl+Win+D etc. still reach Windows; dictation is dropped)
    mask    inject VK_MASK now: at press, at cancel, and before every Win key-up that follows them

    `physical(vk)` (GetAsyncKeyState in the real hook) prunes keys whose key-up was never seen, e.g. after
    Win+L or Ctrl+Alt+Del, when the up events land on the secure desktop and the hook misses them.
    """

    def __init__(self, accept_injected: bool = False, physical: Callable[[int], bool] | None = None):
        self.accept_injected = accept_injected
        self.physical = physical
        self.down: set[int] = set()
        self.state = IDLE
        self.clean = False
        self.mask_pending = False
        self.resyncs = 0

    def feed(self, vk: int, is_down: bool, injected: bool = False) -> tuple[str, ...]:
        if vk == VK_MASK or (injected and not self.accept_injected):
            return ()
        if not is_down:
            return tuple(self._up(vk))
        actions = self._resync(vk) if self.physical is not None else []
        actions += self._down(vk)
        return tuple(actions)

    # A Ctrl and a Win key both down.
    def _chord_complete(self) -> bool:
        return bool(self.down & CTRL_KEYS) and bool(self.down & WIN_KEYS)

    def _down(self, vk: int) -> list[str]:
        if vk in self.down:
            return []  # autorepeat
        self.down.add(vk)
        chord = vk in CHORD_KEYS
        if self.state == IDLE:
            if chord:
                self.state = ARMED
                self.clean = not (self.down - CHORD_KEYS)
                return self._maybe_press()
            return []
        if self.state == ARMED:
            if not chord:
                self.clean = False
                return []
            return self._maybe_press()
        if self.state == HELD and not chord:
            self.state = SPENT
            self.mask_pending = True
            return [CANCEL, MASK]
        return []

    def _maybe_press(self) -> list[str]:
        if self.clean and self._chord_complete():
            self.state = HELD
            self.mask_pending = True
            return [PRESS, MASK]
        return []

    def _up(self, vk: int) -> list[str]:
        self.down.discard(vk)
        out: list[str] = []
        if self.state == HELD and vk in CHORD_KEYS and not self._chord_complete():
            self.state = SPENT
            out.append(RELEASE)
        if vk in WIN_KEYS and self.mask_pending:
            out.append(MASK)
            if not self.down & WIN_KEYS:
                self.mask_pending = False
        if self.state != IDLE and not self.down & CHORD_KEYS:
            self.state = IDLE
        return out

    def watch(self) -> tuple[str, ...]:
        """Off-hook check while HELD: a key-up the hook never saw (released over an elevated window, or a
        switch to the secure desktop, where GetAsyncKeyState reads 0) is noticed here. When a whole Ctrl or
        Win family reads up, the hold ends exactly as _up would end it."""
        if self.state != HELD or self.physical is None:
            return ()
        gone = {k for k in self.down & CHORD_KEYS if not self.physical(k)}
        if not gone:
            return ()
        self.down -= gone
        out: list[str] = []
        if not self._chord_complete():
            self.state = SPENT
            out.append(RELEASE)
        if not self.down & WIN_KEYS:
            self.mask_pending = False      # the Win key is already up: there is no key-up left to mask
        if not self.down & CHORD_KEYS:
            self.state = IDLE
        return tuple(out)

    def _resync(self, vk: int) -> list[str]:
        stale = {k for k in self.down if k != vk and not self.physical(k)}
        if not stale:
            return []
        self.resyncs += 1
        out: list[str] = []
        self.down -= stale
        # A key believed held alongside stale keys belongs to the same lost epoch: treat this event as fresh.
        self.down.discard(vk)
        cancelled = self.state == HELD and bool(stale & CHORD_KEYS)
        if cancelled:
            out = [CANCEL, MASK]
        if not self.down & CHORD_KEYS:
            self.state = IDLE
        elif cancelled:
            self.state = SPENT
        elif self.state != HELD:
            self.state = ARMED
            self.clean = not (self.down - CHORD_KEYS)
        self.mask_pending = (cancelled or self.mask_pending) and bool(self.down & WIN_KEYS)
        return out


# --- The real hook ---------------------------------------------------------------------------------


class HoldHotkey:
    """Hold Ctrl+Win anywhere in Windows. Keys are never swallowed; callbacks run on a worker thread.

    Every start() is a new generation. A hook thread from an earlier generation that only gets to run after
    its start() gave up (the GIL held elsewhere for seconds) sees the generation has moved on and exits
    without installing a hook, so no orphan hook ever feeds a dead worker or takes over the thread id."""

    WATCH_S = 0.1          # how often the worker checks a held chord against the physical keys

    def __init__(self, on_press: Callable[[], None], on_release: Callable[[], None],
                 on_cancel: Callable[[], None], accept_injected: bool = False):
        self.on_press, self.on_release, self.on_cancel = on_press, on_release, on_cancel
        self.accept_injected = accept_injected
        self._chord = HoldChord(accept_injected, physical=_key_down)
        self._chord_lock = threading.Lock()   # the hook thread and the watchdog both move the chord
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._thread_id = 0
        self._hhook = None
        self._ready = threading.Event()
        self._error: OSError | None = None
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._gen = 0
        self._tids: dict[int, int] = {}       # generation -> its hook thread's id, once that thread has run
        self._stale: list[tuple[int, threading.Thread]] = []

    @property
    def running(self) -> bool:
        """True while the hook is installed and its message loop is alive."""
        return bool(self._hhook) and self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Install WH_KEYBOARD_LL on a new thread with its own message loop. Raises OSError if Windows refuses."""
        with self._lock:
            if self._thread is not None:
                return
            self._reap_stale()
            with self._gen_lock:
                self._gen += 1
                gen = self._gen
            self._chord = HoldChord(self.accept_injected, physical=_key_down)
            self._events = queue.SimpleQueue()
            self._ready = threading.Event()
            self._error = None
            self._worker = threading.Thread(target=self._work, args=(self._events,), name="scribe-hotkey-worker",
                                            daemon=True)
            self._worker.start()
            self._thread = threading.Thread(target=self._run, args=(gen, self._ready), name="scribe-hotkey-hook",
                                            daemon=True)
            self._thread.start()
            if not self._ready.wait(5.0):
                self._error = OSError("keyboard hook thread did not start")
            if self._error is not None:
                err = self._error
                self._shutdown()
                raise err
            log.info("hotkey hook installed (Ctrl+Win hold)")

    def stop(self) -> None:
        """Post WM_QUIT to the hook thread, which unhooks before it exits; then drain the worker."""
        with self._lock:
            if self._thread is None:
                return
            self._shutdown()
            log.info("hotkey hook removed")

    def _reap_stale(self) -> None:
        """Ask leftover hook threads of earlier generations to quit, once they have a thread id."""
        alive = []
        for gen, t in self._stale:
            if not t.is_alive():
                self._tids.pop(gen, None)
                continue
            tid = self._tids.get(gen)
            if tid:
                _PostThreadMessageW(tid, WM_QUIT, 0, 0)
            alive.append((gen, t))
        self._stale = alive

    def _shutdown(self) -> None:
        t, w = self._thread, self._worker
        with self._gen_lock:
            gen = self._gen
            self._gen += 1         # a hook thread of this generation that has not hooked yet never will
            tid = self._tids.get(gen) or self._thread_id
        if t is not None and t.is_alive() and tid:
            if not _PostThreadMessageW(tid, WM_QUIT, 0, 0):
                log.warning("PostThreadMessageW failed (error %d)", ctypes.get_last_error())
        if t is not None:
            t.join(2.0)
            if t.is_alive():
                log.warning("hotkey hook thread did not exit; kept as stale")
                self._stale.append((gen, t))
            else:
                self._tids.pop(gen, None)
        self._events.put(None)
        if w is not None:
            w.join(2.0)
        self._thread = self._worker = None
        self._thread_id = 0

    def _run(self, gen: int | None = None, ready: threading.Event | None = None) -> None:
        ready = ready or self._ready
        msg = wintypes.MSG()
        _PeekMessageW(ctypes.byref(msg), None, WM_USER, WM_USER, PM_NOREMOVE)  # creates this thread's queue
        tid = _GetCurrentThreadId()
        with self._gen_lock:
            if gen is None:
                gen = self._gen
            self._tids[gen] = tid
            if gen != self._gen:
                log.info("hotkey hook thread of an old generation exits without hooking")
                return
            self._thread_id = tid
        proc = HOOKPROC(self._proc)
        _LIVE_PROCS[id(self)] = proc
        hhook = _SetWindowsHookExW(WH_KEYBOARD_LL, proc, _GetModuleHandleW(None), 0)
        if not hhook:
            self._error = ctypes.WinError(ctypes.get_last_error())
            if _LIVE_PROCS.get(id(self)) is proc:
                _LIVE_PROCS.pop(id(self), None)
            ready.set()
            return
        with self._gen_lock:
            current = gen == self._gen
            if current:
                self._hhook = hhook
        if not current:                     # stopped while the hook went in: take it straight out again
            _UnhookWindowsHookEx(hhook)
            if _LIVE_PROCS.get(id(self)) is proc:
                _LIVE_PROCS.pop(id(self), None)
            return
        ready.set()
        try:
            while True:
                r = _GetMessageW(ctypes.byref(msg), None, 0, 0)
                if r == 0 or r == -1:
                    break
                _TranslateMessage(ctypes.byref(msg))
                _DispatchMessageW(ctypes.byref(msg))
        finally:
            if not _UnhookWindowsHookEx(hhook):
                log.warning("UnhookWindowsHookEx failed (error %d)", ctypes.get_last_error())
            with self._gen_lock:
                if self._hhook == hhook:
                    self._hhook = None
            if _LIVE_PROCS.get(id(self)) is proc:   # never drop a newer generation's procedure
                _LIVE_PROCS.pop(id(self), None)

    def _proc(self, n_code: int, w_param: int, l_param: int) -> int:
        """The hook procedure: decide, inject the mask if due, queue callbacks, always pass the key on."""
        try:
            if n_code == HC_ACTION and l_param:
                kb = KBDLLHOOKSTRUCT.from_address(l_param)
                if kb.dwExtraInfo != SCRIBE_TAG:
                    is_down = not (kb.flags & LLKHF_UP)
                    with self._chord_lock:
                        before = self._chord.resyncs
                        actions = self._chord.feed(kb.vkCode, is_down, bool(kb.flags & LLKHF_INJECTED))
                        resynced = self._chord.resyncs != before
                    if resynced:
                        self._events.put("resync")
                    for action in actions:
                        if action == MASK:
                            send_mask()
                        else:
                            self._events.put(action)
        except Exception as exc:  # never let an error stop the key reaching Windows
            self._events.put(("error", exc))
        return _CallNextHookEx(None, n_code, w_param, l_param)

    def _watch(self) -> None:
        """Watchdog tick: end a hold whose key-ups never reached the hook (elevated window, secure desktop)."""
        with self._chord_lock:
            actions = self._chord.watch()
        for action in actions:
            log.info("hotkey %s: key-up seen only by the watchdog", action)
            self._events.put(action)

    def _work(self, events: queue.SimpleQueue | None = None) -> None:
        events = events if events is not None else self._events
        callbacks = {PRESS: lambda: self.on_press(), RELEASE: lambda: self.on_release(),
                     CANCEL: lambda: self.on_cancel()}
        while True:
            try:
                item = events.get(timeout=self.WATCH_S)
            except queue.Empty:
                if events is self._events:
                    try:
                        self._watch()
                    except Exception:
                        log.exception("hotkey watchdog failed")
                continue
            if item is None:
                return
            if isinstance(item, tuple):
                log.error("hotkey hook error: %r", item[1])
                continue
            if item == "resync":
                log.info("hotkey state resynced: a key-up was missed (lock screen or secure desktop)")
                continue
            log.debug("hotkey %s", item)
            try:
                callbacks[item]()
            except Exception:
                log.exception("hotkey %s callback failed", item)


# --- A press-once global shortcut (RegisterHotKey) -------------------------------------------------

WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x0001, 0x0002, 0x0004, 0x0008, 0x4000
ERROR_HOTKEY_ALREADY_REGISTERED = 1409
HOTKEY_ID = 1            # ids are per thread, and every Hotkey has a thread of its own
VK_F1 = 0x70

_RegisterHotKey = _fn(_user32, "RegisterHotKey", wintypes.BOOL,
                      wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT)
_UnregisterHotKey = _fn(_user32, "UnregisterHotKey", wintypes.BOOL, wintypes.HWND, ctypes.c_int)
WIN32_FUNCTIONS += (_RegisterHotKey, _UnregisterHotKey)

_MOD_NAMES = {"ctrl": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN}


def _key_vk(name: str) -> int | None:
    """VK for a letter, a digit or F1-F24 (lower case); None for anything else."""
    if len(name) == 1 and name.isascii() and name.isalnum():
        return ord(name.upper())
    n = name[1:]
    if name[:1] == "f" and n.isascii() and n.isdigit() and not n.startswith("0") and 1 <= int(n) <= 24:
        return VK_F1 + int(n) - 1
    return None


def parse_hotkey(spec: str) -> tuple[int, int]:
    """'ctrl+alt+n' -> (MOD_CONTROL | MOD_ALT, 0x4E). Modifiers ctrl, alt, shift, win; one key: a letter, a digit
    or F1-F24; any case, spaces allowed. ValueError on anything else, and on combinations that would take keys
    from the user: Ctrl with Win (the dictation hold), and a letter, digit or F1-F12 with neither Win nor two
    of Ctrl, Alt and Shift. One of those alone is a typing key or an app's own (Ctrl+V, which Scribe injects to
    paste and a registration would catch, Ctrl+S, Alt+F4, Excel's F2 and Ctrl+1). Win+key is Windows' own:
    RegisterHotKey refuses the ones it holds. F13-F24, which only macro keys send, may stand alone."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"empty shortcut: {spec!r}")
    mods, vk = 0, None
    for part in spec.split("+"):
        name = part.strip().lower()
        bit = _MOD_NAMES.get(name)
        if bit is not None:
            if mods & bit:
                raise ValueError(f"{name} twice in shortcut {spec!r}")
            mods |= bit
            continue
        key = _key_vk(name)
        if key is None:
            raise ValueError(f"unknown key {part.strip()!r} in shortcut {spec!r}")
        if vk is not None:
            raise ValueError(f"more than one key in shortcut {spec!r}")
        vk = key
    if vk is None:
        raise ValueError(f"no key in shortcut {spec!r}")
    if mods & MOD_CONTROL and mods & MOD_WIN:
        raise ValueError(f"shortcut {spec!r} holds Ctrl+Win, the dictation hold")
    if vk < VK_F1 + 12 and not mods & MOD_WIN and mods.bit_count() < 2:
        raise ValueError(f"shortcut {spec!r} would take a key the user types with: give it Win, or two of Ctrl, "
                         f"Alt and Shift")
    return mods, vk


def hotkey_label(modifiers: int, vk: int) -> str:
    """'Ctrl+Alt+N' for (MOD_CONTROL | MOD_ALT, 0x4E): the shortcut as the tray shows it."""
    names = [name.capitalize() for name, bit in _MOD_NAMES.items() if modifiers & bit]
    key = f"F{vk - VK_F1 + 1}" if VK_F1 <= vk < VK_F1 + 24 else chr(vk)
    return "+".join([*names, key])


class _HotkeyRun:
    """One start(): its thread, the thread's id, and whether it registered or was called off."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.thread: threading.Thread | None = None
        self.tid = 0
        self.registered = False
        self.cancelled = False
        self.error = 0


class Hotkey:
    """A global press-once shortcut such as Ctrl+Alt+N, held by RegisterHotKey on its own thread.

    The callback runs on that thread and must return quickly; hand the work on. MOD_NOREPEAT means holding
    the keys fires once. The Ctrl+Win hook sees the same keys, passes every one on, and never fires on them.
    A bad spec raises ValueError here; start() returns False when Windows refuses the combination (another
    program holds it), or while Scribe's own previous thread still holds it because its callback has not
    returned since stop(). Each new reason warns once; a repeat of the same refusal is quiet."""

    READY_S = 5.0
    STOP_S = 2.0

    def __init__(self, spec: str, callback: Callable[[], None]):
        self.spec = spec
        self.modifiers, self.vk = parse_hotkey(spec)
        self.label = hotkey_label(self.modifiers, self.vk)
        self.callback = callback
        self.error = 0                  # Win32 error of the last refused start; 1409: another program holds it
        self._lock = threading.Lock()
        self._run_state: _HotkeyRun | None = None
        self._refusal_why: str | None = None

    @property
    def running(self) -> bool:
        """True while the shortcut is registered, its message loop is alive, and stop() has not called it off."""
        run = self._run_state
        return run is not None and run.registered and not run.cancelled and run.thread.is_alive()

    def start(self) -> bool:
        """Register on a new thread with its own message loop. True once registered or already running."""
        with self._lock:
            if self.running:
                return True
            old = self._run_state
            if old is not None:     # a thread stop() could not wait out, or a loop that ended on its own
                if not self._end(old):
                    self.error = 0
                    self._refused("Scribe's previous thread holds it until its callback returns")
                    return False
                self._run_state = None
            run = _HotkeyRun()
            run.thread = threading.Thread(target=self._run, args=(run,), name="scribe-shortcut", daemon=True)
            run.thread.start()
            started = run.ready.wait(self.READY_S)
            with run.lock:
                registered, error = run.registered, run.error
                if not registered:
                    run.cancelled = True    # a thread that registers late takes it straight out again
            if not registered:
                run.thread.join(1.0)
                self.error = error
                self._refused("its thread did not start" if not started and not error else
                              "another program holds it" if error == ERROR_HOTKEY_ALREADY_REGISTERED else
                              f"error {error}")
                return False
            self._run_state = run
            self.error = 0
            self._refusal_why = None
            log.info("shortcut %s registered", self.label)
            return True

    def stop(self) -> None:
        """Call the shortcut off and post WM_QUIT to its thread, which unregisters before it exits; waits up to
        STOP_S for it. No callback starts after stop() returns. Safe from the callback."""
        with self._lock:
            run = self._run_state
            if run is None:
                return
            if self._end(run):
                self._run_state = None
                log.info("shortcut %s removed", self.label)
            elif run.thread is threading.current_thread():
                log.info("shortcut %s removed once its callback returns", self.label)
            else:               # kept, so start() waits for this thread instead of blaming another program
                log.warning("shortcut %s thread did not exit; it unregisters when its callback returns", self.label)

    def _refused(self, why: str) -> None:
        log.log(logging.DEBUG if why == self._refusal_why else logging.WARNING,
                "shortcut %s not registered: %s", self.label, why)
        self._refusal_why = why

    def _end(self, run: _HotkeyRun) -> bool:
        """Call the run off, post WM_QUIT, and wait up to STOP_S for its thread. True once that thread has exited."""
        with run.lock:
            run.cancelled = True
            tid = run.tid
        if tid and run.thread.is_alive() and not _PostThreadMessageW(tid, WM_QUIT, 0, 0):
            log.warning("PostThreadMessageW failed (error %d)", ctypes.get_last_error())
        if run.thread is not threading.current_thread():    # from the callback: it ends when the callback returns
            run.thread.join(self.STOP_S)
        return not run.thread.is_alive()

    def _run(self, run: _HotkeyRun) -> None:
        msg = wintypes.MSG()
        _PeekMessageW(ctypes.byref(msg), None, WM_USER, WM_USER, PM_NOREMOVE)  # creates this thread's queue
        with run.lock:
            run.tid = _GetCurrentThreadId()
            if run.cancelled:
                return
        ok = bool(_RegisterHotKey(None, HOTKEY_ID, self.modifiers | MOD_NOREPEAT, self.vk))
        error = 0 if ok else ctypes.get_last_error()
        with run.lock:
            late = ok and run.cancelled
            if late:                    # start() gave up while this thread registered
                _UnregisterHotKey(None, HOTKEY_ID)
            else:
                run.registered, run.error = ok, error
        run.ready.set()
        if not ok or late:
            return
        try:
            while True:
                r = _GetMessageW(ctypes.byref(msg), None, 0, 0)
                if r == 0 or r == -1:
                    break
                if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID and not run.cancelled:
                    try:                # a press queued ahead of stop()'s WM_QUIT is dropped
                        self.callback()
                    except Exception:
                        log.exception("shortcut %s callback failed", self.label)
        finally:
            with run.lock:
                if not _UnregisterHotKey(None, HOTKEY_ID):
                    log.warning("UnregisterHotKey failed (error %d)", ctypes.get_last_error())
                run.registered = False
