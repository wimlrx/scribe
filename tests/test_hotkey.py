"""winkeys.Hotkey: the notetaker's press-once shortcut. parse_hotkey, the real RegisterHotKey lifecycle on its
own thread, the refusal path, and its interplay with the Ctrl+Win hold.

No test sends a key. Dispatch is proven by posting WM_HOTKEY to the shortcut's thread with PostThreadMessageW.
Real registrations use a Ctrl+Alt+Shift+F23-style combination, so a running Scribe holding Ctrl+Alt+N is never
disturbed; the real Ctrl+Alt+N is only ever registered through a mock.
"""
from __future__ import annotations

import ctypes
import itertools
import logging
import threading
import time
import unittest
from unittest import mock

from scribe import config, winkeys
from scribe.winkeys import (CANCEL, HOTKEY_ID, MASK, MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, MOD_SHIFT, MOD_WIN, PRESS,
                            RELEASE, VK_CONTROL, VK_LCONTROL, VK_LMENU, VK_LSHIFT, VK_LWIN, VK_MENU, VK_RCONTROL,
                            VK_RMENU, WM_HOTKEY, HoldChord, HoldHotkey, Hotkey, hotkey_label, parse_hotkey)

VK_N = 0x4E
VK_F5 = 0x74
CTRLS = (VK_LCONTROL, VK_RCONTROL, VK_CONTROL)
ALTS = (VK_LMENU, VK_RMENU, VK_MENU)
MOD_VK = {MOD_CONTROL: VK_LCONTROL, MOD_ALT: VK_LMENU, MOD_SHIFT: VK_LSHIFT, MOD_WIN: VK_LWIN}
THREAD = "scribe-shortcut"
CANDIDATES = ("ctrl+alt+shift+f23", "ctrl+alt+shift+f24", "alt+shift+f23", "alt+shift+f24")
REAL_REGISTER = winkeys._RegisterHotKey
SPEC = None


def free(spec: str) -> bool:
    """True if Windows lets this thread register the combination; released again at once."""
    mods, vk = parse_hotkey(spec)
    ok = REAL_REGISTER(None, 99, mods | MOD_NOREPEAT, vk)
    if ok:
        winkeys._UnregisterHotKey(None, 99)
    return bool(ok)


def setUpModule():
    global SPEC
    SPEC = next((s for s in CANDIDATES if free(s)), None)


def orderings(keys):
    """Every ordering of each key's down and up with down before up: 90 orderings for three keys."""
    tokens = [(k, d) for k in keys for d in (True, False)]
    for perm in set(itertools.permutations(tokens)):
        if all(perm.index((k, True)) < perm.index((k, False)) for k in keys):
            yield list(perm)


class TrackPhysical:
    """GetAsyncKeyState as seen inside the hook: the state before the event being processed."""

    def __init__(self):
        self.down: set[int] = set()

    def __call__(self, vk: int) -> bool:
        return vk in self.down

    def after(self, vk: int, is_down: bool) -> None:
        (self.down.add if is_down else self.down.discard)(vk)


def feed(machine: HoldChord, events, phys: TrackPhysical | None = None) -> list[tuple[str, ...]]:
    """Feed (vk, is_down) events; return every non-empty action tuple."""
    out = []
    for vk, down in events:
        acts = machine.feed(vk, down)
        if phys is not None:
            phys.after(vk, down)
        if acts:
            out.append(acts)
    return out


def dictation_still_works(test: unittest.TestCase, m: HoldChord, phys: TrackPhysical | None = None) -> None:
    test.assertEqual(feed(m, [(VK_LCONTROL, True), (VK_LWIN, True)], phys), [(PRESS, MASK)])
    test.assertEqual(feed(m, [(VK_LWIN, False), (VK_LCONTROL, False)], phys), [(RELEASE, MASK)])
    test.assertEqual(m.state, winkeys.IDLE)


class ParseHotkey(unittest.TestCase):
    def test_default_is_ctrl_alt_n(self):
        self.assertEqual(config.DEFAULTS["notetaker_hotkey"], "ctrl+alt+n")
        self.assertEqual(parse_hotkey(config.DEFAULTS["notetaker_hotkey"]), (MOD_CONTROL | MOD_ALT, VK_N))

    def test_valid(self):
        cases = {
            "ctrl+alt+n": (MOD_CONTROL | MOD_ALT, VK_N),
            "Ctrl + Alt + N": (MOD_CONTROL | MOD_ALT, VK_N),
            "  CTRL+ALT+n  ": (MOD_CONTROL | MOD_ALT, VK_N),
            "alt+ctrl+n": (MOD_CONTROL | MOD_ALT, VK_N),
            "ctrl+shift+alt+n": (MOD_CONTROL | MOD_SHIFT | MOD_ALT, VK_N),
            "ctrl+alt+f": (MOD_CONTROL | MOD_ALT, 0x46),     # the letter F, not a function key
            "alt+shift+1": (MOD_ALT | MOD_SHIFT, 0x31),
            "ctrl+shift+0": (MOD_CONTROL | MOD_SHIFT, 0x30),
            "ctrl+alt+v": (MOD_CONTROL | MOD_ALT, 0x56),
            "win+alt+z": (MOD_WIN | MOD_ALT, 0x5A),
            "win+n": (MOD_WIN, VK_N),                         # Win alone: Windows refuses the ones it holds
            "WIN+5": (MOD_WIN, 0x35),
            "win+f5": (MOD_WIN, VK_F5),
            "ctrl+shift+f4": (MOD_CONTROL | MOD_SHIFT, 0x73),
            "ctrl+alt+F12": (MOD_CONTROL | MOD_ALT, 0x7B),
            "f13": (0, 0x7C),
            "F24": (0, 0x87),
            "win+shift+f24": (MOD_WIN | MOD_SHIFT, 0x87),
        }
        for spec, want in cases.items():
            with self.subTest(spec=spec):
                self.assertEqual(parse_hotkey(spec), want)

    def test_invalid(self):
        cases = [
            "", "   ", None, 5, "+",
            "ctrl+alt",                 # no key
            "ctrl+alt+n+m",             # two keys
            "ctrl+ctrl+n", "alt+ALT+n", # a modifier twice
            "ctrl++n", "ctrl+alt+", "+n",
            "control+alt+n", "ctl+n", "windows+n", "lctrl+n",
            "ctrl+alt+space", "ctrl+alt+esc", "ctrl+alt+é", "ctrl+alt+nn", "ctrl+alt+²",
            "ctrl+alt+f0", "ctrl+alt+f25", "ctrl+alt+f01", "ctrl+alt+f1.5", "ctrl+alt+fx",
            "ctrl+win+n", "win+ctrl+f13", "ctrl+alt+shift+win+n",   # Ctrl+Win is the dictation hold
            "n", "1", "shift+n", "shift+5",                         # would take typing keys
            "f5", "f12",                                            # Excel's own keys, bare
            "ctrl+v", "ctrl+c", "ctrl+x", "ctrl+z", "ctrl+s", "ctrl+n", "ctrl+0", "ctrl+1",  # the apps' own
            "alt+n", "alt+f", "alt+1",
            "alt+f4", "ctrl+f4", "shift+f3", "shift+f5", "ctrl+f1", "alt+F12",
        ]
        for spec in cases:
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    parse_hotkey(spec)

    def test_scribes_own_paste_chord_is_refused(self):
        """RegisterHotKey also fires on injected keys: a Ctrl+V shortcut would eat every dictation's paste."""
        self.assertEqual(winkeys.VK_V, ord("V"))
        with self.assertRaises(ValueError):
            parse_hotkey("ctrl+v")
        with self.assertRaises(ValueError):
            Hotkey("Ctrl + V", lambda: None)

    def test_hotkey_rejects_a_bad_spec_at_construction(self):
        with self.assertRaises(ValueError):
            Hotkey("ctrl+win+n", lambda: None)

    def test_label(self):
        self.assertEqual(Hotkey("ctrl+alt+n", lambda: None).label, "Ctrl+Alt+N")
        self.assertEqual(Hotkey("shift + win + f23", lambda: None).label, "Shift+Win+F23")
        self.assertEqual(hotkey_label(*parse_hotkey("alt+shift+ctrl+7")), "Ctrl+Alt+Shift+7")

    def test_functions_declared(self):
        for f in (winkeys._RegisterHotKey, winkeys._UnregisterHotKey):
            self.assertIn(f, winkeys.WIN32_FUNCTIONS)
            self.assertIsNotNone(f.argtypes)
            self.assertIs(f.restype, ctypes.wintypes.BOOL)

    def test_registers_with_norepeat_on_its_own_thread(self):
        seen = []
        hk = Hotkey("ctrl+alt+n", lambda: None)
        with mock.patch.object(winkeys, "_RegisterHotKey",
                               side_effect=lambda *a: seen.append(threading.current_thread()) or True) as reg, \
                mock.patch.object(winkeys, "_UnregisterHotKey", return_value=True) as unreg:
            self.assertTrue(hk.start())
            hk.stop()
        reg.assert_called_once_with(None, HOTKEY_ID, MOD_NOREPEAT | MOD_CONTROL | MOD_ALT, VK_N)
        unreg.assert_called_once_with(None, HOTKEY_ID)
        self.assertEqual(seen[0].name, THREAD)
        self.assertIsNot(seen[0], threading.current_thread())


class ChordInterplay(unittest.TestCase):
    """The notetaker shortcut fed to the Ctrl+Win state machine: it never presses, cancels or masks, and the
    next dictation works."""

    def test_ctrl_alt_n_every_interleaving(self):
        for c, a in itertools.product(CTRLS, ALTS):
            for events in orderings((c, a, VK_N)):
                for tracked in (False, True):
                    with self.subTest(c=hex(c), a=hex(a), events=events, tracked=tracked):
                        phys = TrackPhysical() if tracked else None
                        m = HoldChord(physical=phys)
                        self.assertEqual(feed(m, events, phys), [])
                        self.assertEqual(m.state, winkeys.IDLE)
                        self.assertEqual(m.down, set())
                        self.assertFalse(m.mask_pending)
                        dictation_still_works(self, m, phys)

    def test_held_with_autorepeat(self):
        m = HoldChord()
        events = [(VK_LCONTROL, True), (VK_LMENU, True)] + [(VK_N, True)] * 40 + \
                 [(VK_N, False), (VK_LMENU, False), (VK_LCONTROL, False)]
        self.assertEqual(feed(m, events), [])
        dictation_still_works(self, m)

    def test_every_accepted_modifier_set(self):
        """Any shortcut parse_hotkey accepts, pressed with its modifiers in any order and released in any order."""
        bits = list(MOD_VK)
        for r in range(len(bits) + 1):
            for combo in itertools.combinations(bits, r):
                for key in ("n", "f5", "f23"):
                    spec = "+".join([*(n for n, b in winkeys._MOD_NAMES.items() if b in combo), key])
                    try:
                        mods, vk = parse_hotkey(spec)
                    except ValueError:
                        continue
                    vks = [MOD_VK[b] for b in combo]
                    for downs in itertools.permutations(vks):
                        for ups in itertools.permutations(vks):
                            events = [(k, True) for k in downs] + [(vk, True), (vk, False)] + \
                                     [(k, False) for k in ups]
                            with self.subTest(spec=spec, events=events):
                                m = HoldChord()
                                self.assertEqual(feed(m, events), [])
                                self.assertEqual(m.state, winkeys.IDLE)
                                dictation_still_works(self, m)

    def test_ctrl_win_would_press_which_is_why_it_is_refused(self):
        m = HoldChord()
        acts = feed(m, [(VK_LCONTROL, True), (VK_LWIN, True), (VK_N, True)])
        self.assertEqual(acts, [(PRESS, MASK), (CANCEL, MASK)])
        with self.assertRaises(ValueError):
            parse_hotkey("ctrl+win+n")

    def test_shortcut_right_after_a_dictation_with_ctrl_still_held(self):
        m = HoldChord()
        self.assertEqual(feed(m, [(VK_LCONTROL, True), (VK_LWIN, True)]), [(PRESS, MASK)])
        self.assertEqual(feed(m, [(VK_LWIN, False)]), [(RELEASE, MASK)])
        self.assertEqual(feed(m, [(VK_LMENU, True), (VK_N, True), (VK_N, False), (VK_LMENU, False),
                                  (VK_LCONTROL, False)]), [])
        dictation_still_works(self, m)

    def test_missed_key_ups_do_not_block_the_next_dictation(self):
        phys = TrackPhysical()
        m = HoldChord(physical=phys)
        feed(m, [(VK_RCONTROL, True), (VK_RMENU, True), (VK_N, True)], phys)
        phys.down.clear()                     # every key-up went to the secure desktop
        dictation_still_works(self, m, phys)


class HookPassesShortcut(unittest.TestCase):
    """The real hook procedure, called directly: every key of the shortcut goes on to Windows untouched."""

    def setUp(self):
        self.calls: list[str] = []
        self.hk = HoldHotkey(lambda: self.calls.append("press"), lambda: self.calls.append("release"),
                             lambda: self.calls.append("cancel"))
        self.hk._chord.physical = None
        self.next_hook = mock.patch.object(winkeys, "_CallNextHookEx", return_value=7).start()
        self.mask = mock.patch.object(winkeys, "send_mask", return_value=True).start()
        self.addCleanup(mock.patch.stopall)

    def key(self, vk, down=True):
        kb = winkeys.KBDLLHOOKSTRUCT(vkCode=vk, scanCode=0, time=0, dwExtraInfo=0,
                                     flags=0 if down else winkeys.LLKHF_UP)
        w = 0x104 if down else 0x105              # WM_SYSKEYDOWN / WM_SYSKEYUP: Alt is held
        self.assertEqual(self.hk._proc(winkeys.HC_ACTION, w, ctypes.addressof(kb)), 7)
        self.assertEqual(self.next_hook.call_args.args[1:], (winkeys.HC_ACTION, w, ctypes.addressof(kb)))

    def test_ctrl_alt_n_passes_through(self):
        for c, a in itertools.product(CTRLS[:2], ALTS[:2]):
            for vk, down in [(c, True), (a, True), (VK_N, True), (VK_N, True), (VK_N, False), (a, False),
                             (c, False)]:
                self.key(vk, down)
        self.assertEqual(self.next_hook.call_count, 28)    # never swallowed
        self.mask.assert_not_called()                      # nothing injected between the shortcut's keys
        self.hk._events.put(None)
        self.hk._work()
        self.assertEqual(self.calls, [])


class RealShortcut(unittest.TestCase):
    """RegisterHotKey for real, on the shortcut's own thread. Sends no keys."""

    def setUp(self):
        if SPEC is None:
            self.skipTest(f"none of {CANDIDATES} is free on this machine")
        self.mods, self.vk = parse_hotkey(SPEC)
        self.made: list[Hotkey] = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for hk in self.made:
            hk.stop()
        for t in threading.enumerate():               # a failed test must not leave the next one a held combination
            if t.name == THREAD:
                t.join(2)

    def make(self, callback=lambda: None) -> Hotkey:
        hk = Hotkey(SPEC, callback)
        self.made.append(hk)
        return hk

    def post(self, hk: Hotkey, ident: int = HOTKEY_ID, message: int = WM_HOTKEY) -> None:
        self.assertTrue(winkeys._PostThreadMessageW(hk._run_state.tid, message, ident, (self.vk << 16) | self.mods))

    def assert_nothing_left(self):
        self.assertEqual([t.name for t in threading.enumerate() if t.name == THREAD], [])
        self.assertTrue(free(SPEC), "the registration outlived stop()")

    def test_start_dispatch_stop(self):
        fired, threads = threading.Event(), []
        hk = self.make(lambda: (threads.append(threading.current_thread()), fired.set()))
        self.assertTrue(hk.start())
        self.assertTrue(hk.running)
        self.assertFalse(free(SPEC))                      # Windows holds it for the shortcut's thread
        t = hk._run_state.thread
        self.assertTrue(t.daemon)
        self.assertEqual(t.name, THREAD)
        self.post(hk, ident=HOTKEY_ID + 1)                # another id: not ours
        self.post(hk, message=winkeys.WM_USER)            # another message: ignored
        self.assertFalse(fired.wait(0.2))
        self.post(hk)
        self.assertTrue(fired.wait(2))
        self.assertIs(threads[0], t)                      # the callback runs on the shortcut's thread
        hk.stop()
        self.assertFalse(hk.running)
        self.assertFalse(t.is_alive())
        self.assertIsNone(hk._run_state)
        self.assert_nothing_left()

    def test_every_post_fires_once(self):
        count = []
        done = threading.Event()
        hk = self.make(lambda: (count.append(1), len(count) == 3 and done.set()))
        self.assertTrue(hk.start())
        for _ in range(3):
            self.post(hk)
        self.assertTrue(done.wait(2))
        time.sleep(0.1)
        self.assertEqual(len(count), 3)

    def test_restart_and_idempotent(self):
        hk = self.make()
        hk.stop()                                         # before start: no-op
        for _ in range(3):
            self.assertTrue(hk.start())
            first = hk._run_state.thread
            self.assertTrue(hk.start())                   # second start: no-op, same thread
            self.assertIs(hk._run_state.thread, first)
            hk.stop()
            hk.stop()
            self.assertFalse(hk.running)
            self.assert_nothing_left()

    def test_refused_when_taken(self):
        a, b = self.make(), self.make()
        self.assertTrue(a.start())
        with self.assertLogs("scribe.winkeys", "WARNING") as logs:
            self.assertFalse(b.start())
        self.assertEqual(len(logs.records), 1)
        self.assertIn("another program holds it", logs.output[0])
        self.assertEqual(b.error, winkeys.ERROR_HOTKEY_ALREADY_REGISTERED)
        self.assertFalse(b.running)
        self.assertIsNone(b._run_state)
        with self.assertNoLogs("scribe.winkeys", "WARNING"):  # logged once, not on every retry
            self.assertFalse(b.start())
        self.assertTrue(a.running)                        # the refusal left the holder alone
        self.assertEqual([t.name for t in threading.enumerate() if t.name == THREAD], [THREAD])
        a.stop()
        self.assertTrue(b.start())                        # free again: the retry succeeds and resets the log
        self.assertEqual(b.error, 0)
        b.stop()
        self.assertIsNone(b._refusal_why)
        self.assert_nothing_left()

    def test_callback_error_keeps_the_loop(self):
        calls, second = [], threading.Event()

        def cb():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            second.set()

        hk = self.make(cb)
        self.assertTrue(hk.start())
        with self.assertLogs("scribe.winkeys", "ERROR"):
            self.post(hk)
            self.post(hk)
            self.assertTrue(second.wait(2))
        self.assertTrue(hk.running)

    def test_stop_from_the_callback(self):
        hk = self.make()
        hk.callback = hk.stop
        self.assertTrue(hk.start())
        t = hk._run_state.thread
        with self.assertNoLogs("scribe.winkeys", "ERROR"):    # no attempt to join its own thread
            self.post(hk)
            t.join(2)
        self.assertFalse(t.is_alive())
        self.assertFalse(hk.running)
        self.assert_nothing_left()
        self.assertTrue(hk.start())                       # and it can start again
        hk.stop()

    def test_stop_during_a_slow_callback(self):
        """stop() while the callback runs: the press queued behind it never fires, and start() blames Scribe's
        old thread, not another program, until that callback returns."""
        entered, gate, calls = threading.Event(), threading.Event(), []

        def slow():
            calls.append(1)
            entered.set()
            gate.wait(5)

        hk = self.make(slow)
        hk.STOP_S = 0.1
        self.addCleanup(gate.set)
        self.assertTrue(hk.start())
        t = hk._run_state.thread
        self.post(hk)
        self.assertTrue(entered.wait(2))
        self.post(hk)                                     # queued behind the callback, ahead of stop()'s WM_QUIT
        with self.assertLogs("scribe.winkeys", "WARNING"):
            hk.stop()
        self.assertTrue(t.is_alive())                     # Windows still holds the combination for that thread
        self.assertFalse(hk.running)
        with self.assertLogs("scribe.winkeys", "WARNING") as logs:
            self.assertFalse(hk.start())
        self.assertNotIn("another program", logs.output[0])
        self.assertIn("callback", logs.output[0])
        self.assertEqual(hk.error, 0)
        self.assertEqual([x for x in threading.enumerate() if x.name == THREAD], [t])   # no second thread
        gate.set()
        t.join(2)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(calls), 1)                   # the queued press did not fire after stop()
        self.assert_nothing_left()
        other = self.make()
        self.assertTrue(other.start())
        with self.assertLogs("scribe.winkeys", "WARNING") as logs:   # a new reason warns again
            self.assertFalse(hk.start())
        self.assertIn("another program holds it", logs.output[0])
        other.stop()
        self.assertTrue(hk.start())
        hk.stop()
        self.assert_nothing_left()

    def test_start_right_after_stop_from_the_callback(self):
        """The callback has stopped the shortcut and is still returning when start() comes from another thread:
        start() waits for the old thread rather than reading its registration as another program's."""
        stopped = threading.Event()

        def cb():
            hk.stop()
            stopped.set()
            time.sleep(0.3)

        hk = self.make(cb)
        self.assertTrue(hk.start())
        t = hk._run_state.thread
        self.post(hk)
        self.assertTrue(stopped.wait(2))
        with self.assertNoLogs("scribe.winkeys", "WARNING"):
            self.assertTrue(hk.start())
        self.assertFalse(t.is_alive())
        self.assertIsNot(hk._run_state.thread, t)
        hk.stop()
        self.assert_nothing_left()

    def test_restart_from_its_own_callback_is_refused(self):
        results = []

        def cb():
            hk.stop()
            results.append(hk.start())                    # this thread holds it until the callback returns

        hk = self.make(cb)
        self.assertTrue(hk.start())
        t = hk._run_state.thread
        with self.assertLogs("scribe.winkeys", "WARNING") as logs:
            self.post(hk)
            t.join(2)
        self.assertEqual(results, [False])
        self.assertNotIn("another program", logs.output[0])
        self.assertFalse(t.is_alive())
        self.assert_nothing_left()
        self.assertTrue(hk.start())
        hk.stop()

    def test_restart_after_the_loop_ended_on_its_own(self):
        hk = self.make()
        self.assertTrue(hk.start())
        t = hk._run_state.thread
        self.post(hk, ident=0, message=winkeys.WM_QUIT)   # a quit Scribe did not send
        t.join(2)
        self.assertFalse(t.is_alive())
        self.assertFalse(hk.running)
        self.assertTrue(free(SPEC))                       # the loop unregistered on its way out
        self.assertTrue(hk.start())
        self.assertIsNot(hk._run_state.thread, t)
        hk.stop()
        self.assert_nothing_left()

    def test_late_registration_is_taken_out(self):
        gate, threads = threading.Event(), []

        def slow(*args):
            threads.append(threading.current_thread())
            gate.wait(5)
            return REAL_REGISTER(*args)

        hk = self.make()
        hk.READY_S = 0.1
        with mock.patch.object(winkeys, "_RegisterHotKey", side_effect=slow), \
                mock.patch.object(winkeys, "_UnregisterHotKey", wraps=winkeys._UnregisterHotKey) as unreg:
            with self.assertLogs("scribe.winkeys", "WARNING") as logs:
                self.assertFalse(hk.start())
            self.assertIn("did not start", logs.output[0])
            gate.set()
            threads[0].join(2)
        # Windows also frees a thread's hotkeys when the thread exits; the explicit call is checked here.
        unreg.assert_called_once_with(None, HOTKEY_ID)
        self.assertFalse(threads[0].is_alive())
        self.assertFalse(hk.running)
        self.assert_nothing_left()

    def test_stop_returns_quickly(self):
        hk = self.make()
        self.assertTrue(hk.start())
        t = time.monotonic()
        hk.stop()
        self.assertLess(time.monotonic() - t, 1.0)

    def test_side_by_side_with_the_ctrl_win_hook(self):
        hold = HoldHotkey(lambda: None, lambda: None, lambda: None)
        hold.start()
        try:
            hk = self.make()
            self.assertTrue(hk.start())
            self.assertTrue(hold.running and hk.running)
            self.assertNotEqual(hk._run_state.tid, hold._thread_id)
        finally:
            hold.stop()
        self.assertTrue(hk.running)                       # removing the hook leaves the shortcut alone
        hk.stop()
        self.assert_nothing_left()


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
