"""winkeys: the Ctrl+Win chord state machine (exhaustively), the Win32 structures, and the real hook's start/stop.

No test sends a key. The real hook is installed and removed only; dispatch is tested by calling the hook
procedure directly with a fabricated KBDLLHOOKSTRUCT and SendInput patched out.
"""
from __future__ import annotations

import ctypes
import itertools
import threading
import time
import unittest
from unittest import mock

from scribe import winkeys
from scribe.winkeys import (CANCEL, MASK, PRESS, RELEASE, VK_CONTROL, VK_LCONTROL, VK_LMENU, VK_LSHIFT, VK_LWIN,
                            VK_MASK, VK_RCONTROL, VK_RSHIFT, VK_RWIN, HoldChord, HoldHotkey)

VK_A, VK_D, VK_L, VK_C = 0x41, 0x44, 0x4C, 0x43
CTRLS = (VK_LCONTROL, VK_RCONTROL, VK_CONTROL)
WINS = (VK_LWIN, VK_RWIN)
OTHERS = (VK_A, VK_D, VK_LSHIFT, VK_RSHIFT, VK_LMENU)


def run(machine: HoldChord, events, injected: bool = False) -> list[tuple[int, tuple[str, ...]]]:
    """Feed (vk, is_down) events; return (index, actions) for every event that produced actions."""
    out = []
    for i, (vk, down) in enumerate(events):
        acts = machine.feed(vk, down, injected)
        if acts:
            out.append((i, acts))
    return out


def interleavings(keys):
    """Every ordering of each key's down and up with down before up: 90 orderings for three keys."""
    tokens = [(k, d) for k in keys for d in (True, False)]
    seen = set()
    for perm in itertools.permutations(tokens):
        if perm in seen:
            continue
        seen.add(perm)
        if all(perm.index((k, True)) < perm.index((k, False)) for k in keys):
            yield list(perm)


def oracle(events, c, w, x) -> list[tuple[int, tuple[str, ...]]]:
    """Expected actions written straight from the DESIGN.md rules, independently of HoldChord."""
    pos = {ev: i for i, ev in enumerate(events)}
    c_d, c_u, w_d, w_u = pos[(c, True)], pos[(c, False)], pos[(w, True)], pos[(w, False)]
    x_d, x_u = pos[(x, True)], pos[(x, False)]
    first, second, first_up = min(c_d, w_d), max(c_d, w_d), min(c_u, w_u)
    x_interferes = x_d < second and x_u > first      # X held at any moment from the first chord key to the second
    if not (second < first_up) or x_interferes:
        return []
    out = {second: (PRESS, MASK)}
    if second < x_d < first_up:
        out[x_d] = (CANCEL, MASK)
        out[w_u] = (MASK,)
    else:
        out[first_up] = (RELEASE, MASK) if first_up == w_u else (RELEASE,)
        out.setdefault(w_u, (MASK,))
    return sorted(out.items())


class TrackPhysical:
    """GetAsyncKeyState as seen inside the hook: the state before the event being processed."""

    def __init__(self):
        self.down: set[int] = set()

    def __call__(self, vk: int) -> bool:
        return vk in self.down

    def after(self, vk: int, is_down: bool) -> None:
        (self.down.add if is_down else self.down.discard)(vk)


class ChordExhaustive(unittest.TestCase):
    """All 90 interleavings of a Ctrl key, a Win key and one other key, for every left/right variant."""

    def cases(self):
        for c, w, x in itertools.product(CTRLS, WINS, OTHERS):
            for events in interleavings((c, w, x)):
                yield c, w, x, events

    def test_matches_rules(self):
        n = 0
        for c, w, x, events in self.cases():
            n += 1
            with self.subTest(c=hex(c), w=hex(w), x=hex(x), events=events):
                self.assertEqual(run(HoldChord(), events), oracle(events, c, w, x))
        self.assertEqual(n, 3 * 2 * 5 * 90)

    def test_matches_rules_with_physical_resync(self):
        for c, w, x, events in self.cases():
            phys = TrackPhysical()
            m = HoldChord(physical=phys)
            got = []
            for i, (vk, down) in enumerate(events):
                acts = m.feed(vk, down)
                phys.after(vk, down)
                if acts:
                    got.append((i, acts))
            with self.subTest(c=hex(c), w=hex(w), x=hex(x), events=events):
                self.assertEqual(got, oracle(events, c, w, x))
                self.assertEqual(m.resyncs, 0)

    def test_autorepeat_never_changes_outcome(self):
        """Windows repeats the last key pressed while it is held; repeats must never fire or cancel anything."""
        for c, w, x, events in self.cases():
            m = HoldChord()
            got, held, last = [], set(), None
            for i, (vk, down) in enumerate(events):
                acts = m.feed(vk, down)
                if acts:
                    got.append((i, acts))
                if down:
                    held.add(vk)
                    last = vk
                else:
                    held.discard(vk)
                    if vk == last:
                        last = None
                if last is not None:
                    for _ in range(3):
                        self.assertEqual(m.feed(last, True), (), f"repeat of {last:#x} after {events[:i + 1]}")
            with self.subTest(events=events):
                self.assertEqual(got, oracle(events, c, w, x))

    def test_every_press_closes_exactly_once_and_mask_precedes_win_up(self):
        for c, w, x, events in self.cases():
            acts = [a for _, group in run(HoldChord(), events) for a in group]
            with self.subTest(events=events):
                self.assertLessEqual(acts.count(PRESS), 1)
                self.assertEqual(acts.count(PRESS), acts.count(RELEASE) + acts.count(CANCEL))
                if PRESS in acts:
                    w_up = events.index((w, False))
                    self.assertIn(MASK, dict(run(HoldChord(), events))[w_up])

    def test_injected_ignored_unless_accepted(self):
        for c, w, x, events in self.cases():
            with self.subTest(events=events):
                self.assertEqual(run(HoldChord(), events, injected=True), [])
                self.assertEqual(run(HoldChord(accept_injected=True), events, injected=True), oracle(events, c, w, x))


class ChordCases(unittest.TestCase):
    """Named cases, one behaviour each."""

    def test_ctrl_then_win(self):
        m = HoldChord()
        self.assertEqual(m.feed(VK_LCONTROL, True), ())
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))
        self.assertEqual(m.feed(VK_LWIN, False), (RELEASE, MASK))
        self.assertEqual(m.feed(VK_LCONTROL, False), ())
        self.assertEqual(m.state, winkeys.IDLE)

    def test_win_then_ctrl(self):
        m = HoldChord()
        self.assertEqual(m.feed(VK_RWIN, True), ())
        self.assertEqual(m.feed(VK_RCONTROL, True), (PRESS, MASK))
        self.assertEqual(m.feed(VK_RCONTROL, False), (RELEASE,))
        self.assertEqual(m.feed(VK_RWIN, False), (MASK,))

    def test_all_left_right_pairs(self):
        for c, w in itertools.product(CTRLS, WINS):
            with self.subTest(c=hex(c), w=hex(w)):
                m = HoldChord()
                self.assertEqual(m.feed(c, True) + m.feed(w, True), (PRESS, MASK))

    def test_other_key_while_held_cancels_and_never_releases(self):
        m = HoldChord()
        m.feed(VK_LCONTROL, True)
        m.feed(VK_LWIN, True)
        self.assertEqual(m.feed(VK_D, True), (CANCEL, MASK))      # Ctrl+Win+D still reaches Windows
        self.assertEqual(m.feed(VK_D, False), ())
        self.assertEqual(m.feed(VK_LWIN, False), (MASK,))
        self.assertEqual(m.feed(VK_LCONTROL, False), ())

    def test_shift_or_alt_while_held_cancels(self):
        for other in (VK_LSHIFT, VK_RSHIFT, VK_LMENU):
            m = HoldChord()
            m.feed(VK_LCONTROL, True)
            m.feed(VK_LWIN, True)
            with self.subTest(other=hex(other)):
                self.assertEqual(m.feed(other, True), (CANCEL, MASK))

    def test_autorepeat_while_held_does_not_refire(self):
        m = HoldChord()
        m.feed(VK_LCONTROL, True)
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))
        for _ in range(30):
            self.assertEqual(m.feed(VK_LWIN, True), ())
            self.assertEqual(m.feed(VK_LCONTROL, True), ())
        self.assertEqual(m.feed(VK_LCONTROL, False), (RELEASE,))

    def test_autorepeat_before_second_key(self):
        m = HoldChord()
        for _ in range(10):
            self.assertEqual(m.feed(VK_LCONTROL, True), ())
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))

    def test_other_key_between_first_and_second_blocks_press(self):
        m = HoldChord()
        m.feed(VK_LCONTROL, True)
        m.feed(VK_C, True)
        m.feed(VK_C, False)                                   # Ctrl+C, then Win while Ctrl is still held
        self.assertEqual(m.feed(VK_LWIN, True), ())
        self.assertEqual(m.feed(VK_LWIN, False), ())          # no mask: Windows handles this Win key as usual
        self.assertEqual(m.feed(VK_LCONTROL, False), ())

    def test_key_held_before_chord_blocks_press(self):
        m = HoldChord()
        m.feed(VK_LSHIFT, True)
        m.feed(VK_LCONTROL, True)
        self.assertEqual(m.feed(VK_LWIN, True), ())

    def test_win_tap_alone_is_not_masked(self):
        m = HoldChord()
        self.assertEqual(m.feed(VK_LWIN, True), ())
        self.assertEqual(m.feed(VK_LWIN, False), ())         # the Start menu still opens on a plain Win tap

    def test_win_shortcut_untouched(self):
        m = HoldChord()
        self.assertEqual(run(m, [(VK_LWIN, True), (VK_A + 4, True), (VK_A + 4, False), (VK_LWIN, False)]), [])

    def test_no_second_press_until_both_released(self):
        m = HoldChord()
        m.feed(VK_LCONTROL, True)
        m.feed(VK_LWIN, True)
        self.assertEqual(m.feed(VK_LCONTROL, False), (RELEASE,))
        self.assertEqual(m.feed(VK_LCONTROL, True), ())       # Win still held: no re-fire
        self.assertEqual(m.feed(VK_LCONTROL, False), ())
        self.assertEqual(m.feed(VK_LWIN, False), (MASK,))
        m.feed(VK_LCONTROL, True)
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))

    def test_both_ctrl_keys_release_when_family_empties(self):
        m = HoldChord()
        m.feed(VK_LCONTROL, True)
        m.feed(VK_RCONTROL, True)
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))
        self.assertEqual(m.feed(VK_RCONTROL, False), ())
        self.assertEqual(m.feed(VK_LCONTROL, False), (RELEASE,))

    def test_second_ctrl_while_held_is_not_other_key(self):
        m = HoldChord()
        m.feed(VK_LCONTROL, True)
        m.feed(VK_LWIN, True)
        self.assertEqual(m.feed(VK_RCONTROL, True), ())

    def test_both_win_keys_each_masked(self):
        m = HoldChord()
        m.feed(VK_LCONTROL, True)
        m.feed(VK_LWIN, True)
        m.feed(VK_RWIN, True)
        self.assertEqual(m.feed(VK_RWIN, False), (MASK,))
        self.assertEqual(m.feed(VK_LWIN, False), (RELEASE, MASK))
        self.assertFalse(m.mask_pending)

    def test_mask_key_always_ignored(self):
        m = HoldChord(accept_injected=True)
        m.feed(VK_LCONTROL, True)
        m.feed(VK_LWIN, True)
        self.assertEqual(m.feed(VK_MASK, True, injected=True), ())
        self.assertEqual(m.feed(VK_MASK, False, injected=True), ())
        self.assertEqual(m.feed(VK_MASK, True), ())
        self.assertEqual(m.feed(VK_LWIN, False), (RELEASE, MASK))

    def test_injected_other_key_while_held_does_not_cancel(self):
        m = HoldChord()
        m.feed(VK_LCONTROL, True)
        m.feed(VK_LWIN, True)
        self.assertEqual(m.feed(VK_D, True, injected=True), ())
        self.assertEqual(m.feed(VK_LWIN, False), (RELEASE, MASK))

    def test_injected_chord_ignored(self):
        m = HoldChord()
        self.assertEqual(m.feed(VK_LCONTROL, True, True) + m.feed(VK_LWIN, True, True), ())
        self.assertEqual(m.state, winkeys.IDLE)

    def test_unseen_key_up_is_harmless(self):
        m = HoldChord()
        self.assertEqual(m.feed(VK_LWIN, False), ())
        self.assertEqual(m.feed(VK_A, False), ())
        m.feed(VK_LCONTROL, True)
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))


class ChordResync(unittest.TestCase):
    """Key-ups lost to the secure desktop (Win+L, Ctrl+Alt+Del, UAC) must not wedge the chord."""

    def test_after_win_l_lock(self):
        phys = {VK_LWIN: True, VK_L: True}
        m = HoldChord(physical=lambda vk: phys.get(vk, False))
        m.feed(VK_LWIN, True)
        m.feed(VK_L, True)                       # workstation locks; both key-ups land on the secure desktop
        phys.clear()
        self.assertEqual(m.feed(VK_LCONTROL, True), ())
        phys[VK_LCONTROL] = True
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))
        self.assertEqual(m.resyncs, 1)

    def test_after_win_l_lock_win_pressed_first(self):
        phys = {VK_LWIN: True, VK_L: True}
        m = HoldChord(physical=lambda vk: phys.get(vk, False))
        m.feed(VK_LWIN, True)
        m.feed(VK_L, True)
        phys.clear()
        self.assertEqual(m.feed(VK_LWIN, True), ())      # a fresh press, not an autorepeat of the stale one
        phys[VK_LWIN] = True
        self.assertEqual(m.feed(VK_LCONTROL, True), (PRESS, MASK))

    def test_without_resync_the_stale_lock_blocks(self):
        m = HoldChord()
        m.feed(VK_LWIN, True)
        m.feed(VK_L, True)
        m.feed(VK_LCONTROL, True)
        self.assertEqual(m.feed(VK_LWIN, True), ())      # shows why the hook passes GetAsyncKeyState in

    def test_after_ctrl_alt_del_stale_ctrl_does_not_fire_on_win_alone(self):
        phys = {VK_LCONTROL: True, VK_LMENU: True}
        m = HoldChord(physical=lambda vk: phys.get(vk, False))
        m.feed(VK_LCONTROL, True)
        m.feed(VK_LMENU, True)
        phys.clear()
        self.assertEqual(m.feed(VK_LWIN, True), ())
        self.assertEqual(m.feed(VK_LWIN, False), ())

    def test_stuck_hold_is_cancelled(self):
        phys = {VK_LCONTROL: True, VK_LWIN: True}
        m = HoldChord(physical=lambda vk: phys.get(vk, False))
        m.feed(VK_LCONTROL, True)
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))
        phys.clear()                             # a UAC prompt took the key-ups
        self.assertEqual(m.feed(VK_LWIN, True), (CANCEL, MASK))
        phys[VK_LWIN] = True
        self.assertEqual(m.feed(VK_LCONTROL, True), (PRESS, MASK))

    def test_stale_other_key_does_not_block_forever(self):
        phys = {VK_A: True}
        m = HoldChord(physical=lambda vk: phys.get(vk, False))
        m.feed(VK_A, True)
        phys.clear()
        m.feed(VK_LCONTROL, True)
        phys[VK_LCONTROL] = True
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))


class ChordWatch(unittest.TestCase):
    """The off-hook watchdog: key-ups the hook never sees (elevated window, secure desktop) still end a hold."""

    def held(self):
        phys = {VK_LCONTROL: True, VK_LWIN: True}
        m = HoldChord(physical=lambda vk: phys.get(vk, False))
        m.feed(VK_LCONTROL, True)
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))
        return m, phys

    def test_nothing_while_keys_are_down_or_idle(self):
        m, _ = self.held()
        self.assertEqual(m.watch(), ())
        self.assertEqual(HoldChord(physical=lambda vk: False).watch(), ())

    def test_ctrl_released_over_an_elevated_window(self):
        m, phys = self.held()
        del phys[VK_LCONTROL]
        self.assertEqual(m.watch(), (RELEASE,))
        self.assertEqual(m.state, "spent")
        self.assertEqual(m.watch(), ())                    # once only
        self.assertEqual(m.feed(VK_LWIN, False), (MASK,))  # the Win key-up the hook does see is still masked
        self.assertEqual(m.state, "idle")

    def test_secure_desktop_reads_every_key_up(self):
        m, phys = self.held()
        phys.clear()
        self.assertEqual(m.watch(), (RELEASE,))
        self.assertEqual((m.state, m.down, m.mask_pending), ("idle", set(), False))
        phys.update({VK_LCONTROL: True, VK_LWIN: True})
        m.feed(VK_LCONTROL, True)
        self.assertEqual(m.feed(VK_LWIN, True), (PRESS, MASK))   # the next hold works


class Generations(unittest.TestCase):
    """A hook thread that runs only after its start() gave up must not install an orphan hook."""

    def test_stale_generation_never_hooks(self):
        hk = HoldHotkey(lambda: None, lambda: None, lambda: None)
        hk._gen = 5
        ready = threading.Event()
        with mock.patch.object(winkeys, "_SetWindowsHookExW") as hook:
            hk._run(4, ready)
        hook.assert_not_called()
        self.assertEqual(hk._thread_id, 0)                 # the current generation's id is never overwritten
        self.assertIn(4, hk._tids)
        self.assertNotIn(id(hk), winkeys._LIVE_PROCS)

    def test_old_thread_never_drops_a_newer_procedure(self):
        hk = HoldHotkey(lambda: None, lambda: None, lambda: None)
        hk.start()
        newer = object()
        winkeys._LIVE_PROCS[id(hk)] = newer               # as if a newer generation had hooked meanwhile
        try:
            hk.stop()
            self.assertIs(winkeys._LIVE_PROCS.get(id(hk)), newer)
        finally:
            winkeys._LIVE_PROCS.pop(id(hk), None)

    def test_stuck_thread_is_kept_and_asked_to_quit(self):
        hk = HoldHotkey(lambda: None, lambda: None, lambda: None)
        stuck = threading.Event()
        t = threading.Thread(target=stuck.wait, args=(5,), daemon=True)
        t.start()
        hk._stale.append((3, t))
        hk._tids[3] = 1234
        with mock.patch.object(winkeys, "_PostThreadMessageW", return_value=1) as post:
            hk._reap_stale()
        post.assert_called_once_with(1234, winkeys.WM_QUIT, 0, 0)
        self.assertEqual(len(hk._stale), 1)
        stuck.set()
        t.join(2)
        hk._reap_stale()
        self.assertEqual(hk._stale, [])


class Structures(unittest.TestCase):
    def test_sizes_64bit(self):
        self.assertEqual(ctypes.sizeof(ctypes.c_void_p), 8)
        self.assertEqual(ctypes.sizeof(winkeys.INPUT), 40)
        self.assertEqual(ctypes.sizeof(winkeys.KEYBDINPUT), 24)
        self.assertEqual(ctypes.sizeof(winkeys.MOUSEINPUT), 32)
        self.assertEqual(ctypes.sizeof(winkeys.KBDLLHOOKSTRUCT), 24)
        self.assertEqual(winkeys.INPUT.u.offset, 8)
        self.assertEqual(winkeys.KBDLLHOOKSTRUCT.dwExtraInfo.offset, 16)

    def test_every_win32_function_declared(self):
        for f in winkeys.WIN32_FUNCTIONS:
            with self.subTest(f=f.__name__):
                self.assertIsNotNone(f.argtypes)
                self.assertIsNotNone(f.restype)
        self.assertIs(winkeys._SetWindowsHookExW.restype, ctypes.wintypes.HHOOK)
        self.assertIs(winkeys._CallNextHookEx.restype, ctypes.c_ssize_t)
        self.assertIs(winkeys._CallNextHookEx.argtypes[2], ctypes.c_size_t)
        self.assertIs(winkeys._CallNextHookEx.argtypes[3], ctypes.c_ssize_t)

    def test_build_inputs(self):
        arr = winkeys.build_inputs([(VK_CONTROL, True), (0x56, True), (0x56, False), (VK_CONTROL, False)])
        self.assertEqual(len(arr), 4)
        self.assertEqual([i.type for i in arr], [winkeys.INPUT_KEYBOARD] * 4)
        self.assertEqual([i.ki.wVk for i in arr], [VK_CONTROL, 0x56, 0x56, VK_CONTROL])
        self.assertEqual([i.ki.dwFlags & winkeys.KEYEVENTF_KEYUP for i in arr], [0, 0, 2, 2])
        self.assertTrue(all(i.ki.dwExtraInfo == winkeys.SCRIBE_TAG for i in arr))
        self.assertTrue(all(i.ki.wScan for i in arr))
        win = winkeys.build_inputs([(VK_LWIN, False)])[0]
        self.assertTrue(win.ki.dwFlags & winkeys.KEYEVENTF_EXTENDEDKEY)

    def test_send_keys_uses_one_call_with_cbsize_40(self):
        with mock.patch.object(winkeys, "_SendInput", return_value=2) as send:
            self.assertTrue(winkeys.send_mask())
        n, arr, cb = send.call_args.args
        self.assertEqual((n, cb), (2, 40))
        self.assertEqual([(i.ki.wVk, i.ki.dwFlags & 2) for i in arr], [(VK_MASK, 0), (VK_MASK, 2)])
        with mock.patch.object(winkeys, "_SendInput", return_value=1), self.assertLogs("scribe.winkeys", "WARNING"):
            self.assertFalse(winkeys.send_mask())


class Modifiers(unittest.TestCase):
    def test_modifiers_down(self):
        held = {VK_RSHIFT}
        fake = lambda vk: -32768 if vk in held else 0  # noqa: E731  (high bit set, as a c_short)
        with mock.patch.object(winkeys, "_GetAsyncKeyState", side_effect=fake):
            self.assertTrue(winkeys.modifiers_down())
            held.clear()
            self.assertFalse(winkeys.modifiers_down())
            held.add(VK_A)
            self.assertFalse(winkeys.modifiers_down())

    def test_wait_until_released(self):
        calls = {"n": 0}

        def fake(vk):
            calls["n"] += 1
            return -32768 if vk == VK_LWIN and calls["n"] < 40 else 0

        with mock.patch.object(winkeys, "_GetAsyncKeyState", side_effect=fake):
            self.assertTrue(winkeys.wait_modifiers_released(timeout=2.0))

    def test_wait_times_out(self):
        with mock.patch.object(winkeys, "_GetAsyncKeyState", return_value=-32768):
            t = time.monotonic()
            self.assertFalse(winkeys.wait_modifiers_released(timeout=0.05))
            self.assertLess(time.monotonic() - t, 1.0)


class HookDispatch(unittest.TestCase):
    """The hook procedure called directly: decisions, mask injection, queue to the worker, keys always passed on."""

    def setUp(self):
        self.calls: list[str] = []
        self.hk = HoldHotkey(lambda: self.calls.append("press"), lambda: self.calls.append("release"),
                             lambda: self.calls.append("cancel"))
        self.hk._chord.physical = None
        self.next_hook = mock.patch.object(winkeys, "_CallNextHookEx", return_value=7).start()
        self.mask = mock.patch.object(winkeys, "send_mask", return_value=True).start()
        self.addCleanup(mock.patch.stopall)

    def key(self, vk, down=True, injected=False, tag=0):
        kb = winkeys.KBDLLHOOKSTRUCT(vkCode=vk, scanCode=0, time=0, dwExtraInfo=tag,
                                     flags=(0 if down else winkeys.LLKHF_UP) | (winkeys.LLKHF_INJECTED if injected else 0))
        w = 0x100 if down else 0x101
        self.assertEqual(self.hk._proc(winkeys.HC_ACTION, w, ctypes.addressof(kb)), 7)
        self.assertEqual(self.next_hook.call_args.args[1:], (winkeys.HC_ACTION, w, ctypes.addressof(kb)))

    def drain(self):
        self.hk._events.put(None)
        self.hk._work()

    def test_press_release_with_mask(self):
        self.key(VK_LCONTROL)
        self.key(VK_LWIN)
        self.assertEqual(self.mask.call_count, 1)
        self.key(VK_D, injected=True)                       # ignored
        self.key(VK_D, tag=winkeys.SCRIBE_TAG)              # our own injection: ignored
        self.key(VK_LWIN, down=False)
        self.assertEqual(self.mask.call_count, 2)            # again before the Win key-up is passed on
        self.key(VK_LCONTROL, down=False)
        self.assertEqual(self.next_hook.call_count, 6)       # never swallowed
        self.drain()
        self.assertEqual(self.calls, ["press", "release"])

    def test_cancel(self):
        for vk in (VK_LWIN, VK_LCONTROL, VK_D):
            self.key(vk)
        for vk in (VK_D, VK_LCONTROL, VK_LWIN):
            self.key(vk, down=False)
        self.drain()
        self.assertEqual(self.calls, ["press", "cancel"])
        self.assertEqual(self.mask.call_count, 3)            # press, cancel, Win key-up

    def test_accept_injected(self):
        self.hk._chord = HoldChord(accept_injected=True)
        self.key(VK_LCONTROL, injected=True)
        self.key(VK_LWIN, injected=True)
        self.drain()
        self.assertEqual(self.calls, ["press"])

    def test_negative_code_passed_straight_on(self):
        self.assertEqual(self.hk._proc(-1, 0x100, 0), 7)
        self.drain()
        self.assertEqual(self.calls, [])

    def test_error_in_decision_still_passes_key_on(self):
        self.hk._chord.feed = mock.Mock(side_effect=RuntimeError("boom"))
        with self.assertLogs("scribe.winkeys", "ERROR"):
            self.key(VK_LWIN)
            self.drain()

    def test_watchdog_releases_a_hold_the_hook_lost(self):
        self.key(VK_LCONTROL)
        self.key(VK_LWIN)
        phys = {VK_LCONTROL: False, VK_LWIN: False}                 # both key-ups went to an elevated window
        self.hk._chord.physical = lambda vk: phys.get(vk, False)
        self.hk._watch()
        self.drain()
        self.assertEqual(self.calls, ["press", "release"])

    def test_worker_runs_the_watchdog_between_events(self):
        self.key(VK_LCONTROL)
        self.key(VK_LWIN)
        self.hk._chord.physical = lambda vk: False
        worker = threading.Thread(target=self.hk._work, daemon=True)
        worker.start()
        deadline = time.monotonic() + 2
        while "release" not in self.calls and time.monotonic() < deadline:
            time.sleep(0.02)
        self.hk._events.put(None)
        worker.join(2)
        self.assertEqual(self.calls, ["press", "release"])

    def test_callback_error_does_not_stop_worker(self):
        self.hk.on_press = mock.Mock(side_effect=RuntimeError("boom"))
        self.key(VK_LCONTROL)
        self.key(VK_LWIN)
        self.key(VK_LWIN, down=False)
        with self.assertLogs("scribe.winkeys", "ERROR"):
            self.drain()
        self.assertEqual(self.calls, ["release"])


class RealHook(unittest.TestCase):
    """Install and remove the real WH_KEYBOARD_LL hook on its own thread. Sends no keys."""

    def make(self):
        return HoldHotkey(lambda: None, lambda: None, lambda: None)

    def test_start_stop(self):
        hk = self.make()
        hk.start()
        try:
            self.assertTrue(hk.running)
            self.assertIn(id(hk), winkeys._LIVE_PROCS)
            self.assertNotEqual(hk._thread.ident, threading.main_thread().ident)
            self.assertTrue(hk._thread.daemon)
        finally:
            hk.stop()
        self.assertFalse(hk.running)
        self.assertIsNone(hk._thread)
        self.assertNotIn(id(hk), winkeys._LIVE_PROCS)

    def test_restart_and_idempotent(self):
        hk = self.make()
        hk.stop()                                            # before start: no-op
        for _ in range(3):
            hk.start()
            hk.start()                                       # second start: no-op
            self.assertTrue(hk.running)
            hk.stop()
            hk.stop()
            self.assertFalse(hk.running)
        self.assertEqual([t.name for t in threading.enumerate() if t.name.startswith("scribe-hotkey")], [])

    def test_two_hooks_side_by_side(self):
        a, b = self.make(), self.make()
        a.start()
        b.start()
        try:
            self.assertTrue(a.running and b.running)
        finally:
            a.stop()
            b.stop()

    def test_stop_returns_quickly(self):
        hk = self.make()
        hk.start()
        t = time.monotonic()
        hk.stop()
        self.assertLess(time.monotonic() - t, 1.0)

    def test_install_failure_raises(self):
        hk = self.make()
        with mock.patch.object(winkeys, "_SetWindowsHookExW", return_value=None):
            with self.assertRaises(OSError):
                hk.start()
        self.assertFalse(hk.running)
        self.assertIsNone(hk._thread)


if __name__ == "__main__":
    unittest.main()
