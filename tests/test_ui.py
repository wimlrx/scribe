"""ui.py: the pure parts, and the overlay's rules driven through fake windows. No Tk window, tray icon or
dialog is ever created here.

Covers clock and level formatting, the picker countdown, tray state and labels, layout maths, the icon
glyph and the .ico file, the cross-thread call queue and the Tk-thread routing decorator, and the widget and
picker rules: pointer zones, when a control is live, clicks from the windows' own messages, placement.
"""
from __future__ import annotations

import math
import sys
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging  # noqa: E402

from PIL import Image  # noqa: E402

from scribe import glass, ui  # noqa: E402

logging.getLogger("scribe").addHandler(logging.NullHandler())


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


class FakeClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class TestImportSideEffects(unittest.TestCase):
    def test_import_creates_no_tk_root(self):
        import tkinter

        self.assertIsNone(tkinter._default_root)

    def test_pystray_not_loaded_until_start(self):
        tray = ui.Tray(lambda: None, lambda: None, lambda: None, lambda: None, lambda: None)
        tray.set_state("recording", 2)   # no icon yet: remembered, nothing shown
        tray.notify("Scribe", "x")
        tray.stop()
        self.assertEqual((tray._state, tray._count), ("recording", 2))
        self.assertIsNone(tray._icon)

    def test_win32_functions_declare_types(self):
        for f in ui.WIN32_FUNCTIONS:
            self.assertIsNotNone(f.argtypes, f.__name__)
            self.assertIsNotNone(f.restype, f.__name__)


class TestElapsed(unittest.TestCase):
    def test_formats(self):
        cases = {0: "0:00", 5: "0:05", 59.9: "0:59", 60: "1:00", 754: "12:34", 3599: "59:59",
                 3600: "1:00:00", 5025: "1:23:45", 36000: "10:00:00", -3: "0:00"}
        for secs, text in cases.items():
            self.assertEqual(ui.fmt_elapsed(secs), text, secs)


class TestLevels(unittest.TestCase):
    def test_unit_scale(self):
        self.assertEqual(ui.level_to_unit(0.0), 0.0)
        self.assertEqual(ui.level_to_unit(-1.0), 0.0)
        self.assertAlmostEqual(ui.level_to_unit(1.0), 1.0)
        self.assertAlmostEqual(ui.level_to_unit(2.0), 1.0)
        self.assertEqual(ui.level_to_unit(10 ** (-60 / 20)), 0.0)   # below the floor
        quiet, loud = ui.level_to_unit(0.01), ui.level_to_unit(0.1)
        self.assertTrue(0.0 < quiet < loud < 1.0)

    def test_bars_shape_and_floor(self):
        bars = ui.level_bars(0.3)
        self.assertEqual(len(bars), 5)
        self.assertEqual(bars[2], max(bars))
        self.assertEqual(bars[0], bars[4])
        self.assertEqual(bars[1], bars[3])
        self.assertTrue(all(0.0 <= b <= 1.0 for b in bars))
        self.assertEqual(ui.level_bars(0.0), [0.15] * 5)
        self.assertEqual(len(ui.level_bars(0.3, 7)), 7)
        self.assertEqual(ui.bar_shape(1), [1.0])

    def test_bars_grow_with_level(self):
        self.assertLess(sum(ui.level_bars(0.01)), sum(ui.level_bars(0.2)))

    def test_smooth_rises_fast_falls_slow(self):
        up = ui.smooth(0.0, 1.0)
        down = 1.0 - ui.smooth(1.0, 0.0)
        self.assertGreater(up, down)
        self.assertEqual(ui.smooth(0.5, 0.5), 0.5)


class TestCountdown(unittest.TestCase):
    def test_counts_down_and_expires(self):
        clock = FakeClock()
        cd = ui.Countdown(20, clock=clock)
        self.assertEqual(cd.remaining(), 20)
        self.assertFalse(cd.expired)
        clock.t += 0.2
        self.assertEqual(cd.remaining(), 20)      # rounds up: the number shown is never 0 early
        clock.t += 0.8
        self.assertEqual(cd.remaining(), 19)
        clock.t += 18.5
        self.assertEqual(cd.remaining(), 1)
        self.assertFalse(cd.expired)
        clock.t += 0.5
        self.assertEqual(cd.remaining(), 0)
        self.assertTrue(cd.expired)
        clock.t += 5
        self.assertEqual(cd.remaining(), 0)

    def test_zero_and_negative_timeouts_expire_at_once(self):
        clock = FakeClock()
        self.assertTrue(ui.Countdown(0, clock=clock).expired)
        self.assertTrue(ui.Countdown(-5, clock=clock).expired)
        self.assertEqual(ui.Countdown(-5, clock=clock).remaining(), 0)


class TestTrayState(unittest.TestCase):
    def test_priority(self):
        self.assertEqual(ui.pick_state(False, False, 0), ("idle", 0))
        self.assertEqual(ui.pick_state(False, False, 2), ("working", 2))
        self.assertEqual(ui.pick_state(False, True, 0), ("recording", 0))
        self.assertEqual(ui.pick_state(False, True, 3), ("recording", 3))
        self.assertEqual(ui.pick_state(True, True, 3), ("dictating", 3))
        self.assertEqual(ui.pick_state(True, False, 0), ("dictating", 0))
        self.assertEqual(ui.pick_state(False, False, -1), ("idle", 0))

    def test_labels(self):
        self.assertEqual(ui.tray_title("idle"), "Scribe")
        self.assertEqual(ui.tray_title("working", 3), "Scribe · 3")
        self.assertEqual(ui.tray_title("working", 1, waiting=True), "Scribe · 1 · ⏳ GPU")
        self.assertEqual(ui.toggle_text(False), "Start notetaker")
        self.assertEqual(ui.toggle_text(True), "Stop notetaker")
        self.assertEqual(ui.menu_labels(False),
                         ["Start notetaker", "Add file…", "Retry", "Send", "-", "Recordings", "Inbox", "-", "Quit"])
        self.assertEqual(ui.menu_labels(True)[0], "Stop notetaker")

    def test_labels_are_short(self):
        for label in ui.menu_labels(True) + [ui.tray_title("working", 12)]:
            words = [w for w in label.split() if w.isalpha()]
            self.assertLessEqual(len(words), 2, label)      # labels and counts, never a sentence
            self.assertNotIn(".", label)

    def test_recording_flag_comes_from_the_app(self):
        flag = {"on": True}
        tray = ui.Tray(*[lambda: None] * 5, is_recording=lambda: flag["on"])
        self.assertTrue(tray._recording())
        flag["on"] = False
        self.assertFalse(tray._recording())
        plain = ui.Tray(*[lambda: None] * 5)
        plain.set_state("recording")
        self.assertTrue(plain._recording())
        plain.set_state("nonsense")
        self.assertEqual(plain._state, "idle")

    def test_notices_wait_for_the_icon_then_go_in_order(self):
        class Icon:
            def __init__(self):
                self.visible, self.sent = False, []

            def notify(self, message, title):
                assert self.visible, "a notice before NIM_ADD is dropped by Windows"
                self.sent.append((title, message))

        tray = ui.Tray(*[lambda: None] * 5)
        tray.notify("Scribe", "✕ Ctrl+Win")          # before start: the icon does not exist yet
        icon = Icon()
        tray._icon = icon                            # run_detached has returned, the setup has not run
        tray.notify("Scribe", "✓ Meeting 09:00 · Local")
        self.assertEqual(icon.sent, [])
        tray._setup(icon)
        self.assertTrue(icon.visible)
        self.assertEqual(icon.sent, [("Scribe", "✕ Ctrl+Win"), ("Scribe", "✓ Meeting 09:00 · Local")])
        tray.notify("Scribe", "⚠ Disk")
        self.assertEqual(icon.sent[-1], ("Scribe", "⚠ Disk"))

    def test_notice_queue_is_bounded(self):
        tray = ui.Tray(*[lambda: None] * 5)
        for i in range(30):
            tray.notify("Scribe", str(i))
        self.assertEqual(len(tray._notes), ui.Tray.NOTICE_QUEUE)
        self.assertEqual(tray._notes[-1], ("Scribe", "29"))

    def test_send_menu_items(self):
        sent = []
        tray = ui.Tray(*[lambda: None] * 5, sendable=lambda: [("Meeting 14:05", Path("C:/x"))],
                       vaults=lambda: ["Work", "Personal"], on_send=lambda f, v: sent.append((f, v)))
        self.assertEqual(tray.send_items(), [("Meeting 14:05", Path("C:/x"), ["Work", "Personal"])])
        tray._send(Path("C:/x"), "Personal")(None, None)
        self.assertEqual(sent, [(Path("C:/x"), "Personal")])
        self.assertEqual(ui.Tray(*[lambda: None] * 5).send_items(), [])
        no_vaults = ui.Tray(*[lambda: None] * 5, sendable=lambda: [("Meeting 14:05", Path("C:/x"))],
                            vaults=lambda: [], on_send=lambda f, v: sent.append((f, v)))
        self.assertEqual(no_vaults.send_items(), [])        # no vaults: no Send menu, not an empty one

    def test_failed_flag(self):
        flag = {"on": False}
        tray = ui.Tray(*[lambda: None] * 5, on_retry=lambda: None, has_failed=lambda: flag["on"])
        self.assertFalse(tray._failed())
        flag["on"] = True
        self.assertTrue(tray._failed())


class TestUnattendedPick(unittest.TestCase):
    def test_expiry_never_picks_a_vault(self):
        self.assertEqual(ui.unattended_pick(["Work", "Personal", "Local"]), "Local")
        self.assertEqual(ui.unattended_pick(["Local", "Work"]), "Local")


class TestTrayActions(unittest.TestCase):
    def test_tray_action_swallows_errors(self):
        def boom():
            raise RuntimeError("x")
        ui.Tray._act(boom)(None, None)   # must not raise into pystray's loop


class TestLayout(unittest.TestCase):
    def test_layout_row(self):
        self.assertEqual(ui.layout_row([10, 20, 5], 3, 2), [(3, 13), (15, 35), (37, 42)])
        self.assertEqual(ui.layout_row([], 3, 2), [])

    def test_positions(self):
        work = (0, 0, 1920, 1040)
        self.assertEqual(ui.bottom_centre(work, 140, 36, 24), (890, 980))
        self.assertEqual(ui.top_right(work, 160, 36, 16), (1744, 16))
        self.assertEqual(ui.top_right(work, 200, 36, 16, row=1, row_h=36, gap=8), (1704, 60))
        self.assertEqual(ui.bottom_right(work, 140, 36, 0), (1780, 1004))
        offset = (100, 50, 2020, 1090)      # a work area that does not start at 0,0
        self.assertEqual(ui.bottom_centre(offset, 140, 36, 24), (990, 1030))
        self.assertEqual(ui.bottom_right(offset, 140, 36, 0), (1880, 1054))
        self.assertEqual(ui.top_right(offset, 160, 36, 16), (1844, 66))

    def test_work_area_is_a_rectangle(self):
        left, top, right, bottom = ui.work_area()
        self.assertLess(left, right)
        self.assertLess(top, bottom)

    def test_pill_styles(self):
        base = ui.pill_exstyle(False)
        for flag in (ui.WS_EX_NOACTIVATE, ui.WS_EX_TOOLWINDOW, ui.WS_EX_TOPMOST, ui.WS_EX_LAYERED):
            self.assertTrue(base & flag)
        self.assertFalse(base & ui.WS_EX_TRANSPARENT)
        self.assertTrue(ui.pill_exstyle(True) & ui.WS_EX_TRANSPARENT)


class TestIcon(unittest.TestCase):
    def test_glyph_colour_per_state(self):
        for state, colour in ui.STATE_COLOURS.items():
            img = ui.draw_icon(state, 0, 64)
            self.assertEqual(img.size, (64, 64))
            self.assertEqual(img.mode, "RGBA")
            r, g, b, a = img.getpixel((32, 18))          # middle of the capsule
            self.assertEqual(a, 255, state)
            for got, want in zip((r, g, b), _rgb(colour)):
                self.assertLessEqual(abs(got - want), 2, state)
            self.assertEqual(img.getpixel((1, 1))[3], 0)  # transparent background

    def test_states_differ(self):
        seen = {ui.draw_icon(s, 0, 32).tobytes() for s in ui.STATES}
        self.assertEqual(len(seen), len(ui.STATES))

    def test_count_badge(self):
        plain = ui.draw_icon("working", 0, 64)
        one = ui.draw_icon("working", 1, 64)
        many = ui.draw_icon("working", 12, 64)
        self.assertNotEqual(plain.tobytes(), one.tobytes())
        self.assertNotEqual(one.tobytes(), many.tobytes())
        self.assertGreater(one.getpixel((56, 56))[3], 0)   # badge sits bottom right
        self.assertEqual(plain.getpixel((62, 40))[3], 0)

    def test_unknown_state_is_grey(self):
        self.assertEqual(ui.draw_icon("bogus").tobytes(), ui.draw_icon("idle").tobytes())

    def test_write_ico(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = ui.write_ico(Path(tmp) / "sub" / "scribe.ico")
            self.assertTrue(path.is_file())
            with Image.open(path) as ico:
                self.assertEqual(ico.format, "ICO")
                sizes = set(ico.info["sizes"])
            for n in (16, 32, 48, 256):
                self.assertIn((n, n), sizes)


class TestCallQueue(unittest.TestCase):
    def test_runs_in_order_and_isolates_failures(self):
        q = ui.CallQueue()
        out = []
        q.put(out.append, 1)
        q.put(lambda: 1 / 0)
        q.put(out.append, 2)
        q.put(dict.update, {}, a=1)
        self.assertEqual(q.run_pending(), 4)
        self.assertEqual(out, [1, 2])
        self.assertEqual(q.run_pending(), 0)

    def test_limit(self):
        q = ui.CallQueue()
        out = []
        for i in range(5):
            q.put(out.append, i)
        self.assertEqual(q.run_pending(limit=3), 3)
        self.assertEqual(out, [0, 1, 2])
        self.assertEqual(q.run_pending(), 2)
        self.assertEqual(out, [0, 1, 2, 3, 4])

    def test_threads_can_put(self):
        q = ui.CallQueue()
        out = []
        threads = [threading.Thread(target=lambda i=i: q.put(out.append, i)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        q.run_pending()
        self.assertEqual(sorted(out), list(range(20)))


class Routed:
    """Stands in for Overlay: owns a thread and a call queue, nothing else."""

    def __init__(self):
        self._thread = threading.current_thread()
        self._calls = ui.CallQueue()
        self.log = []

    @ui.on_tk
    def act(self, value, extra=None):
        self.log.append((threading.current_thread().name, value, extra))
        return value


class TestOnTk(unittest.TestCase):
    def test_inline_on_owner_thread(self):
        r = Routed()
        self.assertEqual(r.act(1, extra="x"), 1)
        self.assertEqual(r.log, [(threading.current_thread().name, 1, "x")])

    def test_queued_from_other_threads(self):
        r = Routed()
        t = threading.Thread(target=lambda: r.act(2, extra="y"), name="worker")
        t.start()
        t.join()
        self.assertEqual(r.log, [])                  # nothing ran off the owner thread
        self.assertEqual(r._calls.run_pending(), 1)
        self.assertEqual(r.log, [(threading.current_thread().name, 2, "y")])

    def test_overlay_view_choice(self):
        ov = ui.Overlay.__new__(ui.Overlay)        # no Tk: the view rule only reads these fields
        ov._mode, ov._recording, ov._over, ov._picker = None, False, False, None
        ov._confirm_until = ov._error_until = 0.0
        ov.idle_pill, ov._fullscreen, ov.on_toggle = True, False, object()
        self.assertEqual(ov._target_view(10.0), "rest")
        ov._over = True
        self.assertEqual(ov._target_view(10.0), "start")
        ov._recording, ov._over = True, False
        ov._confirm_until = 12.5
        self.assertEqual(ov._target_view(10.0), "rec")             # confirmation after a start
        self.assertEqual(ov._target_view(13.0), "rest")            # then it folds away
        ov._over = True
        self.assertEqual(ov._target_view(13.0), "rec")             # pointer over it: time and stop
        ov._mode = "listening"
        self.assertEqual(ov._target_view(13.0), "listening")       # dictation wins
        ov._mode, ov._error_until = "error", 14.0
        self.assertEqual(ov._target_view(13.0), "error")
        self.assertEqual(ov._target_view(14.5), "rec")             # the error flash ends by itself
        ov._recording, ov._over, ov._fullscreen = False, False, True
        self.assertIsNone(ov._target_view(20.0))                    # a full-screen app hides the resting pill
        ov._fullscreen, ov.idle_pill = False, False
        self.assertIsNone(ov._target_view(20.0))
        ov._recording = True
        self.assertEqual(ov._target_view(20.0), "rest")            # the recording dot shows anyway

    def test_clicks(self):
        c = ui.Clicks()
        seq = [(False, "start"), (True, "start"), (True, "start"), (False, "start")]
        self.assertEqual([c.feed(d, o) for d, o in seq], [None, None, None, "start"])
        c = ui.Clicks()
        self.assertEqual([c.feed(d, o) for d, o in [(True, "start"), (False, None)]], [None, None])  # moved off
        c = ui.Clicks()
        self.assertEqual([c.feed(d, o) for d, o in [(True, None), (False, "stop")]], [None, None])  # dragged on
        c = ui.Clicks()
        self.assertEqual([c.feed(d, o) for d, o in [(True, 0), (False, 0)]], [None, 0])  # chip 0 counts
        c = ui.Clicks()
        self.assertEqual([c.feed(d, o) for d, o in [(True, "start"), (True, "stop"), (False, "stop")]],
                         [None, None, None])                       # pressed on one control, released on another

    def test_pulse_range(self):
        self.assertTrue(all(0.0 <= ui.pulse(t / 10) <= 1.0 for t in range(50)))

    def test_overlay_methods_are_routed(self):
        for name in ("show_listening", "show_working", "hide", "flash_error", "show_recording",
                     "hide_recording", "pick_vault", "ask_files", "close"):
            self.assertTrue(hasattr(getattr(ui.Overlay, name), "__wrapped__"), name)


# ---------------------------------------------------------------- the overlay, driven through fake windows

NAMES = ["Work", "Personal", "Local"]


class FakeTop:
    """Stands in for the Tk toplevel: keeps what was bound to it."""

    def __init__(self):
        self.bindings = {}

    def bind(self, sequence, fn):
        self.bindings[sequence] = fn


class FakeWin:
    """Stands in for _Layered: remembers where the last frame went. The hit test is the real one."""

    contains = ui._Layered.contains

    def __init__(self, root=None, click_through=False):
        self.top = FakeTop()
        self.x = self.y = ui.OFFSCREEN
        self.size, self.visible, self.presents = (0, 0), False, 0
        self.frame: Image.Image | None = None

    def present(self, img, x, y):
        self.x, self.y, self.size, self.visible, self.frame = x, y, img.size, True, img
        self.presents += 1
        return True

    def hide(self):
        self.visible = False

    def destroy(self):
        pass


def fire(win: FakeWin, pt) -> None:
    """A press and release delivered to the window's own bindings, with no frame between them."""
    event = SimpleNamespace(x_root=pt[0], y_root=pt[1])
    win.top.bindings["<ButtonPress-1>"](event)
    win.top.bindings["<ButtonRelease-1>"](event)


class OverlayCase(unittest.TestCase):
    """An Overlay without Tk: fake windows, a recorder in place of _spawn, a fake clock and work area."""

    def setUp(self):
        self.clock = FakeClock(1000.0)
        self.area = [(0, 0, 2000, 1200)]
        for p in (mock.patch.object(ui, "time", SimpleNamespace(monotonic=self.clock)),
                  mock.patch.object(ui, "work_area", lambda: self.area[0])):
            p.start()
            self.addCleanup(p.stop)

    def make(self, scale: float = 1.0) -> ui.Overlay:
        ov = ui.Overlay.__new__(ui.Overlay)
        ov.root, ov._thread, ov._calls = None, threading.current_thread(), ui.CallQueue()
        ov.scale, ov.idle_pill, ov.on_toggle, ov.anchor = scale, True, (lambda: None), "right"
        ov._win, ov._pick_win = FakeWin(), FakeWin()
        ov._mode, ov._token, ov._level_fn, ov._level, ov._error_until = None, 0, None, 0.0, 0.0
        ov._recording, ov._elapsed_fn, ov._on_stop, ov._confirm_until = False, None, None, 0.0
        ov._view = "rest"
        ov._w, ov._h = (float(v) for v in glass.SIZES["rest"])
        ov._content, ov._opacity, ov._over, ov._hover = 1.0, 0.0, False, None
        ov._last_click, ov._clicks, ov._pick_clicks = 0.0, ui.Clicks(), ui.Clicks()
        ov._drawn, ov._phase, ov._last, ov._fullscreen, ov._fs_checked = None, 0.0, 0.0, False, 0.0
        ov._picker, ov._picks, ov._pick_hover, ov._pick_drawn, ov._closed = None, deque(), None, None, False
        ov.spawned = []
        ov._spawn = lambda fn, what: ov.spawned.append(fn)
        return ov

    def step(self, ov, pt, n: int = 1, dt: float = 0.016) -> list[str]:
        views = []
        for _ in range(n):
            self.clock.t += dt
            ov._widget_step(self.clock(), dt, pt)
            views.append(ov._view)
        return views

    def settle(self, ov, pt, view: str) -> None:
        self.step(ov, None)                                   # the resting pill is on screen
        for _ in range(60):
            self.step(ov, pt)
            if ov._view == view and ov._content >= 1.0:
                self.step(ov, pt)                             # hover is read at the start of a frame
                return
        self.fail(f"never settled on {view}")

    def at(self, ov, view: str, fx: float = 0.5, fy: float = 0.5, dx: float = 0.0, dy: float = 0.0):
        """A screen point on the pill of `view` (fx, fy across it, then dx, dy logical px), wherever the
        current work area puts the widget."""
        wx, wy = ov._widget_origin(self.area[0], *glass.widget_canvas(ov.scale))
        x0, y0, x1, y1 = glass.widget_rect(ov.scale, *glass.SIZES[view])
        return (int(wx + x0 + (x1 - x0) * fx + dx * ov.scale), int(wy + y0 + (y1 - y0) * fy + dy * ov.scale))

    def stop_point(self, ov):
        wx, wy = ov._widget_origin(self.area[0], *glass.widget_canvas(ov.scale))
        sx, sy = glass.rec_layout(glass.widget_rect(ov.scale, *glass.SIZES["rec"]), ov.scale)["stop"]
        return int(wx + sx), int(wy + sy)

    @staticmethod
    def chip(ov, i: int):
        x0, y0, x1, y1 = glass.picker_layout(ov.scale, NAMES)["chips"][i]
        return int(ov._pick_win.x + (x0 + x1) / 2), int(ov._pick_win.y + (y0 + y1) / 2)

    @staticmethod
    def tap(ov, pt) -> None:
        ov._widget_button(True, pt)
        ov._widget_button(False, pt)

    @staticmethod
    def tap_picker(ov, pt) -> None:
        ov._picker_button(True, pt)
        ov._picker_button(False, pt)


class TestOverlayRules(OverlayCase):
    """The rules the widget and picker keep, called directly (SAFE-4 and the click rules)."""

    def test_expiry_settles_on_local_not_the_highlighted_vault(self):
        ov, got = self.make(), []
        ov.pick_vault(NAMES, "Work", 20, got.append)
        ov._picker["countdown"] = ui.Countdown(20, clock=self.clock)
        ov._picker_step(self.clock(), None)
        self.clock.t += 19.5
        ov._picker_step(self.clock(), None)
        self.assertEqual(got, [])
        self.clock.t += 0.5
        ov._picker_step(self.clock(), None)
        self.assertEqual(got, ["Local"])
        self.assertIsNone(ov._picker)

    def test_a_chip_click_picks_that_chip(self):
        for i, name in enumerate(NAMES):
            ov, got = self.make(1.5), []
            ov.pick_vault(NAMES, "Work", 20, got.append)
            ov._picker_step(self.clock(), None)               # drawn
            self.tap_picker(ov, self.chip(ov, i))
            self.assertEqual(got, [name])

    def test_a_press_for_one_request_never_picks_for_the_next(self):
        ov, got = self.make(), []
        ov.pick_vault(NAMES, "Work", 20, got.append)
        ov.pick_vault(NAMES, "Work", 20, got.append)    # a second meeting waits its turn
        ov._picker_step(self.clock(), None)
        pt = self.chip(ov, 1)
        ov._picker_button(True, pt)                           # pressed on the first request's Personal
        ov._picker["countdown"] = ui.Countdown(0)             # as its countdown runs out
        ov._picker_step(self.clock(), None)                   # first settles on Local; the second is up
        ov._picker_step(self.clock(), None)                   # drawn at the same spot
        ov._picker_button(False, pt)                          # the release lands on the second's Personal
        self.assertEqual(got, ["Local"])
        self.tap_picker(ov, pt)
        self.assertEqual(got, ["Local", "Personal"])

    def test_start_only_when_idle_stop_only_when_recording(self):
        ov = self.make()
        stop = lambda: None  # noqa: E731
        ov._on_stop = stop
        ov._click("stop")
        self.assertEqual(ov.spawned, [])
        self.clock.t += 1
        ov._recording = True
        ov._click("start")
        self.assertEqual(ov.spawned, [])
        self.clock.t += 1
        ov._click("stop")
        self.assertEqual(ov.spawned, [stop])
        self.clock.t += 1
        ov._recording = False
        ov._click("start")
        self.assertEqual(ov.spawned, [stop, ov.on_toggle])

    def test_a_second_click_within_the_guard_is_ignored(self):
        ov = self.make()
        ov._click("start")
        self.clock.t += ui.CLICK_GUARD_S / 2
        ov._click("start")
        self.assertEqual(len(ov.spawned), 1)
        self.clock.t += ui.CLICK_GUARD_S
        ov._click("start")
        self.assertEqual(len(ov.spawned), 2)

    def test_no_control_is_live_during_dictation(self):
        ov = self.make()
        pt = self.at(ov, "start")
        self.settle(ov, pt, "start")
        self.assertEqual(ov._hover, "start")                  # the point is on the button
        ov.show_listening(lambda: 0.2)
        ov._track_pointer(pt)
        self.assertEqual((ov._over, ov._hover), (False, None))
        self.tap(ov, pt)
        self.step(ov, pt, 10)
        self.tap(ov, pt)
        self.assertEqual(ov._view, "listening")
        self.assertIsNone(ov._hover)
        self.assertEqual(ov.spawned, [])

    def test_the_confirmation_lasts_confirm_s(self):
        ov = self.make()
        ov.show_recording(lambda: 0.0, lambda: None)
        self.assertEqual(ov._target_view(self.clock() + ui.CONFIRM_S - 0.05), "rec")
        self.assertEqual(ov._target_view(self.clock() + ui.CONFIRM_S + 0.05), "rest")

    def test_hide_recording_drops_the_stop_callback(self):
        ov = self.make()
        ov.show_recording(lambda: 0.0, lambda: None)
        ov.hide_recording()
        self.assertIsNone(ov._on_stop)
        self.assertFalse(ov._recording)
        ov._click("stop")
        self.assertEqual(ov.spawned, [])


class TestOverlayPointer(OverlayCase):
    """Where the pointer counts, when a control is live, and where the windows go (GLASS-1 to GLASS-5)."""

    def test_picker_clears_the_widget_at_its_tallest(self):
        for scale in (1.0, 1.25, 1.5, 1.75, 2.0):
            for area in ((0, 0, 1920, 1040), (0, 0, 2560, 1568)):
                self.area[0] = area
                ov = self.make(scale)
                ov.pick_vault(NAMES, "Work", 20, lambda _n: None)
                ov._picker_step(self.clock(), None)
                chips_bottom = max(ov._pick_win.y + c[3] for c in glass.picker_layout(scale, NAMES)["chips"])
                wx, wy = ov._widget_origin(area, *glass.widget_canvas(scale))
                self.assertGreaterEqual(ov._pick_win.x, area[0], (scale, area))                  # on screen
                self.assertLessEqual(ov._pick_win.x + ov._pick_win.size[0], area[2], (scale, area))
                cw, ch = glass.widget_canvas(scale)
                for view in ("start", "rec"):
                    w, h = glass.SIZES[view]
                    top = next(y for y in range(ch) if any(glass.widget_hit(scale, w, h, view, x, y)
                                                           for x in range(cw)))
                    self.assertLess(chips_bottom, wy + top, (scale, area, view))   # no chip over a button

    def test_no_control_sits_under_the_pickers_shadow(self):
        """A press goes to the topmost window whose pixel there takes clicks (alpha >= CLICK_ALPHA). With the
        picker up, every start and stop pixel must still be the widget's, or its hover ring promises a dead
        click."""
        # custom scales too: at 12 px, 107%, 157%, 207% and 232% left the top row of the start circle under the shadow
        for scale in (1.0, 1.07, 1.25, 1.32, 1.5, 1.57, 1.75, 1.82, 2.0, 2.07, 2.25, 2.32, 2.5):
            for area in ((0, 0, 1920, 1040), (0, 0, 2560, 1568)):
                self.area[0] = area
                ov = self.make(scale)
                ov.pick_vault(NAMES, "Work", 20, lambda _n: None)
                ov._picker_step(self.clock(), None)
                pick = ov._pick_win
                alpha = pick.frame.getchannel("A").load()
                wx, wy = ov._widget_origin(area, *glass.widget_canvas(scale))
                cw, ch = glass.widget_canvas(scale)
                rows = range(max(0, pick.y - wy), min(ch, pick.y + pick.size[1] - wy))   # where the two overlap
                for view, control in (("start", "start"), ("rec", "stop")):
                    w, h = glass.SIZES[view]
                    under = [(x, y) for y in rows for x in range(cw)
                             if pick.contains((wx + x, wy + y)) is not None
                             and alpha[wx + x - pick.x, wy + y - pick.y] >= glass.CLICK_ALPHA
                             and glass.widget_hit(scale, w, h, view, x, y) == control]
                    self.assertEqual(under, [], (scale, area, view))

    def test_the_widget_gives_way_to_a_chip(self):
        ov, got = self.make(), []
        pt = self.at(ov, "start")
        self.settle(ov, pt, "start")
        self.assertEqual(ov._hover, "start")
        ov.pick_vault(NAMES, "Work", 20, got.append)
        x0, y0, x1, y1 = glass.picker_layout(1.0, NAMES)["chips"][1]
        ov._pick_win.present(glass.picker(1.0, NAMES, "Work", None, 20, 1.0),
                             int(pt[0] - (x0 + x1) / 2), int(pt[1] - (y0 + y1) / 2))   # a chip over the button
        ov._track_pointer(pt)
        self.assertIsNone(ov._hover)
        self.tap(ov, pt)                                      # whichever window the click lands on,
        self.tap_picker(ov, pt)                               # it does one thing
        self.assertEqual((ov.spawned, got), ([], ["Personal"]))

    def test_start_takes_no_click_until_its_dot_is_drawn(self):
        ov = self.make(1.5)
        pt = self.at(ov, "rest", dy=-8)                       # just above the resting pill
        self.step(ov, None)
        seen = []
        for _ in range(40):
            before = ov._content
            self.step(ov, pt)
            seen.append((ov._view, before, ov._hover))
            if ov._view == "start" and ov._content < ui.LIVE_CONTENT:
                self.tap(ov, pt)
        self.assertTrue(any(v == "start" and c < ui.LIVE_CONTENT for v, c, _h in seen))
        for view, content, hover in seen:
            if content < ui.LIVE_CONTENT:
                self.assertIsNone(hover, (view, content))
        self.assertEqual(ov.spawned, [])
        self.assertEqual(ov._hover, "start")                  # drawn: now it is a button
        self.tap(ov, pt)
        self.assertEqual(ov.spawned, [ov.on_toggle])

    def test_stop_takes_no_click_while_the_pill_grows(self):
        ov = self.make()
        ov._recording, ov._on_stop = True, (lambda: None)     # a recording runs; its confirmation is over
        pt = self.at(ov, "rest", dx=24, dy=-8)                # over the clock once the rec view is drawn
        self.step(ov, None)
        seen = []
        for _ in range(40):
            before = ov._content
            self.step(ov, pt)
            seen.append((ov._view, before, ov._hover))
            self.tap(ov, pt)
        self.assertTrue(any(v == "rec" and c < ui.LIVE_CONTENT for v, c, _h in seen))
        for view, content, hover in seen:
            if content < ui.LIVE_CONTENT:
                self.assertIsNone(hover, (view, content))
        self.assertEqual(ov._view, "rec")
        self.assertIsNone(ov._hover)
        self.assertEqual(ov.spawned, [])

    def test_a_tap_between_two_frames_starts(self):
        ov = self.make(1.25)
        pt = self.at(ov, "start")
        self.settle(ov, pt, "start")
        self.tap(ov, pt)                                      # press and release, no frame between
        self.assertEqual(ov.spawned, [ov.on_toggle])

    def test_a_tap_between_two_frames_stops(self):
        ov = self.make(1.25)
        stop = lambda: None  # noqa: E731
        ov.show_recording(lambda: 3.0, stop)
        pt = self.stop_point(ov)
        self.settle(ov, pt, "rec")
        self.assertEqual(ov._hover, "stop")
        self.tap(ov, pt)
        self.assertEqual(ov.spawned, [stop])

    def test_a_tap_is_hit_tested_at_its_own_point(self):
        """The click is judged where it lands, not where the last frame saw the pointer."""
        stop = lambda: None  # noqa: E731
        ov = self.make()
        ov.show_recording(lambda: 3.0, stop)
        clock_pt = self.at(ov, "rec", fx=0.35)                # over the clock: not a control
        self.settle(ov, clock_pt, "rec")
        self.assertIsNone(ov._hover)
        self.tap(ov, self.stop_point(ov))                     # moved onto stop and tapped before the next frame
        self.assertEqual(ov.spawned, [stop])
        ov = self.make()
        ov.show_recording(lambda: 3.0, stop)
        self.settle(ov, self.stop_point(ov), "rec")
        self.assertEqual(ov._hover, "stop")
        self.tap(ov, clock_pt)                                # moved off stop onto the clock, then tapped
        self.assertEqual(ov.spawned, [])

    def test_both_windows_take_clicks_from_their_own_messages(self):
        with mock.patch.object(ui, "_Layered", FakeWin):
            ov, got = self.make(), []
            ov._win = ov._pick_win = None                     # made by the overlay, as in the app
            pt = self.at(ov, "start")
            self.settle(ov, pt, "start")
            fire(ov._win, pt)
            self.assertEqual(ov.spawned, [ov.on_toggle])
            ov.pick_vault(NAMES, "Work", 20, got.append)
            ov._picker_step(self.clock(), None)
            fire(ov._pick_win, self.chip(ov, 1))
            self.assertEqual(got, ["Personal"])

    def test_bind_button_hands_on_screen_points_and_keeps_errors_from_tk(self):
        top, seen = FakeTop(), []
        ui.bind_button(top, lambda down, pt: seen.append((down, pt)))
        fire(SimpleNamespace(top=top), (812, 1534))
        self.assertEqual(seen, [(True, (812, 1534)), (False, (812, 1534))])

        def boom(_down, _pt):
            raise RuntimeError("x")

        ui.bind_button(top, boom)
        with self.assertLogs("scribe.ui", "ERROR"):
            fire(SimpleNamespace(top=top), (1, 2))            # must not raise into Tk

    def test_the_view_settles_with_the_pointer_beside_the_pill(self):
        for side in (1, -1):
            ov = self.make()
            pt = self.at(ov, "rest", fy=1.0, dx=30 * side, dy=-10)   # 30 px beside the centre, 10 up
            self.step(ov, None)
            views = self.step(ov, pt, 40)
            self.assertEqual(set(views[-20:]), {"start"}, (side, "".join(v[0] for v in views)))
            presents = ov._win.presents
            self.step(ov, pt, 10)
            self.assertEqual(ov._win.presents, presents, side)   # settled: nothing redrawn

    def test_band_edge_column_settles(self):
        """A pointer parked on the band's own edge column opens start and keeps it open."""
        for scale in (1.0, 1.5, 2.0):
            self.area[0] = (0, 0, 2560, 1568)                 # a real display
            for edge in ("right", "left"):
                ov = self.make(scale)
                self.step(ov, None)
                band = ov._band_screen_rect()
                x = int(band[2]) if edge == "right" else math.ceil(band[0])   # the last column inside
                views = self.step(ov, (x, int(band[3] - 10 * scale)), 80)
                self.assertEqual(set(views[-30:]), {"start"}, (scale, edge, "".join(v[0] for v in views[-30:])))

    def test_the_idle_pill_follows_the_work_area(self):
        ov = self.make(1.5)
        canvas = glass.widget_canvas(1.5)
        for area in ((0, 0, 1920, 1040), (0, 0, 1280, 680), (0, 0, 1280, 720), (1920, 0, 4480, 1400)):
            self.area[0] = area
            self.step(ov, None, 60)
            self.assertEqual(ov._view, "rest")
            self.assertEqual((ov._win.x, ov._win.y), ui.bottom_right(area, *canvas, 0), area)

    def test_the_widget_sits_above_the_clock_or_centred(self):
        canvas = glass.widget_canvas(1.25)
        area = (0, 0, 2560, 1528)
        for anchor, place in (("right", ui.bottom_right), ("centre", ui.bottom_centre)):
            ov = self.make(1.25)
            ov.anchor = anchor
            self.area[0] = area
            self.step(ov, None, 60)
            self.assertEqual((ov._win.x, ov._win.y), place(area, *canvas, 0), anchor)


if __name__ == "__main__":
    unittest.main()
