"""glass.py: the widget and picker frames, hit tests and the bytes handed to UpdateLayeredWindow."""
import unittest

import numpy as np
from PIL import Image

from scribe import glass


def alpha(img: Image.Image) -> np.ndarray:
    return np.asarray(img.getchannel("A"))


class TestWidget(unittest.TestCase):
    def test_every_view_draws_on_the_fixed_canvas(self):
        for scale in (1.0, 1.25, 1.5, 2.0):
            size = glass.widget_canvas(scale)
            for view, (w, h) in glass.SIZES.items():
                img = glass.widget(scale, w, h, view, heights=glass.listening_heights(0.5, 0.3), clock="0:07",
                                   recording=True, pulse=0.5)
                self.assertEqual(img.size, size, (scale, view))
                self.assertEqual(img.mode, "RGBA")
                self.assertGreater(alpha(img).max(), 150, (scale, view))
                self.assertEqual(alpha(img)[0, 0], 0)                    # corners stay clear

    def test_pill_grows_upward_from_a_fixed_bottom(self):
        small = glass.widget_rect(1.5, *glass.SIZES["rest"])
        big = glass.widget_rect(1.5, *glass.SIZES["listening"])
        self.assertEqual(small[3], big[3])
        self.assertLess(big[1], small[1])
        self.assertAlmostEqual((small[0] + small[2]) / 2, (big[0] + big[2]) / 2, delta=1)

    def test_rounded_edges_are_antialiased(self):
        img = glass.widget(1.5, 116, 34, "listening", heights=[0.5] * glass.BARS)
        a = alpha(img)
        partial = ((a > 20) & (a < 180)).sum()
        self.assertGreater(partial, 50)                                  # soft edge and shadow, not a hard cut

    def test_opacity_scales_the_frame(self):
        full = alpha(glass.widget(1.0, 44, 9, "rest")).astype(int).sum()
        faint = alpha(glass.widget(1.0, 44, 9, "rest", opacity=0.5)).astype(int).sum()
        self.assertAlmostEqual(faint / full, 0.5, delta=0.02)

    def test_content_follows_the_view(self):
        rest = glass.widget(1.5, 44, 9, "rest")
        dot = glass.widget(1.5, 44, 9, "rest", recording=True, pulse=1.0)
        self.assertNotEqual(rest.tobytes(), dot.tobytes())               # the copper dot while recording
        loud = glass.widget(1.5, 116, 34, "listening", heights=glass.listening_heights(0.9, 0.2))
        quiet = glass.widget(1.5, 116, 34, "listening", heights=glass.listening_heights(0.0, 0.2))
        self.assertGreater(alpha(loud).astype(int).sum(), alpha(quiet).astype(int).sum())
        hidden = glass.widget(1.5, 116, 34, "listening", heights=[1.0] * glass.BARS, content=0.0)
        shown = glass.widget(1.5, 116, 34, "listening", heights=[1.0] * glass.BARS, content=1.0)
        self.assertNotEqual(hidden.tobytes(), shown.tobytes())

    def test_clock_text_changes_the_frame(self):
        a = glass.widget(1.5, 128, 36, "rec", clock="0:01")
        b = glass.widget(1.5, 128, 36, "rec", clock="0:02")
        self.assertNotEqual(a.tobytes(), b.tobytes())

    def test_heights_stay_in_range(self):
        for level in (0.0, 0.3, 1.0, 5.0):
            for phase in (0.0, 0.7, 13.1):
                hs = glass.listening_heights(level, phase)
                self.assertEqual(len(hs), glass.BARS)
                self.assertTrue(all(0.0 <= v <= 1.0 for v in hs))
        quiet = glass.listening_heights(0.0, 0.5)
        self.assertGreater(min(quiet), 0.0)                              # never flat while listening
        self.assertTrue(all(0.0 <= v <= 1.0 for v in glass.working_heights(2.2)))


class TestHits(unittest.TestCase):
    def test_start_button(self):
        s = 1.5
        x0, y0, x1, y1 = glass.widget_rect(s, *glass.SIZES["start"])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.assertEqual(glass.widget_hit(s, 36, 36, "start", cx, cy), "start")
        self.assertIsNone(glass.widget_hit(s, 36, 36, "start", x0 - 20, cy))
        self.assertIsNone(glass.widget_hit(s, 36, 36, "rest", cx, cy))    # the resting pill has no control

    def test_stop_button_only_on_its_circle(self):
        s = 1.25
        rect = glass.widget_rect(s, 128, 36)
        sx, sy = glass.rec_layout(rect, s)["stop"]
        self.assertEqual(glass.widget_hit(s, 128, 36, "rec", sx, sy), "stop")
        dx, dy = glass.rec_layout(rect, s)["dot"]
        self.assertIsNone(glass.widget_hit(s, 128, 36, "rec", dx, dy))   # the clock side is not a button

    def test_picker_chips(self):
        names = ["Work", "Personal", "Local"]
        L = glass.picker_layout(1.5, names)
        self.assertEqual(len(L["chips"]), 3)
        for i, (x0, y0, x1, y1) in enumerate(L["chips"]):
            self.assertEqual(glass.hit_chip(1.5, names, (x0 + x1) / 2, (y0 + y1) / 2), i)
        rx, ry = L["ring"]
        self.assertIsNone(glass.hit_chip(1.5, names, rx, ry))
        lefts = [c[0] for c in L["chips"]]
        self.assertEqual(lefts, sorted(lefts))
        img = glass.picker(1.5, names, "Work", 1, 14, 0.7)
        self.assertEqual(img.size, L["size"])


class TestBytes(unittest.TestCase):
    def test_premultiplied_bgra(self):
        img = Image.new("RGBA", (2, 1))
        img.putpixel((0, 0), (200, 100, 50, 128))
        img.putpixel((1, 0), (255, 255, 255, 3))                          # fainter than CLICK_ALPHA
        data = glass.premultiplied_bgra(img)
        self.assertEqual(len(data), 8)
        b, g, r, a = data[:4]
        self.assertEqual(a, 128)
        self.assertEqual((r, g, b), ((200 * 128 + 127) // 255, (100 * 128 + 127) // 255, (50 * 128 + 127) // 255))
        self.assertEqual(tuple(data[4:]), (0, 0, 0, 0))                   # left fully transparent: clicks pass

    def test_never_exceeds_alpha(self):
        img = glass.widget(1.5, 116, 34, "working", heights=glass.working_heights(1.0))
        px = np.frombuffer(glass.premultiplied_bgra(img), np.uint8).reshape(-1, 4)
        self.assertTrue((px[:, :3].max(axis=1) <= px[:, 3]).all())


if __name__ == "__main__":
    unittest.main()
